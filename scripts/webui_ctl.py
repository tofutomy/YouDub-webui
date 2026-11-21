"""YouDub WebUI 前后端统一启停控制脚本。

支持一键后台启动/停止 FastAPI 后端与 Next.js 前端，提供健康检查、
进程树回收、端口归属校验、状态查看以及浏览器自动打开能力。
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil  # type: ignore[import-untyped]
except ImportError:
    psutil = None

# 项目根目录及运行态文件路径
REPO_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = REPO_ROOT / "apps" / "web"
DATA_DIR = REPO_ROOT / "data"
RUN_DIR = DATA_DIR / "run"
LOG_DIR = DATA_DIR / "logs"
STATE_PATH = RUN_DIR / "webui_runtime.json"
BACKEND_LOG_PATH = LOG_DIR / "webui-backend.log"
FRONTEND_LOG_PATH = LOG_DIR / "webui-frontend.log"

# 默认监听地址与端口（支持通过环境变量覆盖）
DEFAULT_BACKEND_HOST = os.getenv("YOUDUB_BACKEND_HOST", "0.0.0.0")
DEFAULT_BACKEND_PORT = int(os.getenv("YOUDUB_BACKEND_PORT", "8000"))
DEFAULT_FRONTEND_HOST = os.getenv("YOUDUB_FRONTEND_HOST", "0.0.0.0")
DEFAULT_FRONTEND_PORT = int(os.getenv("YOUDUB_FRONTEND_PORT", "3080"))

DEFAULT_BACKEND_TIMEOUT = 90.0
DEFAULT_FRONTEND_TIMEOUT = 60.0
DEFAULT_STOP_TIMEOUT = 15.0


class ControlError(RuntimeError):
    """启停控制过程中的可预期错误，用于向用户展示清晰提示。"""


def _norm_path(value: str | Path) -> str:
    """将路径规范化为适合跨大小写比较的绝对路径字符串。"""
    try:
        resolved = str(Path(value).resolve())
    except OSError:
        resolved = os.path.abspath(str(value))
    return os.path.normcase(resolved)


def _now_iso() -> str:
    """返回本地时区的 ISO 时间字符串。"""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _probe_host(bind_host: str) -> str:
    """把通配监听地址转换为本机可访问的探测地址。"""
    if bind_host in {"", "0.0.0.0", "::", "[::]"}:
        return "127.0.0.1"
    return bind_host


def _browser_host(bind_host: str) -> str:
    """将通配监听地址转换为适合浏览器打开的本机访问域名（默认 localhost）。"""
    if bind_host in {"", "0.0.0.0", "::", "[::]"}:
        return "localhost"
    return bind_host


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """检查目标主机与端口是否处于 TCP 监听状态。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_get(
    host: str,
    port: int,
    path: str,
    *,
    timeout: float = 3.0,
    max_bytes: int = 65536,
) -> tuple[int, bytes] | None:
    """直连本机 HTTP 服务（不走系统代理），返回状态码与响应体片段。"""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request("GET", path, headers={"Host": f"{host}:{port}"})
        resp = conn.getresponse()
        body = resp.read(max_bytes)
        return resp.status, body
    except (OSError, http.client.HTTPException):
        return None
    finally:
        conn.close()


def is_youdub_backend(port: int, host: str = "127.0.0.1") -> bool:
    """判断指定端口上是否运行着 YouDub FastAPI 后端。"""
    health = _http_get(host, port, "/api/health", timeout=2.0)
    if not health or health[0] != 200:
        return False
    try:
        payload = json.loads(health[1].decode("utf-8"))
    except (ValueError, UnicodeError):
        return False
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        return False

    # 进一步校验 OpenAPI 标题或任务接口，避免误判同端口的其他 FastAPI 服务
    openapi = _http_get(host, port, "/openapi.json", timeout=2.0, max_bytes=16384)
    if openapi and openapi[0] == 200:
        text = openapi[1].decode("utf-8", errors="ignore")
        if '"title":"YouDub API"' in text or '"title": "YouDub API"' in text:
            return True
        if '"openapi"' in text:
            return False
    tasks = _http_get(host, port, "/api/tasks?limit=1", timeout=2.0, max_bytes=4096)
    return bool(tasks and tasks[0] == 200)


def is_youdub_frontend(port: int, host: str = "127.0.0.1", *, timeout: float = 4.0) -> bool:
    """判断指定端口上是否运行着 YouDub Next.js 前端。"""
    root = _http_get(host, port, "/", timeout=timeout, max_bytes=65536)
    if root and root[0] == 200:
        html = root[1].decode("utf-8", errors="ignore")
        if "YouDub" in html or "__next" in html or "/_next/" in html:
            return True
    # 兼容首页尚在首次编译时，通过代理路由确认前端已就绪
    proxied = _http_get(host, port, "/api/health", timeout=2.0)
    if proxied and proxied[0] == 200:
        try:
            payload = json.loads(proxied[1].decode("utf-8"))
            if isinstance(payload, dict) and payload.get("status") == "ok":
                return True
        except (ValueError, UnicodeError):
            pass
    return False


def load_state(state_path: Path = STATE_PATH) -> dict[str, Any]:
    """读取运行态状态文件；文件缺失或损坏时返回空字典。"""
    if not state_path.is_file():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def save_state(state: dict[str, Any], state_path: Path = STATE_PATH) -> None:
    """原子写入运行态状态文件；若无任何记录则删除状态文件。"""
    cleaned = {k: v for k, v in state.items() if v}
    if not cleaned:
        try:
            state_path.unlink(missing_ok=True)
        except OSError:
            pass
        return
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(cleaned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(state_path)


def _backend_python(repo_root: Path = REPO_ROOT) -> Path:
    """定位 YouDub 虚拟环境中的 Python 解释器。"""
    rel = Path(".venv") / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python_path = repo_root / rel
    if not python_path.is_file():
        raise ControlError(f"未找到后端 Python 虚拟环境：{python_path}")
    return python_path


def build_backend_command(
    *,
    host: str,
    port: int,
    reload: bool = False,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    """构造 FastAPI 后端启动命令。"""
    python_path = _backend_python(repo_root)
    cmd = [
        str(python_path),
        "-m",
        "uvicorn",
        "backend.app.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    return cmd


def build_frontend_command(
    *,
    host: str,
    port: int,
    repo_root: Path = REPO_ROOT,
) -> tuple[list[str], Path]:
    """构造 Next.js 前端启动命令及其工作目录。

    优先使用 ``node`` 直接执行 ``apps/web/node_modules/next/dist/bin/next``，
    避免 Windows 下 ``npm.cmd`` 引入多层 ``cmd.exe`` 包装进程，提升启停稳定性；
    若找不到局部入口则回退到 ``npm.cmd`` / ``npm``。
    """
    web_dir = repo_root / "apps" / "web"
    node_modules = web_dir / "node_modules"
    if not node_modules.is_dir():
        raise ControlError(
            f"未找到前端依赖目录：{node_modules}。请先执行 npm.cmd --prefix apps/web install"
        )

    next_bin = node_modules / "next" / "dist" / "bin" / "next"
    node_exe = shutil.which("node.exe" if os.name == "nt" else "node") or shutil.which("node")
    if node_exe and next_bin.is_file():
        return (
            [node_exe, str(next_bin), "dev", "--hostname", host, "--port", str(port)],
            web_dir,
        )

    npm_exe = (
        shutil.which("npm.cmd") if os.name == "nt" else None
    ) or shutil.which("npm")
    if not npm_exe:
        raise ControlError("未找到 node 或 npm，请先安装 Node.js 20+ 并加入 PATH。")
    return (
        [
            npm_exe,
            "--prefix",
            str(web_dir),
            "run",
            "dev",
            "--",
            "--hostname",
            host,
            "--port",
            str(port),
        ],
        repo_root,
    )


def _cleanup_stale_next_lock(repo_root: Path = REPO_ROOT) -> None:
    """在确认没有前端运行时，尝试清理残留的 Next.js dev 锁文件。"""
    lock_file = repo_root / "apps" / "web" / ".next" / "dev" / "lock"
    if not lock_file.is_file():
        return
    try:
        lock_file.unlink()
    except OSError:
        # 若锁文件正被其他活跃进程持有则保持不动
        pass


def _read_log_tail(log_path: Path, offset: int = 0, max_lines: int = 25) -> str:
    """读取日志文件中自指定偏移量之后的末尾若干行，便于排错展示。"""
    if not log_path.is_file():
        return ""
    try:
        with log_path.open("rb") as fh:
            fh.seek(max(0, offset))
            raw = fh.read()
    except OSError:
        return ""
    lines = raw.decode("utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:]).strip()


def _spawn_background(
    cmd: list[str],
    *,
    cwd: Path,
    log_path: Path,
    banner: str,
    env: dict[str, str] | None = None,
) -> tuple[subprocess.Popen[bytes], int]:
    """以后台无窗口方式启动子进程，并将输出追加到日志文件。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_file:
        header = f"\n===== [{_now_iso()}] {banner} =====\n".encode("utf-8")
        log_file.write(header)
        log_file.flush()
        offset = log_file.tell()

        kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "env": env,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            )
        else:
            kwargs["start_new_session"] = True
        proc = subprocess.Popen(cmd, **kwargs)
    return proc, offset


def _process_create_time(pid: int) -> float | None:
    """获取进程创建时间戳，用于防止停止时误杀被操作系统复用的 PID。"""
    if psutil is None:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except (psutil.Error, OSError):
        return None


def _process_is_alive(pid: int, expected_create_time: float | None = None) -> bool:
    """检查指定 PID 是否存活；若提供创建时间则同时校验是否为同一进程实例。"""
    if pid <= 0:
        return False
    if psutil is not None:
        try:
            proc = psutil.Process(pid)
            if not proc.is_running() or proc.status() == psutil.STATUS_ZOMBIE:
                return False
            if expected_create_time is not None:
                if abs(float(proc.create_time()) - expected_create_time) > 2.0:
                    return False
            return True
        except (psutil.Error, OSError):
            return False

    # 无 psutil 时的回退检查
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _process_matches_service(
    pid: int,
    service: str,
    repo_root: Path = REPO_ROOT,
) -> bool:
    """判断指定 PID 是否属于当前仓库的 YouDub 后端或前端进程。"""
    if psutil is None:
        return False
    try:
        proc = psutil.Process(pid)
        cmdline_list = proc.cmdline()
        cmdline = " ".join(cmdline_list).lower()
        exe = _norm_path(proc.exe()) if proc.exe() else ""
        try:
            cwd = _norm_path(proc.cwd())
        except (psutil.Error, OSError):
            cwd = ""
    except (psutil.Error, OSError):
        return False

    root_norm = _norm_path(repo_root)
    web_norm = _norm_path(repo_root / "apps" / "web")
    cmdline_norm = os.path.normcase(cmdline.replace("/", os.sep))
    in_repo = (
        (bool(cwd) and (cwd == root_norm or cwd.startswith(root_norm + os.sep)))
        or (bool(exe) and exe.startswith(root_norm + os.sep))
        or (root_norm in cmdline_norm)
    )

    if service == "backend":
        if "backend.app.main:app" in cmdline:
            return in_repo or "uvicorn" in cmdline
        # 兼容 uvicorn --reload 在 Windows 下派生出的 multiprocessing.spawn 工作进程
        if in_repo and ("uvicorn" in cmdline or "multiprocessing.spawn" in cmdline):
            return True
        return False

    if service == "frontend":
        if web_norm in cmdline_norm or (cwd == web_norm):
            return any(token in cmdline for token in ("next", "npm", "node"))
        if in_repo and ("apps" + os.sep + "web" in cmdline_norm or "apps/web" in cmdline):
            return any(token in cmdline for token in ("next", "npm", "node"))
        return False

    return False


def _find_service_root_pid(
    pid: int,
    service: str,
    repo_root: Path = REPO_ROOT,
) -> int | None:
    """从监听端口的进程向上回溯，找到所属的最高层 YouDub 服务根进程。

    这样可以一次性终止例如 ``npm.cmd -> node -> next-server`` 或
    ``uvicorn reloader -> worker`` 的整棵进程树，而绝不会向上越界到用户终端或
    LootGameVideo 等外部进程。
    """
    if psutil is None:
        return None
    try:
        current = psutil.Process(pid)
    except (psutil.Error, OSError):
        return None

    matched: list[int] = []
    visited: set[int] = set()
    depth = 0
    while current is not None and depth < 6:
        cur_pid = current.pid
        if cur_pid in visited or cur_pid <= 4:
            break
        visited.add(cur_pid)
        if _process_matches_service(cur_pid, service, repo_root):
            matched.append(cur_pid)
        elif matched:
            # 一旦脱离当前服务链立即停止，防止误伤父级 shell 或调用方工作台
            break
        try:
            current = current.parent()
        except (psutil.Error, OSError):
            break
        depth += 1

    return matched[-1] if matched else None


def _listening_pids_on_port(port: int) -> list[int]:
    """查询当前正在监听指定 TCP 端口的所有进程 PID。"""
    pids: list[int] = []
    if psutil is not None:
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if (
                    conn.status == psutil.CONN_LISTEN
                    and conn.laddr
                    and conn.laddr.port == port
                    and conn.pid
                ):
                    if conn.pid not in pids:
                        pids.append(int(conn.pid))
            return pids
        except (psutil.Error, OSError):
            pass

    # 回退到 Windows netstat 解析
    if os.name == "nt":
        result = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        suffix = f":{port}"
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[3].upper() == "LISTENING" and parts[1].endswith(suffix):
                try:
                    pid = int(parts[4])
                except ValueError:
                    continue
                if pid > 0 and pid not in pids:
                    pids.append(pid)
    return pids


def _terminate_process_tree(pid: int, timeout: float = 5.0) -> bool:
    """优雅终止并按需强制清理指定 PID 及其全部子进程。"""
    if pid <= 0 or pid == os.getpid():
        return False

    if psutil is not None:
        try:
            root = psutil.Process(pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

        try:
            children = root.children(recursive=True)
        except (psutil.Error, OSError):
            children = []
        # 先向根进程和全部子进程发送终止信号，防止守护/热重载父进程拉起新子进程
        targets = [root, *children]
        for proc in targets:
            try:
                proc.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        _, alive = psutil.wait_procs(targets, timeout=timeout)
        if alive:
            for proc in alive:
                try:
                    proc.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            psutil.wait_procs(alive, timeout=3.0)
        return True

    if os.name == "nt":
        result = subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(pid)],
            capture_output=True,
            text=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return result.returncode == 0

    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
    except OSError:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _process_is_alive(pid):
            return True
        time.sleep(0.2)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except OSError:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return True


def _wait_for_ready(
    *,
    label: str,
    check_fn: Any,
    proc: subprocess.Popen[bytes],
    log_path: Path,
    log_offset: int,
    timeout: float,
) -> None:
    """轮询等待新启动的服务就绪，若进程提前退出或超时则抛出带日志尾部的错误。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check_fn():
            return
        exit_code = proc.poll()
        if exit_code is not None:
            tail = _read_log_tail(log_path, log_offset)
            detail = f"\n--- 最近日志 ({log_path}) ---\n{tail}" if tail else f"（详见 {log_path}）"
            raise ControlError(f"{label}进程提前退出（退出码 {exit_code}）。{detail}")
        time.sleep(0.5)

    tail = _read_log_tail(log_path, log_offset)
    detail = f"\n--- 最近日志 ({log_path}) ---\n{tail}" if tail else f"（详见 {log_path}）"
    raise ControlError(f"{label}启动超时（超过 {int(timeout)} 秒未就绪）。{detail}")


VALID_TARGETS = ("all", "backend", "frontend")


def _validate_target(target: str) -> str:
    """校验服务目标参数，仅允许 all / backend / frontend。"""
    if target not in VALID_TARGETS:
        raise ControlError(f"无效的目标服务：{target}（可选：all / backend / frontend）")
    return target


def start_services(
    *,
    target: str = "all",
    backend_host: str = DEFAULT_BACKEND_HOST,
    backend_port: int | None = None,
    frontend_host: str = DEFAULT_FRONTEND_HOST,
    frontend_port: int = DEFAULT_FRONTEND_PORT,
    reload: bool = False,
    open_browser: bool = False,
    backend_timeout: float = DEFAULT_BACKEND_TIMEOUT,
    frontend_timeout: float = DEFAULT_FRONTEND_TIMEOUT,
    repo_root: Path = REPO_ROOT,
    state_path: Path = STATE_PATH,
    backend_log_path: Path = BACKEND_LOG_PATH,
    frontend_log_path: Path = FRONTEND_LOG_PATH,
) -> dict[str, Any]:
    """按目标启动 YouDub 后端和/或前端，如已运行则直接复用。"""
    _validate_target(target)
    run_backend = target in {"all", "backend"}
    run_frontend = target in {"all", "frontend"}

    state = load_state(state_path)
    # 单独启动前端且未显式指定后端端口时，优先沿用状态文件中活跃后端的端口
    if backend_port is not None:
        effective_backend_port = backend_port
    else:
        b_entry = state.get("backend") if isinstance(state.get("backend"), dict) else {}
        if (
            not run_backend
            and isinstance(b_entry.get("port"), int)
            and isinstance(b_entry.get("pid"), int)
            and _process_is_alive(b_entry["pid"], b_entry.get("create_time"))
        ):
            effective_backend_port = b_entry["port"]
        else:
            effective_backend_port = DEFAULT_BACKEND_PORT

    if run_backend and run_frontend and effective_backend_port == frontend_port:
        raise ControlError(f"后端端口与前端端口不能相同（均为 {effective_backend_port}）。")

    backend_probe = _probe_host(backend_host)
    frontend_probe = _probe_host(frontend_host)
    backend_url = f"http://{backend_probe}:{effective_backend_port}"
    frontend_url = f"http://{_browser_host(frontend_host)}:{frontend_port}"

    result: dict[str, Any] = {"target": target}
    backend_prefix = "[1/2] " if target == "all" else ""
    frontend_prefix = "[2/2] " if target == "all" else ""

    # 1. 启动或复用后端
    if run_backend:
        result["backend_url"] = backend_url
        result["backend_reused"] = False
        if is_youdub_backend(effective_backend_port, backend_probe):
            print(f"{backend_prefix}后端已在运行：{backend_url}（直接复用）", flush=True)
            result["backend_reused"] = True
        elif _is_port_open(backend_probe, effective_backend_port):
            raise ControlError(
                f"端口 {effective_backend_port} 已被其他非 YouDub 程序占用，无法启动后端。"
            )
        else:
            print(f"{backend_prefix}正在启动后端 ({backend_url})...", flush=True)
            backend_cmd = build_backend_command(
                host=backend_host,
                port=effective_backend_port,
                reload=reload,
                repo_root=repo_root,
            )
            backend_env = os.environ.copy()
            backend_env.setdefault("PYTHONUTF8", "1")
            backend_proc, backend_offset = _spawn_background(
                backend_cmd,
                cwd=repo_root,
                log_path=backend_log_path,
                banner=f"启动 YouDub 后端 ({backend_host}:{effective_backend_port})",
                env=backend_env,
            )
            state["backend"] = {
                "pid": backend_proc.pid,
                "create_time": _process_create_time(backend_proc.pid),
                "host": backend_host,
                "port": effective_backend_port,
                "started_at": _now_iso(),
                "log": str(backend_log_path),
            }
            save_state(state, state_path)
            try:
                _wait_for_ready(
                    label="后端",
                    check_fn=lambda: is_youdub_backend(effective_backend_port, backend_probe),
                    proc=backend_proc,
                    log_path=backend_log_path,
                    log_offset=backend_offset,
                    timeout=backend_timeout,
                )
            except Exception:
                # 启动失败时回收刚刚拉起的残留进程并清理状态
                _terminate_process_tree(backend_proc.pid)
                state.pop("backend", None)
                save_state(state, state_path)
                raise
            print(f"{backend_prefix}后端已就绪：{backend_url} (PID {backend_proc.pid})", flush=True)
            result["backend_pid"] = backend_proc.pid

    # 2. 启动或复用前端
    if run_frontend:
        result["frontend_url"] = frontend_url
        result["frontend_reused"] = False
        if is_youdub_frontend(frontend_port, frontend_probe, timeout=3.0):
            print(f"{frontend_prefix}前端已在运行：{frontend_url}（直接复用）", flush=True)
            result["frontend_reused"] = True
        elif _is_port_open(frontend_probe, frontend_port):
            raise ControlError(
                f"端口 {frontend_port} 已被其他非 YouDub 程序占用，无法启动前端。"
            )
        else:
            print(f"{frontend_prefix}正在启动前端 ({frontend_url})...", flush=True)
            _cleanup_stale_next_lock(repo_root)
            frontend_cmd, frontend_cwd = build_frontend_command(
                host=frontend_host,
                port=frontend_port,
                repo_root=repo_root,
            )
            frontend_env = os.environ.copy()
            # 确保前端 /api/* 代理目标与当前后端端口保持一致
            frontend_env.setdefault("NEXT_SERVER_API_BASE_URL", backend_url)
            frontend_proc, frontend_offset = _spawn_background(
                frontend_cmd,
                cwd=frontend_cwd,
                log_path=frontend_log_path,
                banner=f"启动 YouDub 前端 ({frontend_host}:{frontend_port})",
                env=frontend_env,
            )
            state["frontend"] = {
                "pid": frontend_proc.pid,
                "create_time": _process_create_time(frontend_proc.pid),
                "host": frontend_host,
                "port": frontend_port,
                "started_at": _now_iso(),
                "log": str(frontend_log_path),
            }
            save_state(state, state_path)
            try:
                _wait_for_ready(
                    label="前端",
                    check_fn=lambda: is_youdub_frontend(frontend_port, frontend_probe, timeout=5.0),
                    proc=frontend_proc,
                    log_path=frontend_log_path,
                    log_offset=frontend_offset,
                    timeout=frontend_timeout,
                )
            except Exception:
                _terminate_process_tree(frontend_proc.pid)
                state.pop("frontend", None)
                save_state(state, state_path)
                raise
            print(f"{frontend_prefix}前端已就绪：{frontend_url} (PID {frontend_proc.pid})", flush=True)
            result["frontend_pid"] = frontend_proc.pid

        if not run_backend and not is_youdub_backend(effective_backend_port, backend_probe):
            print(
                f"提示：当前未检测到后端 ({backend_url})，如需加载或运行任务请先双击 Start-YouDub-Backend.cmd。",
                flush=True,
            )

    if target == "backend":
        print(f"\nYouDub 后端已启动：{backend_url}", flush=True)
        print(f"后端日志：{backend_log_path}", flush=True)
        print("如需停止后端，请双击 Stop-YouDub-Backend.cmd。", flush=True)
    elif target == "frontend":
        print(f"\nYouDub 前端已启动：{frontend_url}", flush=True)
        print(f"前端日志：{frontend_log_path}", flush=True)
        print("如需停止前端，请双击 Stop-YouDub-Frontend.cmd。", flush=True)
    else:
        print(f"\nYouDub WebUI 已启动：{frontend_url}", flush=True)
        print(f"后端日志：{backend_log_path}", flush=True)
        print(f"前端日志：{frontend_log_path}", flush=True)
        print("如需停止服务，请双击 Stop-YouDub-Frontend.cmd / Stop-YouDub-Backend.cmd。", flush=True)

    if open_browser and run_frontend:
        webbrowser.open(frontend_url)
    return result


def _stop_one_service(
    service: str,
    label: str,
    *,
    port: int | None,
    default_port: int,
    state: dict[str, Any],
    timeout: float,
    repo_root: Path = REPO_ROOT,
) -> str:
    """停止单个服务（优先按记录的 PID，再按端口兜底校验归属）。

    返回状态字符串：``stopped``、``not_running`` 或 ``external``。
    """
    stopped_any = False
    entry = state.get(service)
    recorded_port = entry.get("port") if isinstance(entry, dict) and isinstance(entry.get("port"), int) and entry.get("port") > 0 else None
    effective_port = port if port is not None else (recorded_port or default_port)
    entry_matches_port = port is None or recorded_port is None or recorded_port == port

    if isinstance(entry, dict) and entry_matches_port:
        recorded_pid = entry.get("pid")
        recorded_ctime = entry.get("create_time")
        if isinstance(recorded_pid, int) and _process_is_alive(recorded_pid, recorded_ctime):
            # 有 psutil 时再校验归属，彻底避免误杀无关进程
            if psutil is None or _process_matches_service(recorded_pid, service, repo_root):
                if _terminate_process_tree(recorded_pid):
                    stopped_any = True

    # 检查端口是否仍在监听（处理手动启动、LootGameVideo 拉起或状态文件丢失的情况）
    listening_pids = _listening_pids_on_port(effective_port)
    external_pids: list[int] = []
    if listening_pids:
        targets: list[int] = []
        for lpid in listening_pids:
            root_pid = _find_service_root_pid(lpid, service, repo_root)
            if root_pid is not None:
                if root_pid not in targets:
                    targets.append(root_pid)
            else:
                external_pids.append(lpid)
        for tpid in targets:
            if _terminate_process_tree(tpid):
                stopped_any = True

    # 等待端口释放
    if stopped_any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _is_port_open("127.0.0.1", effective_port, timeout=0.3):
                break
            time.sleep(0.25)
        if _is_port_open("127.0.0.1", effective_port, timeout=0.3):
            raise ControlError(f"{label}（端口 {effective_port}）停止超时，端口仍被占用。")
        if entry_matches_port:
            state.pop(service, None)
        print(f"{label}（端口 {effective_port}）已停止。", flush=True)
        return "stopped"

    if entry_matches_port:
        state.pop(service, None)
    if external_pids:
        pid_text = ", ".join(str(p) for p in external_pids)
        print(
            f"警告：端口 {effective_port} 被其他非 YouDub 进程占用（PID: {pid_text}），已跳过未终止。",
            flush=True,
        )
        return "external"

    print(f"{label}（端口 {effective_port}）未运行。", flush=True)
    return "not_running"


def stop_services(
    *,
    target: str = "all",
    backend_port: int | None = None,
    frontend_port: int | None = None,
    timeout: float = DEFAULT_STOP_TIMEOUT,
    repo_root: Path = REPO_ROOT,
    state_path: Path = STATE_PATH,
) -> dict[str, str]:
    """按目标停止 YouDub 前端和/或后端，并更新运行态状态文件。"""
    _validate_target(target)
    state = load_state(state_path)
    result: dict[str, str] = {}
    if target in {"all", "frontend"}:
        result["frontend"] = _stop_one_service(
            "frontend",
            "前端",
            port=frontend_port,
            default_port=DEFAULT_FRONTEND_PORT,
            state=state,
            timeout=timeout,
            repo_root=repo_root,
        )
    if target in {"all", "backend"}:
        result["backend"] = _stop_one_service(
            "backend",
            "后端",
            port=backend_port,
            default_port=DEFAULT_BACKEND_PORT,
            state=state,
            timeout=timeout,
            repo_root=repo_root,
        )
    save_state(state, state_path)
    return result


def status_services(
    *,
    target: str = "all",
    backend_port: int | None = None,
    frontend_port: int | None = None,
    state_path: Path = STATE_PATH,
) -> dict[str, Any]:
    """查询并输出当前 YouDub 后端和/或前端的运行状态。"""
    _validate_target(target)
    state = load_state(state_path)
    result: dict[str, Any] = {}

    if target in {"all", "backend"}:
        b_entry = state.get("backend") if isinstance(state.get("backend"), dict) else {}
        effective_backend_port = (
            backend_port
            if backend_port is not None
            else (b_entry.get("port") if isinstance(b_entry.get("port"), int) else DEFAULT_BACKEND_PORT)
        )
        backend_running = is_youdub_backend(effective_backend_port)
        b_pid = (
            b_entry.get("pid")
            if b_entry.get("port") == effective_backend_port
            and isinstance(b_entry.get("pid"), int)
            and _process_is_alive(b_entry["pid"], b_entry.get("create_time"))
            else None
        )
        b_text = (
            f"运行中 (http://127.0.0.1:{effective_backend_port}" + (f", PID {b_pid})" if b_pid else ")")
            if backend_running
            else f"未运行 (端口 {effective_backend_port})"
        )
        print(f"后端：{b_text}", flush=True)
        result["backend"] = {"running": backend_running, "port": effective_backend_port, "pid": b_pid}

    if target in {"all", "frontend"}:
        f_entry = state.get("frontend") if isinstance(state.get("frontend"), dict) else {}
        effective_frontend_port = (
            frontend_port
            if frontend_port is not None
            else (f_entry.get("port") if isinstance(f_entry.get("port"), int) else DEFAULT_FRONTEND_PORT)
        )
        frontend_running = is_youdub_frontend(effective_frontend_port, timeout=3.0)
        f_pid = (
            f_entry.get("pid")
            if f_entry.get("port") == effective_frontend_port
            and isinstance(f_entry.get("pid"), int)
            and _process_is_alive(f_entry["pid"], f_entry.get("create_time"))
            else None
        )
        f_text = (
            f"运行中 (http://127.0.0.1:{effective_frontend_port}" + (f", PID {f_pid})" if f_pid else ")")
            if frontend_running
            else f"未运行 (端口 {effective_frontend_port})"
        )
        print(f"前端：{f_text}", flush=True)
        result["frontend"] = {"running": frontend_running, "port": effective_frontend_port, "pid": f_pid}

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="YouDub WebUI 前后端启停工具")
    sub = parser.add_subparsers(dest="command", required=True)

    def _add_common_args(p: argparse.ArgumentParser, *, frontend_none_default: bool = False) -> None:
        p.add_argument(
            "--target",
            choices=VALID_TARGETS,
            default="all",
            help="操作目标：all（前后端）、backend（仅后端）、frontend（仅前端），默认 all",
        )
        p.add_argument(
            "--backend-port",
            type=int,
            default=None,
            help=f"后端端口（默认 {DEFAULT_BACKEND_PORT}）",
        )
        p.add_argument(
            "--frontend-port",
            type=int,
            default=None if frontend_none_default else DEFAULT_FRONTEND_PORT,
            help=f"前端端口（默认 {DEFAULT_FRONTEND_PORT}）",
        )

    for name in ("start", "restart"):
        cmd_parser = sub.add_parser(name, help="启动服务" if name == "start" else "重启服务")
        _add_common_args(cmd_parser, frontend_none_default=False)
        cmd_parser.add_argument("--backend-host", default=DEFAULT_BACKEND_HOST, help=f"后端监听地址（默认 {DEFAULT_BACKEND_HOST}）")
        cmd_parser.add_argument("--frontend-host", default=DEFAULT_FRONTEND_HOST, help=f"前端监听地址（默认 {DEFAULT_FRONTEND_HOST}）")
        cmd_parser.add_argument("--reload", action="store_true", help="后端启用 uvicorn --reload 热重载")
        cmd_parser.add_argument(
            "--open",
            dest="open_browser",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="启动前端完成后自动在浏览器中打开页面（可用 --no-open 关闭）",
        )
        cmd_parser.add_argument("--backend-timeout", type=float, default=DEFAULT_BACKEND_TIMEOUT, help="等待后端就绪超时秒数")
        cmd_parser.add_argument("--frontend-timeout", type=float, default=DEFAULT_FRONTEND_TIMEOUT, help="等待前端就绪超时秒数")

    stop_parser = sub.add_parser("stop", help="停止服务")
    _add_common_args(stop_parser, frontend_none_default=True)
    stop_parser.add_argument("--timeout", type=float, default=DEFAULT_STOP_TIMEOUT, help="等待端口释放超时秒数")

    status_parser = sub.add_parser("status", help="查看服务运行状态")
    _add_common_args(status_parser, frontend_none_default=True)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI 主入口。"""
    args = parse_args(argv)
    try:
        if args.command == "start":
            start_services(
                target=args.target,
                backend_host=args.backend_host,
                backend_port=args.backend_port,
                frontend_host=args.frontend_host,
                frontend_port=args.frontend_port,
                reload=args.reload,
                open_browser=args.open_browser,
                backend_timeout=args.backend_timeout,
                frontend_timeout=args.frontend_timeout,
            )
        elif args.command == "stop":
            stop_services(
                target=args.target,
                backend_port=args.backend_port,
                frontend_port=args.frontend_port,
                timeout=args.timeout,
            )
        elif args.command == "restart":
            stop_services(
                target=args.target,
                backend_port=args.backend_port,
                frontend_port=args.frontend_port,
            )
            start_services(
                target=args.target,
                backend_host=args.backend_host,
                backend_port=args.backend_port,
                frontend_host=args.frontend_host,
                frontend_port=args.frontend_port,
                reload=args.reload,
                open_browser=args.open_browser,
                backend_timeout=args.backend_timeout,
                frontend_timeout=args.frontend_timeout,
            )
        elif args.command == "status":
            status_services(
                target=args.target,
                backend_port=args.backend_port,
                frontend_port=args.frontend_port,
            )
    except ControlError as error:
        print(f"错误：{error}", file=sys.stderr, flush=True)
        return 1
    except KeyboardInterrupt:
        print("\n操作已取消。", file=sys.stderr, flush=True)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
