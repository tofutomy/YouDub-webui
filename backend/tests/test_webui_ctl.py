"""YouDub WebUI 前后端统一启停脚本单元测试。"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from scripts import webui_ctl


def _start_mock_http_server(routes: dict[str, tuple[int, bytes, str]]):
    """在本地临时端口启动一个轻量级模拟 HTTP 服务，供健康检查测试使用。"""

    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

        def do_GET(self) -> None:
            item = routes.get(self.path)
            if item is None:
                self.send_response(404)
                self.end_headers()
                return
            status, body, content_type = item
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _make_fake_repo(tmp_path: Path, *, with_next_bin: bool = True) -> Path:
    """在临时目录中构造最小化的 YouDub 仓库结构。"""
    py = tmp_path / ".venv" / ("Scripts/python.exe" if webui_ctl.os.name == "nt" else "bin/python")
    py.parent.mkdir(parents=True, exist_ok=True)
    py.write_text("", encoding="utf-8")

    node_modules = tmp_path / "apps" / "web" / "node_modules"
    node_modules.mkdir(parents=True, exist_ok=True)
    if with_next_bin:
        next_bin = node_modules / "next" / "dist" / "bin" / "next"
        next_bin.parent.mkdir(parents=True, exist_ok=True)
        next_bin.write_text("// fake next", encoding="utf-8")
    return tmp_path


def test_build_backend_command_default_and_reload(tmp_path: Path):
    """验证后端命令构造在默认模式和热重载模式下的参数正确性。"""
    repo = _make_fake_repo(tmp_path)
    cmd = webui_ctl.build_backend_command(host="0.0.0.0", port=8000, repo_root=repo)
    assert cmd[1:] == ["-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

    reload_cmd = webui_ctl.build_backend_command(
        host="127.0.0.1", port=8010, reload=True, repo_root=repo
    )
    assert reload_cmd[-1] == "--reload"
    assert "--port" in reload_cmd and "8010" in reload_cmd


def test_build_frontend_command_prefers_node_next_and_falls_back_to_npm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证前端命令优先直接调用 node + next，缺失时回退到 npm。"""
    repo = _make_fake_repo(tmp_path, with_next_bin=True)
    monkeypatch.setattr(webui_ctl.shutil, "which", lambda name: "/usr/bin/node" if "node" in name else None)
    cmd, cwd = webui_ctl.build_frontend_command(host="0.0.0.0", port=3080, repo_root=repo)
    assert cmd[0] == "/usr/bin/node"
    assert cmd[1].endswith("next")
    assert cmd[2:] == ["dev", "--hostname", "0.0.0.0", "--port", "3080"]
    assert cwd == repo / "apps" / "web"

    # 无本地 next 入口时回退到 npm
    repo_fallback = _make_fake_repo(tmp_path / "fb", with_next_bin=False)
    monkeypatch.setattr(webui_ctl.shutil, "which", lambda name: "/usr/bin/npm" if "npm" in name else None)
    fb_cmd, fb_cwd = webui_ctl.build_frontend_command(host="0.0.0.0", port=3080, repo_root=repo_fallback)
    assert fb_cmd[0] == "/usr/bin/npm"
    assert "--prefix" in fb_cmd
    assert fb_cmd[-4:] == ["--hostname", "0.0.0.0", "--port", "3080"]
    assert fb_cwd == repo_fallback


def test_health_probes_distinguish_youdub_and_other_services():
    """验证后端与前端健康探针能正确识别 YouDub 并拒绝同端口的其他服务。"""
    youdub_backend, b_port = _start_mock_http_server(
        {
            "/api/health": (200, b'{"status":"ok"}', "application/json"),
            "/openapi.json": (200, b'{"openapi":"3.1.0","info":{"title":"YouDub API"}}', "application/json"),
        }
    )
    other_backend, o_port = _start_mock_http_server(
        {
            "/api/health": (200, b'{"status":"ok"}', "application/json"),
            "/openapi.json": (200, b'{"openapi":"3.1.0","info":{"title":"Other Service"}}', "application/json"),
        }
    )
    youdub_frontend, f_port = _start_mock_http_server(
        {
            "/": (200, b"<html><head><title>YouDub</title></head><body></body></html>", "text/html"),
        }
    )
    other_frontend, of_port = _start_mock_http_server(
        {
            "/": (200, b"<html><head><title>Something Else</title></head></html>", "text/html"),
        }
    )
    try:
        assert webui_ctl.is_youdub_backend(b_port) is True
        assert webui_ctl.is_youdub_backend(o_port) is False
        assert webui_ctl.is_youdub_frontend(f_port) is True
        assert webui_ctl.is_youdub_frontend(of_port) is False
    finally:
        youdub_backend.shutdown()
        other_backend.shutdown()
        youdub_frontend.shutdown()
        other_frontend.shutdown()


def test_start_services_reuses_running_services_and_opens_browser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证前后端均已运行时不重复拉起进程，直接复用并按需打开浏览器。"""
    youdub_backend, b_port = _start_mock_http_server(
        {
            "/api/health": (200, b'{"status":"ok"}', "application/json"),
            "/openapi.json": (200, b'{"openapi":"3.1.0","info":{"title":"YouDub API"}}', "application/json"),
        }
    )
    youdub_frontend, f_port = _start_mock_http_server(
        {
            "/": (200, b"<html><title>YouDub</title></html>", "text/html"),
        }
    )
    opened: list[str] = []
    spawned: list[Any] = []
    monkeypatch.setattr(webui_ctl.webbrowser, "open", lambda url: opened.append(url) or True)
    monkeypatch.setattr(webui_ctl, "_spawn_background", lambda *a, **kw: spawned.append(a))

    try:
        res = webui_ctl.start_services(
            backend_port=b_port,
            frontend_port=f_port,
            open_browser=True,
            repo_root=_make_fake_repo(tmp_path),
            state_path=tmp_path / "state.json",
            backend_log_path=tmp_path / "b.log",
            frontend_log_path=tmp_path / "f.log",
        )
        assert res["backend_reused"] is True
        assert res["frontend_reused"] is True
        assert spawned == []
        assert opened == [f"http://localhost:{f_port}"]
    finally:
        youdub_backend.shutdown()
        youdub_frontend.shutdown()


def test_start_services_rejects_port_occupied_by_other_program(tmp_path: Path):
    """验证目标端口被其他非 YouDub 程序占用时明确报错而不误覆盖。"""
    other, port = _start_mock_http_server({"/": (200, b"hello", "text/plain")})
    try:
        with pytest.raises(webui_ctl.ControlError, match="已被其他非 YouDub 程序占用"):
            webui_ctl.start_services(
                backend_port=port,
                frontend_port=port + 1,
                repo_root=_make_fake_repo(tmp_path),
                state_path=tmp_path / "state.json",
                backend_log_path=tmp_path / "b.log",
                frontend_log_path=tmp_path / "f.log",
            )
    finally:
        other.shutdown()


def test_start_services_spawns_and_persists_state_and_cleans_stale_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证正常启动流程：清理残留锁文件、拉起前后端、写入状态文件并等待就绪。"""
    repo = _make_fake_repo(tmp_path)
    lock_file = repo / "apps" / "web" / ".next" / "dev" / "lock"
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file.write_text("stale", encoding="utf-8")

    state_path = tmp_path / "run" / "webui_runtime.json"
    ready = {"backend": False, "frontend": False}
    spawned_calls: list[dict[str, Any]] = []

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> int | None:
            return None

    def _fake_spawn(cmd: list[str], *, cwd: Path, log_path: Path, banner: str, env: dict[str, str] | None = None):
        is_backend = "uvicorn" in cmd
        spawned_calls.append({"cmd": cmd, "cwd": cwd, "env": env})
        if is_backend:
            ready["backend"] = True
            return _FakeProc(11111), 0
        ready["frontend"] = True
        return _FakeProc(22222), 0

    monkeypatch.setattr(webui_ctl.shutil, "which", lambda name: "/usr/bin/node" if "node" in name else None)
    monkeypatch.setattr(webui_ctl, "is_youdub_backend", lambda port, host="127.0.0.1": ready["backend"])
    monkeypatch.setattr(webui_ctl, "is_youdub_frontend", lambda port, host="127.0.0.1", timeout=4.0: ready["frontend"])
    monkeypatch.setattr(webui_ctl, "_is_port_open", lambda host, port, timeout=0.5: False)
    monkeypatch.setattr(webui_ctl, "_spawn_background", _fake_spawn)
    monkeypatch.setattr(webui_ctl, "_process_create_time", lambda pid: 1000.0 + pid)

    res = webui_ctl.start_services(
        backend_port=18000,
        frontend_port=13080,
        open_browser=False,
        repo_root=repo,
        state_path=state_path,
        backend_log_path=tmp_path / "b.log",
        frontend_log_path=tmp_path / "f.log",
    )

    assert not lock_file.exists()
    assert res["backend_pid"] == 11111
    assert res["frontend_pid"] == 22222
    assert spawned_calls[1]["env"]["NEXT_SERVER_API_BASE_URL"] == "http://127.0.0.1:18000"

    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["backend"]["pid"] == 11111
    assert saved["backend"]["port"] == 18000
    assert saved["frontend"]["pid"] == 22222
    assert saved["frontend"]["port"] == 13080


def test_stop_services_terminates_tracked_and_skips_unrelated_port_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证停止流程会终止已记录的 YouDub 进程，并对非 YouDub 占用端口安全跳过。"""
    state_path = tmp_path / "webui_runtime.json"
    webui_ctl.save_state(
        {
            "frontend": {"pid": 3001, "create_time": 123.0, "port": 13080},
        },
        state_path,
    )

    killed: list[int] = []
    monkeypatch.setattr(webui_ctl, "_process_is_alive", lambda pid, ctime=None: pid == 3001)
    monkeypatch.setattr(webui_ctl, "_process_matches_service", lambda pid, svc, root=webui_ctl.REPO_ROOT: pid == 3001 and svc == "frontend")
    monkeypatch.setattr(webui_ctl, "_terminate_process_tree", lambda pid, timeout=5.0: killed.append(pid) or True)
    monkeypatch.setattr(
        webui_ctl,
        "_listening_pids_on_port",
        lambda port: [9999] if port == 18000 else [],
    )
    monkeypatch.setattr(webui_ctl, "_find_service_root_pid", lambda pid, svc, root=webui_ctl.REPO_ROOT: None)
    monkeypatch.setattr(webui_ctl, "_is_port_open", lambda host, port, timeout=0.3: False)

    res = webui_ctl.stop_services(
        backend_port=18000,
        frontend_port=13080,
        repo_root=tmp_path,
        state_path=state_path,
    )

    assert res == {"frontend": "stopped", "backend": "external"}
    assert killed == [3001]
    assert not state_path.exists()


def test_stop_services_uses_port_fallback_for_untracked_youdub_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证即使没有状态文件（如由外部拉起后端），也能通过端口归属校验安全停止。"""
    state_path = tmp_path / "webui_runtime.json"
    killed: list[int] = []
    monkeypatch.setattr(
        webui_ctl,
        "_listening_pids_on_port",
        lambda port: [4567] if port == 8000 else [],
    )
    monkeypatch.setattr(
        webui_ctl,
        "_find_service_root_pid",
        lambda pid, svc, root=webui_ctl.REPO_ROOT: 4560 if (pid == 4567 and svc == "backend") else None,
    )
    monkeypatch.setattr(webui_ctl, "_terminate_process_tree", lambda pid, timeout=5.0: killed.append(pid) or True)
    monkeypatch.setattr(webui_ctl, "_is_port_open", lambda host, port, timeout=0.3: False)

    res = webui_ctl.stop_services(
        backend_port=8000,
        frontend_port=3080,
        repo_root=tmp_path,
        state_path=state_path,
    )
    assert res == {"frontend": "not_running", "backend": "stopped"}
    assert killed == [4560]


def test_parse_args_open_override():
    """验证 Start-YouDub-Frontend.cmd 默认附带 --open 时，仍可用 --no-open 覆盖。"""
    args = webui_ctl.parse_args(
        ["start", "--target", "frontend", "--open", "--no-open", "--frontend-port", "3000"]
    )
    assert args.command == "start"
    assert args.target == "frontend"
    assert args.open_browser is False
    assert args.frontend_port == 3000


def test_start_and_stop_backend_only_preserves_frontend_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证仅启停后端时不拉起也不影响已记录的前端状态。"""
    repo = _make_fake_repo(tmp_path)
    state_path = tmp_path / "run" / "webui_runtime.json"
    webui_ctl.save_state(
        {"frontend": {"pid": 5555, "create_time": 100.0, "port": 13080}},
        state_path,
    )

    ready = {"backend": False}
    spawned: list[list[str]] = []
    killed: list[int] = []

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> int | None:
            return None

    def _fake_spawn(cmd: list[str], *, cwd: Path, log_path: Path, banner: str, env: dict[str, str] | None = None):
        spawned.append(cmd)
        ready["backend"] = True
        return _FakeProc(4444), 0

    monkeypatch.setattr(webui_ctl, "is_youdub_backend", lambda port, host="127.0.0.1": ready["backend"])
    monkeypatch.setattr(webui_ctl, "_is_port_open", lambda host, port, timeout=0.5: False)
    monkeypatch.setattr(webui_ctl, "_spawn_background", _fake_spawn)
    monkeypatch.setattr(webui_ctl, "_process_create_time", lambda pid: 200.0)

    start_res = webui_ctl.start_services(
        target="backend",
        backend_port=18000,
        open_browser=True,
        repo_root=repo,
        state_path=state_path,
        backend_log_path=tmp_path / "b.log",
        frontend_log_path=tmp_path / "f.log",
    )
    assert start_res["backend_pid"] == 4444
    assert "frontend_pid" not in start_res
    assert len(spawned) == 1

    saved = webui_ctl.load_state(state_path)
    assert saved["backend"]["pid"] == 4444
    assert saved["frontend"]["pid"] == 5555

    monkeypatch.setattr(webui_ctl, "_process_is_alive", lambda pid, ctime=None: pid in {4444, 5555})
    monkeypatch.setattr(webui_ctl, "_process_matches_service", lambda pid, svc, root=webui_ctl.REPO_ROOT: pid == 4444 and svc == "backend")
    monkeypatch.setattr(webui_ctl, "_terminate_process_tree", lambda pid, timeout=5.0: killed.append(pid) or True)
    monkeypatch.setattr(webui_ctl, "_listening_pids_on_port", lambda port: [])

    stop_res = webui_ctl.stop_services(
        target="backend",
        repo_root=repo,
        state_path=state_path,
    )
    assert stop_res == {"backend": "stopped"}
    assert killed == [4444]
    # 前端状态未被清除
    remaining = webui_ctl.load_state(state_path)
    assert "backend" not in remaining
    assert remaining["frontend"]["pid"] == 5555


def test_start_frontend_only_uses_active_backend_port_from_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """验证单独启动前端时自动沿用状态文件中活跃后端的端口作为代理目标。"""
    repo = _make_fake_repo(tmp_path)
    state_path = tmp_path / "run" / "webui_runtime.json"
    webui_ctl.save_state(
        {"backend": {"pid": 7777, "create_time": 300.0, "port": 18088}},
        state_path,
    )

    ready = {"frontend": False}
    captured_env: dict[str, str] = {}
    opened: list[str] = []

    class _FakeProc:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def poll(self) -> int | None:
            return None

    def _fake_spawn(cmd: list[str], *, cwd: Path, log_path: Path, banner: str, env: dict[str, str] | None = None):
        if env:
            captured_env.update(env)
        ready["frontend"] = True
        return _FakeProc(8888), 0

    monkeypatch.setattr(webui_ctl.shutil, "which", lambda name: "/usr/bin/node" if "node" in name else None)
    monkeypatch.setattr(webui_ctl, "_process_is_alive", lambda pid, ctime=None: pid == 7777)
    monkeypatch.setattr(webui_ctl, "is_youdub_backend", lambda port, host="127.0.0.1": port == 18088)
    monkeypatch.setattr(webui_ctl, "is_youdub_frontend", lambda port, host="127.0.0.1", timeout=4.0: ready["frontend"])
    monkeypatch.setattr(webui_ctl, "_is_port_open", lambda host, port, timeout=0.5: False)
    monkeypatch.setattr(webui_ctl, "_spawn_background", _fake_spawn)
    monkeypatch.setattr(webui_ctl, "_process_create_time", lambda pid: 400.0)
    monkeypatch.setattr(webui_ctl.webbrowser, "open", lambda url: opened.append(url) or True)

    res = webui_ctl.start_services(
        target="frontend",
        frontend_port=13088,
        open_browser=True,
        repo_root=repo,
        state_path=state_path,
        backend_log_path=tmp_path / "b.log",
        frontend_log_path=tmp_path / "f.log",
    )
    assert res["frontend_pid"] == 8888
    assert "backend_pid" not in res
    assert captured_env["NEXT_SERVER_API_BASE_URL"] == "http://127.0.0.1:18088"
    assert opened == ["http://localhost:13088"]

