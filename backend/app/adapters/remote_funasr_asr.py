"""Remote FunASR service adapter.

This adapter keeps vLLM/FunASR in a separate Linux/WSL service and converts
its HTTP response back into YouDub's local ``asr.json`` schema.
"""

from __future__ import annotations

import json
import os
import hashlib
import re
import shlex
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urljoin

import httpx
from pydub import AudioSegment

from ._time_utils import seconds_to_ms as _seconds_to_ms
from ._lang_map import LANG_TO_REMOTE_FUNASR as _LANG_TO_FUNASR
from ._lang_map import LANG_TO_OPENAI as _LANG_TO_OPENAI

_REMOTE_MODEL_KEYWORDS = (
    "Fun-ASR",
    "GLM-ASR-Nano",
    "LLMASR",
)
_SENTENCE_END = re.compile(r"[.!?\u3002\uff01\uff1f]+$")


def is_configured() -> bool:
    return bool(os.getenv("FUNASR_REMOTE_BASE_URL", "").strip())


def requires_remote_vllm(model_id: str | None) -> bool:
    target = model_id or os.getenv("FUNASR_MODEL", "")
    return any(keyword in target for keyword in _REMOTE_MODEL_KEYWORDS)


def _base_url() -> str:
    value = os.getenv("FUNASR_REMOTE_BASE_URL", "").strip()
    if not value:
        raise RuntimeError("FUNASR_REMOTE_BASE_URL is not configured.")
    return value.rstrip("/") + "/"


def _endpoint() -> str:
    return os.getenv("FUNASR_REMOTE_ENDPOINT", "/asr").strip() or "/asr"


def _timeout() -> float:
    raw = os.getenv("FUNASR_REMOTE_TIMEOUT", "900").strip()
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 900.0


def _bool_env(name: str, default: bool = False) -> str:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return "true" if default else "false"
    return "true" if raw in {"1", "true", "yes", "on"} else "false"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on", "auto"}


def _path_mode() -> str:
    return os.getenv("FUNASR_REMOTE_PATH_MODE", "auto").strip().lower() or "auto"


def _windows_path_to_wsl(path: Path) -> str | None:
    raw = str(path.resolve())
    if len(raw) >= 3 and raw[1:3] == ":\\":
        drive = raw[0].lower()
        tail = raw[3:].replace("\\", "/")
        return f"/mnt/{drive}/{tail}"
    return None


def _remote_audio_path(vocals_file: Path) -> str | None:
    mode = _path_mode()
    if mode in {"off", "upload", "false", "0", "no"}:
        return None
    override = os.getenv("FUNASR_REMOTE_AUDIO_PATH", "").strip()
    if override:
        return override
    if mode in {"wsl", "auto", "on", "true", "1", "yes"}:
        return _windows_path_to_wsl(vocals_file)
    return None


def _url() -> str:
    endpoint = _endpoint()
    if endpoint.startswith("http://") or endpoint.startswith("https://"):
        return endpoint
    return urljoin(_base_url(), endpoint.lstrip("/"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _service_dir() -> Path:
    return _repo_root() / "services" / "funasr-vllm"


def _service_shell_path(path: Path) -> str:
    wsl_path = _windows_path_to_wsl(path)
    return wsl_path or str(path)


def _service_log_path() -> Path:
    override = os.getenv("FUNASR_REMOTE_START_LOG", "").strip()
    if override:
        return Path(override)
    return _service_dir() / "autostart.log"


def _start_timeout() -> float:
    raw = os.getenv("FUNASR_REMOTE_START_TIMEOUT", "300").strip()
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 300.0


def _stop_timeout() -> float:
    raw = os.getenv("FUNASR_REMOTE_STOP_TIMEOUT", "30").strip()
    try:
        return max(5.0, float(raw))
    except ValueError:
        return 30.0


def _start_command() -> str:
    override = os.getenv("FUNASR_REMOTE_START_COMMAND", "").strip()
    if override:
        return override
    service_dir = shlex.quote(_service_shell_path(_service_dir()))
    return f"cd {service_dir} && bash ./start.sh"


def _stop_command() -> str:
    override = os.getenv("FUNASR_REMOTE_STOP_COMMAND", "").strip()
    if override:
        return override
    service_dir = shlex.quote(_service_shell_path(_service_dir()))
    return f"cd {service_dir} && bash ./stop.sh"


def _shell_args(command: str) -> list[str]:
    if os.name == "nt":
        return ["wsl", "bash", "-lc", command]
    return ["bash", "-lc", command]


def _creation_flags() -> int:
    if os.name != "nt":
        return 0
    return int(getattr(subprocess, "CREATE_NO_WINDOW", 0))


def _service_is_running() -> bool:
    try:
        httpx.get(_base_url(), timeout=2.0)
    except httpx.RequestError:
        return False
    return True


def _start_service() -> subprocess.Popen:
    log_path = _service_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _start_command()
    try:
        log_handle = log_path.open("ab")
    except OSError as exc:
        raise RuntimeError(f"Cannot open FunASR autostart log {log_path}: {exc}") from exc
    try:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_handle.write(f"\n[{stamp}] auto-start: {command}\n".encode("utf-8"))
        log_handle.flush()
        return subprocess.Popen(
            _shell_args(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=_creation_flags(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError("Cannot auto-start remote FunASR service: WSL or bash was not found.") from exc
    except OSError as exc:
        raise RuntimeError(f"Cannot auto-start remote FunASR service: {exc}") from exc
    finally:
        log_handle.close()


def _wait_for_service(process: subprocess.Popen) -> None:
    deadline = time.monotonic() + _start_timeout()
    log_path = _service_log_path()
    while time.monotonic() < deadline:
        if _service_is_running():
            return
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                "Remote FunASR service exited before it became ready "
                f"(exit code {return_code}). See {log_path}."
            )
        time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for remote FunASR service. See {log_path}.")


def _stop_service(process: subprocess.Popen | None) -> None:
    log_path = _service_log_path()
    command = _stop_command()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as log_handle:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_handle.write(f"\n[{stamp}] auto-stop: {command}\n".encode("utf-8"))
            log_handle.flush()
            subprocess.run(
                _shell_args(command),
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=_stop_timeout(),
                creationflags=_creation_flags(),
                check=False,
            )
    except Exception:
        pass
    if process is not None and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception:
            pass


@contextmanager
def _managed_remote_service() -> Iterator[None]:
    process: subprocess.Popen | None = None
    started = False
    if _env_enabled("FUNASR_REMOTE_AUTOSTART"):
        if not _service_is_running():
            process = _start_service()
            started = True
            try:
                _wait_for_service(process)
            except Exception:
                _stop_service(process)
                raise
    try:
        yield
    finally:
        if started and _env_enabled("FUNASR_REMOTE_AUTOSTOP", True):
            _stop_service(process)


def _is_openai_endpoint(url: str) -> bool:
    return "/v1/audio/transcriptions" in url


def _model_alias(model_id: str | None) -> str:
    override = os.getenv("FUNASR_REMOTE_MODEL", "").strip()
    if override:
        return override
    target = model_id or os.getenv("FUNASR_MODEL", "")
    if "SenseVoice" in target:
        return "sensevoice"
    if "Fun-ASR" in target:
        return "fun-asr-nano"
    if "GLM-ASR" in target:
        return "glm-asr-nano"
    return target or "fun-asr-nano"


def _audio_duration_ms(vocals_file: Path) -> int:
    return len(AudioSegment.from_file(vocals_file))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request_debug(vocals_file: Path, language: str, model_id: str | None, url: str) -> dict[str, Any]:
    return {
        "url": url,
        "audio_path": str(vocals_file),
        "audio_name": vocals_file.name,
        "audio_size": vocals_file.stat().st_size,
        "audio_sha256": _file_sha256(vocals_file),
        "audio_duration_ms": _audio_duration_ms(vocals_file),
        "language": language,
        "model_id": model_id,
    }


def _duration_ms(data: dict[str, Any], vocals_file: Path) -> int:
    if "duration" in data:
        return _seconds_to_ms(data["duration"])
    return _audio_duration_ms(vocals_file)


def _word_to_youdub(word: dict[str, Any]) -> dict[str, Any]:
    text = word.get("word") or word.get("text") or word.get("token") or ""
    return {
        "text": str(text),
        "start_time": _seconds_to_ms(word.get("start", word.get("start_time"))),
        "end_time": _seconds_to_ms(word.get("end", word.get("end_time"))),
    }


def _clean_word_text(text: Any) -> str:
    return str(text or "").strip()


def _words_to_utterances(segment: dict[str, Any]) -> list[dict[str, Any]]:
    raw_words = [word for word in segment.get("words", []) if isinstance(word, dict)]
    if not raw_words:
        return []

    utterances: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []
    current_text: list[str] = []

    def emit() -> None:
        nonlocal current_words, current_text
        if not current_words:
            return
        text = " ".join(part for part in current_text if part).strip()
        if text:
            utterances.append({
                "text": text,
                "start_time": current_words[0]["start_time"],
                "end_time": current_words[-1]["end_time"],
                "words": current_words,
            })
        current_words = []
        current_text = []

    for raw_word in raw_words:
        word = _word_to_youdub(raw_word)
        token = _clean_word_text(word["text"])
        if not token:
            continue
        if word["end_time"] <= word["start_time"]:
            continue

        if token in {".", "!", "?", "。", "！", "？"}:
            if current_text:
                current_text[-1] = current_text[-1] + token
            else:
                current_text.append(token)
            current_words.append(word)
            emit()
            continue

        if token in {",", ";", ":", "，", "；", "："}:
            if current_text:
                current_text[-1] = current_text[-1] + token
            else:
                current_text.append(token)
            current_words.append(word)
            continue

        current_text.append(token)
        current_words.append(word)
        if _SENTENCE_END.search(token):
            emit()

    emit()
    return utterances


def _segment_to_utterance(segment: dict[str, Any], duration_ms: int) -> dict[str, Any]:
    text = str(segment.get("text") or "").strip()
    start_ms = _seconds_to_ms(segment.get("start", segment.get("start_time")), 0)
    end_ms = _seconds_to_ms(segment.get("end", segment.get("end_time")), duration_ms)
    if end_ms <= start_ms:
        end_ms = min(duration_ms, start_ms + 1)

    utterance: dict[str, Any] = {
        "text": text,
        "start_time": max(0, start_ms),
        "end_time": max(0, end_ms),
        "words": [
            _word_to_youdub(word)
            for word in segment.get("words", [])
            if isinstance(word, dict)
        ],
    }
    speaker = segment.get("speaker", segment.get("spk"))
    if speaker is not None:
        utterance["additions"] = {"speaker": str(speaker)}
    return utterance


def _segment_to_utterances(segment: dict[str, Any], duration_ms: int) -> list[dict[str, Any]]:
    word_utterances = _words_to_utterances(segment)
    if len(word_utterances) > 1:
        return word_utterances
    return [_segment_to_utterance(segment, duration_ms)]


def _convert_response(data: dict[str, Any], vocals_file: Path) -> dict[str, Any]:
    duration_ms = _duration_ms(data, vocals_file)
    text = str(data.get("text") or "").strip()
    segments = data.get("segments") or data.get("sentences") or []
    utterances: list[dict[str, Any]] = []
    for segment in segments:
        if isinstance(segment, dict) and str(segment.get("text") or "").strip():
            utterances.extend(_segment_to_utterances(segment, duration_ms))

    if not utterances and text:
        utterances = [{
            "text": text,
            "start_time": 0,
            "end_time": duration_ms,
            "words": [],
        }]
    if not utterances:
        raise RuntimeError("Remote FunASR did not return any segments.")
    if not text:
        text = " ".join(item["text"] for item in utterances)

    return {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": text,
            "utterances": utterances,
        },
    }


def _post_transcription(vocals_file: Path, language: str, model_id: str | None) -> tuple[dict[str, Any], dict[str, Any]]:
    url = _url()
    debug = _request_debug(vocals_file, language, model_id, url)
    timeout = httpx.Timeout(_timeout(), connect=30.0)
    remote_audio_path = None if _is_openai_endpoint(url) else _remote_audio_path(vocals_file)
    if remote_audio_path:
        files = None
        handle = None
    else:
        handle = vocals_file.open("rb")
        files = {"file": (vocals_file.name, handle, "application/octet-stream")}
    try:
        if _is_openai_endpoint(url):
            data = {
                "model": _model_alias(model_id),
                "response_format": "verbose_json",
                "language": _LANG_TO_OPENAI.get(language, language),
                "timestamp_granularities": "segment",
            }
            if _bool_env("FUNASR_REMOTE_SPK") == "true":
                data["spk"] = "true"
        else:
            data = {
                "language": _LANG_TO_FUNASR.get(language, language),
                "timestamp": "true",
                "spk": _bool_env("FUNASR_REMOTE_SPK"),
            }
            if remote_audio_path:
                data["audio_path"] = remote_audio_path
            hotwords = os.getenv("FUNASR_REMOTE_HOTWORDS", "").strip()
            if hotwords:
                data["hotwords"] = hotwords
        debug["form"] = {key: value for key, value in data.items() if key != "file"}
        debug["transport"] = "path" if remote_audio_path else "upload"
        if remote_audio_path:
            debug["remote_audio_path"] = remote_audio_path

        try:
            response = httpx.post(url, files=files, data=data, timeout=timeout)
        except httpx.RequestError as exc:
            raise RuntimeError(f"Remote FunASR request failed: {exc}") from exc
    finally:
        if handle is not None:
            handle.close()
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = response.text[:500].strip()
        raise RuntimeError(f"Remote FunASR request failed: HTTP {response.status_code}. {detail}") from exc
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Remote FunASR returned non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Remote FunASR returned an unexpected response shape.")
    return payload, debug


def recognize_speech(
    vocals_file: Path, session: Path, language: str, *, model_id: str | None = None
) -> Path:
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    with _managed_remote_service():
        remote_payload, request_debug = _post_transcription(vocals_file, language, model_id)
    (metadata_dir / "asr.remote_request.json").write_text(
        json.dumps(request_debug, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "asr.raw.remote_funasr.json").write_text(
        json.dumps(remote_payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    payload = _convert_response(remote_payload, vocals_file)
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
