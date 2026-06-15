from __future__ import annotations

import json
import os
from pathlib import Path

from pydub import AudioSegment

_MODEL = None
_MODEL_NAME = None


def _get_model_id() -> str:
    return os.getenv("FUNASR_MODEL", "iic/SenseVoiceSmall")


def _get_vad_model() -> str:
    return os.getenv("FUNASR_VAD_MODEL", "fsmn-vad")


def _load_model(model_id: str | None = None):
    global _MODEL, _MODEL_NAME
    target = model_id or _get_model_id()
    if _MODEL is not None and _MODEL_NAME == target:
        return _MODEL

    from funasr import AutoModel

    device = os.getenv("FUNASR_DEVICE") or os.getenv("DEVICE", "auto")
    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = dict(
        model=target,
        vad_model=_get_vad_model(),
        vad_kwargs={"max_single_segment_time": 30000},
        device=device,
    )
    if "Fun-ASR-Nano" in target or "Fun-ASR" in target:
        kwargs["trust_remote_code"] = True

    _MODEL = AutoModel(**kwargs)
    _MODEL_NAME = target
    return _MODEL


def _to_ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def _convert_result(result: dict, duration_ms: int) -> list[dict]:
    utterances = []
    sentences = result.get("sentence_info") or result.get("sentences") or []
    if sentences:
        for sent in sentences:
            text = sent.get("text", "").strip()
            if not text:
                continue
            start_raw = sent.get("start", 0)
            end_raw = sent.get("end", 0)
            # FunASR may return ms or seconds depending on model
            if start_raw > 100000:
                start_ms = int(start_raw)
                end_ms = int(end_raw)
            else:
                start_ms = _to_ms(start_raw / 1000.0) if start_raw > 1000 else _to_ms(start_raw)
                end_ms = _to_ms(end_raw / 1000.0) if end_raw > 1000 else _to_ms(end_raw)
            utterances.append({
                "text": text,
                "start_time": start_ms,
                "end_time": end_ms,
                "words": [],
            })
    else:
        text = result.get("text", "").strip()
        if text:
            utterances.append({
                "text": text,
                "start_time": 0,
                "end_time": duration_ms,
                "words": [],
            })
    return utterances


def recognize_speech(vocals_file: Path, session: Path, language: str, *, model_id: str | None = None) -> Path:
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    model = _load_model(model_id)

    lang_map = {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "yue": "yue"}
    funasr_lang = lang_map.get(language, "auto")

    is_nano = "Fun-ASR-Nano" in (model_id or _get_model_id())

    generate_kwargs: dict = dict(
        input=str(vocals_file),
        cache={},
        language=funasr_lang,
        use_itn=True,
        merge_vad=True,
        merge_length_s=15,
    )
    if not is_nano:
        generate_kwargs["batch_size_s"] = 60

    result = model.generate(**generate_kwargs)

    if not result:
        raise RuntimeError("FunASR did not return any results.")

    raw = result[0] if isinstance(result, list) else result
    duration_ms = len(AudioSegment.from_file(vocals_file))
    utterances = _convert_result(raw, duration_ms)
    if not utterances:
        raise RuntimeError("FunASR did not return any segments.")

    full_text = raw.get("text", "").strip()
    if not full_text:
        full_text = " ".join(u["text"] for u in utterances)

    payload = {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": full_text,
            "utterances": utterances,
        },
    }
    output_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_file
