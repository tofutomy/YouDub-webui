from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

from ..sources import SourceConfig
from ._translate_prompts import BATCH_TRANSLATE_RULES, PREPROCESS_PROMPT, TRANSLATE_RULES
from .openai_client import normalize_openai_base_url

log = logging.getLogger(__name__)

API_SETTING_KEYS = ("base_url", "api_key", "model")
PREPROCESS_RETRY = 2
TRANSLATE_RETRY = 2
DESCRIPTION_LIMIT = 500
DEFAULT_CONCURRENCY = 50
DEFAULT_BATCH_SIZE = 15
VALID_MODES = ("sentence", "batch")


class HotwordItem(BaseModel):
    src: str
    dst: str


class CorrectionItem(BaseModel):
    wrong: str
    correct: str


class PreprocessResponse(BaseModel):
    summary: str = ""
    hotwords: list[HotwordItem] = Field(default_factory=list)
    corrections: list[CorrectionItem] = Field(default_factory=list)


class TranslationItem(BaseModel):
    dst: str


def list_models(*, base_url: str, api_key: str) -> list[str]:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    client = OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))
    response = client.models.list()
    seen: set[str] = set()
    models: list[str] = []
    for item in response.data:
        model_id = getattr(item, "id", "")
        if model_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_id)
    return models


def _client(base_url: str, api_key: str) -> OpenAI:
    if not api_key:
        raise ValueError("OpenAI API key is not configured.")
    return OpenAI(api_key=api_key, base_url=normalize_openai_base_url(base_url))


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = _JSON_BLOCK_RE.search(raw)
    if not match:
        raise json.JSONDecodeError(f"no JSON object found; raw[:300]={raw[:300]!r}", raw, 0)
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise json.JSONDecodeError(
            f"{exc.msg}; len={len(raw)}; raw[:300]={raw[:300]!r}; raw[-200:]={raw[-200:]!r}",
            raw,
            exc.pos,
        ) from None


def _call_json(client: OpenAI, model: str, system: str, user: str) -> dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return _extract_json(raw)


def _format_terms(items: list, fmt: str, empty: str) -> str:
    if not items:
        return empty
    return "\n".join(fmt.format(**item.model_dump()) for item in items)


def _meta_view(meta: dict[str, Any]) -> dict[str, str]:
    description = (meta.get("description") or "").strip()
    if len(description) > DESCRIPTION_LIMIT:
        description = description[:DESCRIPTION_LIMIT] + "..."
    return {
        "title": str(meta.get("title") or "").strip() or "(unknown)",
        "uploader": str(meta.get("uploader") or "").strip() or "(unknown)",
        "description": description or "(none)",
    }


def preprocess(
    full_text: str,
    meta: dict[str, Any],
    source: SourceConfig,
    *,
    base_url: str,
    api_key: str,
    model: str,
) -> PreprocessResponse:
    user = PREPROCESS_PROMPT.format(
        src_language_name=source.asr_language_name,
        dst_language_name=source.target_language_name,
        full_text=full_text,
        **_meta_view(meta),
    )
    client = _client(base_url, api_key)
    last_error: Exception | None = None
    for attempt in range(PREPROCESS_RETRY + 1):
        try:
            data = _call_json(client, model, "You output strict JSON only.", user)
            return PreprocessResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log.warning("preprocess attempt %d failed: %s", attempt + 1, exc)
    log.error("preprocess gave up, returning empty: %s", last_error)
    return PreprocessResponse()


def _translate_system(source: SourceConfig, meta: dict[str, Any], pre: PreprocessResponse) -> str:
    rules = TRANSLATE_RULES[source.target_language]
    return rules.format(
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )


def _post_process(text: str, target_language: str) -> str:
    cleaned = text.strip()
    if target_language == "zh":
        cleaned = cleaned.replace("——", "，")
    return cleaned


def translate_sentence(
    text: str,
    target_language: str,
    client: OpenAI,
    model: str,
    system: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(TRANSLATE_RETRY):
        try:
            data = _call_json(client, model, system, text)
            item = TranslationItem.model_validate(data)
            if not item.dst.strip():
                raise ValueError("empty dst")
            return _post_process(item.dst, target_language)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            log.warning("translate attempt %d failed for %r: %s", attempt + 1, text[:60], exc)
    raise RuntimeError(f"translate_sentence failed after {TRANSLATE_RETRY} attempts: {last_error}")


def translate_batch(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[str]:
    if not texts:
        return []
    system = _translate_system(source, meta, pre)
    client = _client(base_url, api_key)
    log.info(
        "translate_batch: %d sentences, concurrency=%d", len(texts), concurrency,
    )
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        return list(pool.map(
            lambda t: translate_sentence(t, source.target_language, client, model, system),
            texts,
        ))


def _translate_batch_context_chunk(
    chunk: list[str],
    target_language: str,
    client: OpenAI,
    model: str,
    system: str,
    expected_count: int,
) -> list[str]:
    user = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
    last_error: Exception | None = None
    for attempt in range(TRANSLATE_RETRY + 1):
        data = _call_json(client, model, system, user)
        translations = data.get("translations")
        if isinstance(translations, list) and len(translations) == expected_count:
            return [_post_process(str(t), target_language) for t in translations]
        actual = len(translations) if isinstance(translations, list) else type(translations)
        last_error = ValueError(f"expected {expected_count}, got {actual}")
        log.warning("batch chunk attempt %d: %s", attempt + 1, last_error)
        # On retry, reinforce the count requirement
        user = (
            f"Translate exactly {expected_count} sentences. "
            f"Return exactly {expected_count} items in the translations array.\n\n"
            + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(chunk))
        )
    # Fallback: translate each sentence individually
    log.warning("batch chunk failed after retries, falling back to per-sentence: %s", last_error)
    system_single = TRANSLATE_RULES[target_language].format(
        summary="(none)", hotwords="(none)", corrections="(none)",
        title="(unknown)", uploader="(unknown)", description="(none)",
    )
    return [
        _post_process(
            translate_sentence(t, target_language, client, model, system_single),
            target_language,
        )
        for t in chunk
    ]


def translate_batch_context(
    texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int = DEFAULT_CONCURRENCY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[str]:
    if not texts:
        return []

    rules = BATCH_TRANSLATE_RULES[source.target_language]
    system = rules.format(
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )
    client = _client(base_url, api_key)

    half = max(1, batch_size // 2)
    chunks: list[tuple[int, list[str]]] = []
    for start in range(0, len(texts), half):
        end = min(start + batch_size, len(texts))
        chunks.append((start, texts[start:end]))
        if end >= len(texts):
            break

    log.info(
        "translate_batch_context: %d sentences -> %d chunks (batch_size=%d, concurrency=%d)",
        len(texts), len(chunks), batch_size, concurrency,
    )

    def _do_chunk(item: tuple[int, list[str]]) -> tuple[int, list[str]]:
        idx, chunk = item
        result = _translate_batch_context_chunk(
            chunk, source.target_language, client, model, system, len(chunk),
        )
        return (idx, result)

    results: list[tuple[int, list[str]]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        results = list(pool.map(_do_chunk, chunks))

    merged: list[str | None] = [None] * len(texts)
    for offset, chunk_translations in results:
        for i, t in enumerate(chunk_translations):
            pos = offset + i
            if pos < len(merged) and merged[pos] is None:
                merged[pos] = t
    return [t or "" for t in merged]


def _read_meta(session: Path) -> dict[str, Any]:
    info_file = session / "metadata" / "ytdlp_info.json"
    if not info_file.exists():
        return {}
    return json.loads(info_file.read_text(encoding="utf-8"))


def _speaker(utt: dict[str, Any]) -> str:
    additions = utt.get("additions") or {}
    if isinstance(additions, dict):
        return str(additions.get("speaker") or "1")
    return "1"


def _full_text(data: dict[str, Any], texts: list[str]) -> str:
    raw = data.get("result", {}).get("text") or ""
    if raw.strip():
        return raw
    return " ".join(texts)


def _concurrency_from(settings: dict[str, str]) -> int:
    raw = str(settings.get("translate_concurrency") or "").strip()
    if not raw or not all("0" <= char <= "9" for char in raw):
        return DEFAULT_CONCURRENCY
    concurrency = int(raw)
    if concurrency < 1 or concurrency > 200:
        return DEFAULT_CONCURRENCY
    return concurrency


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    if output_file.exists():
        _generate_srt_if_needed(output_file, session, source.target_language)
        return output_file

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    full_text = _full_text(data, texts)
    meta = _read_meta(session)

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre = preprocess(full_text, meta, source, **api)

    # Save preprocess output for debugging / analysis
    pre_file = session / "metadata" / "preprocess.json"
    pre_file.write_text(
        json.dumps(pre.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("preprocess saved to %s", pre_file.name)

    mode = (settings.get("translate_mode") or "sentence").strip().lower()
    if mode not in VALID_MODES:
        mode = "sentence"
    concurrency = _concurrency_from(settings)

    if mode == "batch":
        log.info("translate mode=batch")
        dst_list = translate_batch_context(
            texts, source, meta, pre, **api, concurrency=concurrency,
        )
    else:
        log.info("translate mode=sentence")
        dst_list = translate_batch(
            texts, source, meta, pre, **api, concurrency=concurrency,
        )

    translation = [
        {
            "src": text,
            "dst": dst,
            "src_lang": source.asr_language,
            "dst_lang": source.target_language,
            "start_time": utt["start_time"],
            "end_time": utt["end_time"],
            "speaker": _speaker(utt),
        }
        for text, dst, utt in zip(texts, dst_list, utterances)
    ]
    output_file.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _generate_srt_if_needed(output_file, session, source.target_language)
    return output_file


def _generate_srt_if_needed(translation_file: Path, session: Path, target_language: str) -> None:
    from .ffmpeg import write_srt
    srt_file = session / "metadata" / f"subtitles.{target_language}.srt"
    if not srt_file.exists():
        write_srt(translation_file, session)
