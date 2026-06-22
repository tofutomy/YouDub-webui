from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from openai import BadRequestError, OpenAI
from pydantic import BaseModel, Field, ValidationError

from ..sources import SourceConfig
from ._translate_prompts import (
    PREPROCESS_PROMPT,
    get_translate_rules,
)
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


class ValidationIssue(BaseModel):
    index: int
    type: str
    problem: str
    suggestion: str


class ValidationBatchResult(BaseModel):
    score: int = 0
    issues: list[ValidationIssue] = Field(default_factory=list)


DEFAULT_VALIDATION_THRESHOLD = 99  # 校正阈值，评分高于该值的批次将不会进行校正
DEFAULT_MAX_CORRECTIONS = 2
VALIDATION_CONCURRENCY = 10
DEFAULT_VALIDATION_BATCH_SIZE = 25  # 校验用更大窗口，与翻译(15)错开，提供不同上下文视角


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
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
    except BadRequestError:
        # Some providers don't support json_object response_format; retry without it.
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
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
    rules = get_translate_rules(source.asr_language, source.target_language)
    return rules.format(
        src_language_name=source.asr_language_name,
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )


_LEADING_NUM_RE = re.compile(r"^\d{1,3}\s*[.、．]\s*")


def _post_process(text: str, target_language: str, src: str = "") -> str:
    cleaned = text.strip()
    # Strip model-injected sentence numbers (e.g. "9. 众所周知" -> "众所周知")
    # Only when the source text does NOT start with a digit, to avoid
    # stripping legitimate numbers like "3.14" or "100人".
    if src and cleaned and not src[0:1].isdigit():
        cleaned = _LEADING_NUM_RE.sub("", cleaned)
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
            return _post_process(item.dst, target_language, src=text)
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
    on_progress: Callable[[int, str], None] | None = None,
) -> list[str]:
    if not texts:
        return []
    system = _translate_system(source, meta, pre)
    client = _client(base_url, api_key)
    log.info(
        "translate_batch: %d sentences, concurrency=%d", len(texts), concurrency,
    )
    results: list[str] = [""] * len(texts)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        future_to_idx = {
            pool.submit(translate_sentence, t, source.target_language, client, model, system): i
            for i, t in enumerate(texts)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            if on_progress:
                on_progress(idx, results[idx])
    return results


def _translate_batch_context_chunk(
    chunk: list[str],
    asr_language: str,
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
            return [_post_process(str(t), target_language, src=c) for t, c in zip(translations, chunk)]
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
    system_single = get_translate_rules(asr_language, target_language)
    system_single = system_single.format(
        src_language_name="(unknown)",
        summary="(none)", hotwords="(none)", corrections="(none)",
        title="(unknown)", uploader="(unknown)", description="(none)",
    )
    return [
        _post_process(
            translate_sentence(t, target_language, client, model, system_single),
            target_language,
            src=t,
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
    on_progress: Callable[[int, str], None] | None = None,
    completed: set[int] | None = None,
    dst_list: list[str] | None = None,
    checkpoint_file: Path | None = None,
) -> list[str]:
    if not texts:
        return []

    _completed = completed or set()

    rules = get_translate_rules(source.asr_language, source.target_language, "batch")
    system = rules.format(
        src_language_name=source.asr_language_name,
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        corrections=_format_terms(pre.corrections, "{wrong} -> {correct}", "(none)"),
        **_meta_view(meta),
    )
    client = _client(base_url, api_key)

    half = max(1, batch_size // 2)
    chunks: list[tuple[int, list[str]]] = []
    skipped = 0
    for start in range(0, len(texts), half):
        end = min(start + batch_size, len(texts))
        # Skip chunk if ALL sentences in this range are already completed
        if _completed and all(i in _completed for i in range(start, end)):
            skipped += 1
            continue
        chunks.append((start, texts[start:end]))
        if end >= len(texts):
            break

    # Initialize merged from dst_list (preserving already-completed translations)
    if dst_list and len(dst_list) == len(texts):
        merged: list[str | None] = list(dst_list)
    else:
        merged = [None] * len(texts)

    log.info(
        "translate_batch_context: %d sentences -> %d chunks (batch_size=%d, concurrency=%d, resumed=%d, skipped=%d)",
        len(texts), len(chunks), batch_size, concurrency, len(_completed), skipped,
    )

    def _do_chunk(item: tuple[int, list[str]]) -> tuple[int, list[str]]:
        idx, chunk = item
        result = _translate_batch_context_chunk(
            chunk, source.asr_language, source.target_language, client, model, system, len(chunk),
        )
        return (idx, result)

    import threading as _threading
    _checkpoint_lock = _threading.Lock()

    def _save_checkpoint() -> None:
        if not checkpoint_file:
            return
        with _checkpoint_lock:
            cp = {str(i): merged[i] for i in range(len(merged)) if merged[i]}
            checkpoint_file.write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")

    failed_chunks: list[tuple[int, Exception]] = []

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(_do_chunk, c): c[0] for c in chunks}
        for future in as_completed(futures):
            chunk_start = futures[future]
            try:
                offset, chunk_translations = future.result()
            except Exception as exc:
                log.error("translate chunk starting at sentence %d failed: %s", chunk_start, exc)
                failed_chunks.append((chunk_start, exc))
                continue
            for i, t in enumerate(chunk_translations):
                pos = offset + i
                if pos < len(merged) and not merged[pos]:
                    merged[pos] = t
                    if on_progress:
                        on_progress(pos, t)
            # Real-time incremental checkpoint save after each chunk
            _save_checkpoint()

    if failed_chunks:
        _save_checkpoint()
        succeeded = len(chunks) - len(failed_chunks)
        log.warning(
            "%d/%d chunks failed; %d chunks succeeded and saved to checkpoint for resume",
            len(failed_chunks), len(chunks), succeeded,
        )
        raise RuntimeError(
            f"{len(failed_chunks)}/{len(chunks)} translation chunks failed; "
            f"{succeeded} chunks saved to checkpoint. "
            f"Last error: {failed_chunks[-1][1]}"
        )

    return [t or "" for t in merged]


def _validation_system(source: SourceConfig, meta: dict[str, Any], pre: PreprocessResponse) -> str:
    rules = get_translate_rules(source.asr_language, source.target_language, "validation")
    return rules.format(
        src_language_name=source.asr_language_name,
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        **_meta_view(meta),
    )


def _correction_system(source: SourceConfig, meta: dict[str, Any], pre: PreprocessResponse) -> str:
    rules = get_translate_rules(source.asr_language, source.target_language, "correction")
    return rules.format(
        src_language_name=source.asr_language_name,
        summary=pre.summary or "(none)",
        hotwords=_format_terms(pre.hotwords, "{src} -> {dst}", "(none)"),
        **_meta_view(meta),
    )


VALIDATE_RETRY = 2
CORRECT_RETRY = 2


def validate_batch(
    src_texts: list[str],
    dst_texts: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    client: OpenAI,
    model: str,
) -> ValidationBatchResult:
    """Validate a batch of translations and return issues + score."""
    system = _validation_system(source, meta, pre)
    lines = []
    for i, (src, dst) in enumerate(zip(src_texts, dst_texts)):
        lines.append(f"{i}. {src} -> {dst}")
    user = "\n".join(lines)

    last_error: Exception | None = None
    for attempt in range(VALIDATE_RETRY + 1):
        try:
            data = _call_json(client, model, system, user)
            result = ValidationBatchResult.model_validate(data)
            # Clamp score to 0-100
            result.score = max(0, min(100, result.score))
            return result
        except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as exc:
            last_error = exc
            log.warning("validate_batch attempt %d failed: %s", attempt + 1, exc)
    log.error("validate_batch gave up after retries: %s", last_error)
    return ValidationBatchResult(score=100, issues=[])


def correct_sentence(
    src_text: str,
    dst_text: str,
    problem: str,
    suggestion: str,
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    client: OpenAI,
    model: str,
) -> str:
    """Correct a single translation based on validation feedback."""
    system = _correction_system(source, meta, pre)
    user = (
        f"原文：{src_text}\n"
        f"原译文：{dst_text}\n"
        f"问题：{problem}\n"
        f"建议：{suggestion}"
    )

    last_error: Exception | None = None
    for attempt in range(CORRECT_RETRY):
        try:
            data = _call_json(client, model, system, user)
            item = TranslationItem.model_validate(data)
            if not item.dst.strip():
                raise ValueError("empty dst")
            return _post_process(item.dst, source.target_language, src=src_text)
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            last_error = exc
            log.warning("correct_sentence attempt %d failed for %r: %s", attempt + 1, src_text[:60], exc)
    log.warning("correct_sentence failed, keeping original: %s", last_error)
    return dst_text


def validate_and_correct(
    texts: list[str],
    dst_list: list[str],
    source: SourceConfig,
    meta: dict[str, Any],
    pre: PreprocessResponse,
    *,
    base_url: str,
    api_key: str,
    model: str,
    threshold: int = DEFAULT_VALIDATION_THRESHOLD,
    max_corrections: int = DEFAULT_MAX_CORRECTIONS,
    checkpoint_file: Path | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate translations in batches, then correct issues.

    Returns (corrected_translations, validation_report).
    If checkpoint_file is provided, batch results are saved incrementally
    and resumed on next call.
    """
    if not texts:
        return dst_list, {"overall_score": 100, "batches": [], "corrected_count": 0}

    client = _client(base_url, api_key)

    # Validation uses a larger, non-overlapping batch to provide a different
    # contextual view than translation (which uses batch_size=15 with overlap).
    validation_batch_size = DEFAULT_VALIDATION_BATCH_SIZE
    chunks: list[tuple[int, int]] = []  # (start, end)
    for start in range(0, len(texts), validation_batch_size):
        end = min(start + validation_batch_size, len(texts))
        chunks.append((start, end))

    # Load checkpoint if available
    saved_batches: dict[int, dict[str, Any]] = {}  # batch_start -> batch_report
    if checkpoint_file and checkpoint_file.exists():
        try:
            cp = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            for entry in cp.get("batches", []):
                saved_batches[entry["start"]] = entry
            # Restore corrected dst_list entries
            for entry in cp.get("corrections", []):
                idx = entry["idx"]
                if 0 <= idx < len(dst_list):
                    dst_list[idx] = entry["dst"]
            if saved_batches:
                log.info("validation checkpoint loaded: %d/%d batches completed", len(saved_batches), len(chunks))
        except Exception as exc:
            log.warning("failed to load validation checkpoint, starting fresh: %s", exc)

    log.info(
        "validate_and_correct: %d sentences -> %d validation batches (threshold=%d, resuming=%d)",
        len(texts), len(chunks), threshold, len(saved_batches),
    )

    # Validate each batch (skip already-completed ones)
    batch_reports: list[dict[str, Any]] = []
    all_issues: list[tuple[int, ValidationIssue]] = []  # (global_index, issue)
    total_score = 0
    pending_chunks: list[tuple[int, int]] = []

    # First, restore saved batch results
    for start, end in chunks:
        if start in saved_batches:
            batch_report = saved_batches[start]
            batch_reports.append(batch_report)
            total_score += batch_report["score"]
            for issue_entry in batch_report.get("issues", []):
                global_idx = issue_entry["sentence_index"]
                all_issues.append((global_idx, ValidationIssue(
                    index=global_idx - start,
                    type=issue_entry["type"],
                    problem=issue_entry["problem"],
                    suggestion=issue_entry["suggestion"],
                )))
        else:
            pending_chunks.append((start, end))
            batch_reports.append(None)  # placeholder

    # Validate pending batches
    _validated_so_far = len(saved_batches)
    if pending_chunks:
        def _validate_chunk(item: tuple[int, int]) -> tuple[int, ValidationBatchResult]:
            start, end = item
            return start, validate_batch(
                texts[start:end], dst_list[start:end],
                source, meta, pre, client=client, model=model,
            )

        with ThreadPoolExecutor(max_workers=max(1, VALIDATION_CONCURRENCY)) as pool:
            results = list(pool.map(_validate_chunk, pending_chunks))

        for start, batch_result in results:
            _validated_so_far += 1
            if progress_callback is not None:
                pct = 50 + round(_validated_so_far / max(1, len(chunks)) * 45)
                progress_callback(pct, f"校验 {_validated_so_far}/{len(chunks)} 批")
            end = min(start + validation_batch_size, len(texts))
            total_score += batch_result.score
            batch_report = {
                "start": start,
                "batch_index": next(i for i, c in enumerate(chunks) if c[0] == start),
                "sentences": list(range(start, end)),
                "score": batch_result.score,
                "issues": [],
            }
            for issue in batch_result.issues:
                global_idx = start + issue.index
                if 0 <= global_idx < len(dst_list):
                    all_issues.append((global_idx, issue))
                    batch_report["issues"].append({
                        "sentence_index": global_idx,
                        "type": issue.type,
                        "problem": issue.problem,
                        "suggestion": issue.suggestion,
                    })
            # Update placeholder
            batch_idx = next(i for i, c in enumerate(chunks) if c[0] == start)
            batch_reports[batch_idx] = batch_report

            # Save checkpoint after each batch
            if checkpoint_file:
                _save_validation_checkpoint(checkpoint_file, batch_reports, dst_list)

    overall_score = total_score // max(1, len(chunks))

    # Correct issues for batches below threshold
    corrected_count = 0
    correctable = [(idx, issue) for idx, issue in all_issues
                   if batch_reports[next(i for i, c in enumerate(chunks) if c[0] <= idx < c[1])]["score"] < threshold]

    # Limit corrections per sentence
    seen_indices: dict[int, int] = {}
    for idx, issue in correctable:
        seen_indices[idx] = seen_indices.get(idx, 0) + 1
    to_correct = [(idx, issue) for idx, issue in correctable if seen_indices[idx] <= max_corrections]

    log.info("validate_and_correct: %d issues found, %d correctable", len(all_issues), len(to_correct))

    for idx, issue in to_correct:
        original = dst_list[idx]
        corrected = correct_sentence(
            texts[idx], original, issue.problem, issue.suggestion,
            source, meta, pre, client=client, model=model,
        )
        if corrected != original:
            dst_list[idx] = corrected
            corrected_count += 1

    validation_report = {
        "overall_score": overall_score,
        "batches": batch_reports,
        "corrected_count": corrected_count,
    }
    return dst_list, validation_report


def _save_validation_checkpoint(
    checkpoint_file: Path,
    batch_reports: list[dict[str, Any]],
    dst_list: list[str],
) -> None:
    """Save validation checkpoint: completed batches + corrected translations."""
    corrections = []
    for report in batch_reports:
        if report is None:
            continue
        for issue in report.get("issues", []):
            idx = issue["sentence_index"]
            if 0 <= idx < len(dst_list):
                corrections.append({"idx": idx, "dst": dst_list[idx]})
    cp = {
        "batches": [r for r in batch_reports if r is not None],
        "corrections": corrections,
    }
    checkpoint_file.write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")
    log.info("validation checkpoint saved: %d batches", len(cp["batches"]))


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


def get_translation_file(session: Path, target_language: str) -> Path:
    """Return the best available translation file: fix > original.

    Downstream stages (SRT generation, split_audio, TTS, etc.) should
    call this instead of hard-coding the filename.
    """
    fix = session / "metadata" / f"translation.fix.{target_language}.json"
    if fix.exists():
        return fix
    return session / "metadata" / f"translation.{target_language}.json"


def translate_asr(
    asr_file: Path,
    session: Path,
    settings: dict[str, str],
    source: SourceConfig,
    *,
    progress_callback: Callable[[int, str], None] | None = None,
) -> Path:
    output_file = session / "metadata" / f"translation.{source.target_language}.json"
    fix_file = session / "metadata" / f"translation.fix.{source.target_language}.json"
    checkpoint_file = session / "metadata" / f"translation.{source.target_language}.checkpoint.json"
    validation_checkpoint = session / "metadata" / "validation.checkpoint.json"

    data = json.loads(asr_file.read_text(encoding="utf-8"))
    utterances = data["result"]["utterances"]
    texts = [u["text"].strip() for u in utterances]
    full_text = _full_text(data, texts)
    meta = _read_meta(session)

    api = {key: settings[key] for key in API_SETTING_KEYS if key in settings}
    pre_file = session / "metadata" / "preprocess.json"

    # Load cached preprocess output if available
    if pre_file.exists():
        try:
            pre = PreprocessResponse.model_validate_json(pre_file.read_text(encoding="utf-8"))
            log.info("preprocess loaded from cache: %s", pre_file.name)
        except Exception as exc:
            log.warning("failed to load preprocess cache, re-running: %s", exc)
            pre = preprocess(full_text, meta, source, **api)
            pre_file.write_text(
                json.dumps(pre.model_dump(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    else:
        pre = preprocess(full_text, meta, source, **api)
        pre_file.write_text(
            json.dumps(pre.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("preprocess saved to %s", pre_file.name)

    mode = (settings.get("translate_mode") or "sentence").strip().lower()
    if mode not in VALID_MODES:
        mode = "sentence"
    concurrency = _concurrency_from(settings)

    # --- Phase 1: Translate (skip if output already exists) ---
    validation_enabled = (settings.get("validation_enabled") or "").strip().lower() in ("1", "true", "on")
    # Validation is optional and time-consuming; cap translation progress at 50% when enabled.
    _translate_cap = 50 if validation_enabled else 100
    _total = len(texts)

    if not output_file.exists():
        dst_list: list[str] = [""] * len(texts)
        completed: set[int] = set()
        if checkpoint_file.exists():
            try:
                cp = json.loads(checkpoint_file.read_text(encoding="utf-8"))
                for k, v in cp.items():
                    idx = int(k)
                    if 0 <= idx < len(dst_list) and v:
                        dst_list[idx] = v
                        completed.add(idx)
                if completed:
                    log.info("checkpoint loaded: %d/%d sentences already translated", len(completed), len(texts))
            except Exception as exc:
                log.warning("failed to load checkpoint, starting fresh: %s", exc)

        def _save_checkpoint() -> None:
            cp = {str(i): dst_list[i] for i in completed if dst_list[i]}
            checkpoint_file.write_text(json.dumps(cp, ensure_ascii=False), encoding="utf-8")
            log.info("checkpoint saved: %d/%d sentences", len(completed), len(texts))

        def _on_progress(idx: int, result: str) -> None:
            dst_list[idx] = result
            completed.add(idx)
            if progress_callback and _total > 0:
                pct = round(len(completed) / _total * _translate_cap)
                preview = (result[:30] + "…") if len(result) > 30 else result
                progress_callback(pct, f"已翻译 {len(completed)}/{_total} 句 | {preview}")

        try:
            if mode == "batch":
                log.info("translate mode=batch")
                # translate_batch_context handles checkpoint saving internally per chunk;
                # completed/dst_list are passed to skip already-translated sentences.
                translate_batch_context(
                    texts, source, meta, pre, **api, concurrency=concurrency,
                    on_progress=_on_progress,
                    completed=completed,
                    dst_list=dst_list,
                    checkpoint_file=checkpoint_file,
                )
            else:
                log.info("translate mode=sentence")
                translate_batch(
                    texts, source, meta, pre, **api, concurrency=concurrency,
                    on_progress=_on_progress,
                )
        except Exception:
            _save_checkpoint()
            raise

        # Translation done — write output and delete checkpoint
        _write_translation(output_file, texts, dst_list, utterances, source)
        checkpoint_file.unlink(missing_ok=True)
        log.info("translation saved to %s (%d sentences)", output_file.name, len(texts))
    else:
        log.info("translation file exists, skipping: %s", output_file.name)
        dst_list = [item["dst"] for item in json.loads(output_file.read_text(encoding="utf-8"))["translation"]]
        if progress_callback:
            progress_callback(_translate_cap, f"已翻译 {_total}/{_total} 句")

    # --- Phase 2: Validate and correct (resumable, optional) ---
    if validation_enabled:
        if fix_file.exists():
            log.info("validation fix file already exists, skipping: %s", fix_file.name)
        else:
            threshold = DEFAULT_VALIDATION_THRESHOLD
            raw_threshold = (settings.get("validation_threshold") or "").strip()
            if raw_threshold.isdigit():
                threshold = max(0, min(100, int(raw_threshold)))

            max_corrections = DEFAULT_MAX_CORRECTIONS
            raw_max = (settings.get("max_corrections") or "").strip()
            if raw_max.isdigit():
                max_corrections = max(0, int(raw_max))

            log.info("validation enabled: threshold=%d, max_corrections=%d", threshold, max_corrections)

            def _report_validation(pct_50_100: int, msg: str) -> None:
                if progress_callback is not None:
                    progress_callback(pct_50_100, msg)

            dst_list, validation_report = validate_and_correct(
                texts, dst_list, source, meta, pre,
                **api,
                threshold=threshold,
                max_corrections=max_corrections,
                checkpoint_file=validation_checkpoint,
                progress_callback=_report_validation,
            )
            # Save validation report
            val_file = session / "metadata" / "validation.json"
            val_file.write_text(
                json.dumps(validation_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            log.info(
                "validation saved to %s (score=%d, corrected=%d)",
                val_file.name,
                validation_report["overall_score"],
                validation_report["corrected_count"],
            )
            # Write fix file if any corrections were made
            if validation_report["corrected_count"] > 0:
                _write_translation(fix_file, texts, dst_list, utterances, source)
                log.info("fix file saved to %s", fix_file.name)
            # Delete validation checkpoint
            validation_checkpoint.unlink(missing_ok=True)

    _generate_srt_if_needed(session, source.target_language)
    if progress_callback is not None:
        progress_callback(100, "翻译完成")
    return get_translation_file(session, source.target_language)


def _write_translation(
    path: Path,
    texts: list[str],
    dst_list: list[str],
    utterances: list[dict[str, Any]],
    source: SourceConfig,
) -> None:
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
    path.write_text(
        json.dumps({"translation": translation}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _generate_srt_if_needed(session: Path, target_language: str) -> None:
    """Generate SRT from the best available source: fix file > original."""
    from .ffmpeg import write_srt
    src = get_translation_file(session, target_language)
    srt_file = session / "metadata" / f"subtitles.{target_language}.srt"
    if not srt_file.exists():
        write_srt(src, session)
