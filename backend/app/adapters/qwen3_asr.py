"""Qwen3-ASR speech recognition adapter (transformers backend).

Uses the ``qwen-asr`` package's transformers backend for local inference on
Windows/Linux without requiring vLLM.  Timestamps are produced by the
companion ``Qwen3-ForcedAligner-0.6B`` model.

Install: ``pip install -U qwen-asr``
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import gc

from pydub import AudioSegment

from ._time_utils import to_ms as _to_ms
from ._lang_map import LANG_TO_QWEN3 as _LANG_MAP

logger = logging.getLogger(__name__)

_MODEL = None
_ALIGNER = None


def _model_dir() -> str | None:
    """Return the local model directory if it exists, else ``None``."""
    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / "models" / "Qwen" / "Qwen3-ASR-1.7B"
        if candidate.is_dir():
            return str(candidate)
    # Fallback: let qwen-asr download automatically.
    return None


def _aligner_dir() -> str | None:
    """Return the local ForcedAligner directory if it exists."""
    cache = os.getenv("MODEL_CACHE_DIR", "").strip()
    if cache:
        candidate = Path(cache) / "models" / "Qwen" / "Qwen3-ForcedAligner-0.6B"
        if candidate.is_dir():
            return str(candidate)
    return None


def _resolve_device() -> str:
    device = os.getenv("DEVICE", "cuda").strip()
    if device == "auto":
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        return "cuda:0"
    return device


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    import torch
    from qwen_asr import Qwen3ASRModel

    model_path = _model_dir() or "Qwen/Qwen3-ASR-1.7B"
    device = _resolve_device()
    aligner_path = _aligner_dir() or "Qwen/Qwen3-ForcedAligner-0.6B"

    logger.info("Loading Qwen3-ASR from %s on %s", model_path, device)
    _MODEL = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=8,
        max_new_tokens=2048,
        forced_aligner=aligner_path,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map=device,
        ),
    )
    return _MODEL


def release_model() -> None:
    """Release Qwen3-ASR and ForcedAligner models from GPU memory.

    Call this after ``recognize_speech`` completes to free VRAM when the
    model is no longer needed.  Subsequent calls to ``recognize_speech``
    will reload the models automatically.

    同时清理由模型加载期间 HuggingFace accelerate 的 ``device_map`` 机制
    在 Windows 上 spawn 出的孤儿子进程。这些子进程持有 PyTorch CUDA context，
    如果不清理会导致后续阶段（如 VoxCPM TTS）因显存不足而 OOM。
    """
    global _MODEL, _ALIGNER
    if _MODEL is None and _ALIGNER is None:
        return
    _MODEL = None
    _ALIGNER = None
    # Multiple GC passes are needed because PyTorch models often contain
    # reference cycles between modules, tensors, and autograd graphs.
    for _ in range(3):
        gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

    # ---- 清理由 device_map 机制 spawn 出的孤儿子进程 -----------------------
    # 在 Windows 上，HuggingFace accelerate 的 infer_auto_device_map 通过
    # multiprocessing.spawn 创建子进程来执行设备映射计算。这些子进程在模型
    # 加载完成后不会被自动清理，仍持有 CUDA context 和大量显存。
    #
    # 特征：命令行包含 "multiprocessing.spawn" 且父进程为当前进程。
    # 注意：此清理只针对 spawn 方式创建的子进程，不通过 PID 去杀。
    _cleanup_spawn_children()
    logger.info("Qwen3-ASR models released from GPU memory")


def _cleanup_spawn_children() -> None:
    """终止当前进程的所有 multiprocessing.spawn 子进程。

    仅处理命令行中包含 ``multiprocessing.spawn`` 的孤儿子进程，不会误伤
    其他正常的 Python 子进程（如 uvicorn 的 reload worker 等）。
    """
    try:
        import psutil
    except ImportError:
        return

    try:
        current = psutil.Process()
        for child in current.children(recursive=False):
            try:
                cmdline = child.cmdline()
                if any("multiprocessing.spawn" in part for part in cmdline):
                    child.terminate()
                    child.wait(timeout=5)
                    logger.info(
                        "Terminated orphan spawn child PID=%d", child.pid
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied,
                    psutil.TimeoutExpired) as exc:
                logger.warning(
                    "Failed to clean up spawn child PID=%d: %s", child.pid, exc
                )
    except Exception as exc:
        logger.warning("Orphan spawn child cleanup skipped: %s", exc)


# ---- 断句防御阈值 -----------------------------------------------------------
# 时间间隔阈值：相邻词 end_time → start_time 间隔超过此值即断句
_TIME_GAP_THRESHOLD_MS = 1500
# 一级防御：一个 utterance 超过此词数后在下一个任意标点处断句
_DEFENSE1_MAX_WORDS = 40
# 二级防御：一个 utterance 超过此词数后强制每 N 个词断句
_DEFENSE2_MAX_WORDS = 60

# 所有标点（含句末标点和逗号等软标点），用于防御规则检测
_PROTECTION_PUNCT = set(".!?,;:\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a")


def _is_cjk_first(text: str) -> bool:
    """Return True if the first character of *text* is CJK."""
    if not text:
        return False
    code = ord(text[0])
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def _words_chunk_to_utterance(words: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从词列表构建一个 utterance。

    ``text`` 由词文本拼接：CJK 直接拼接，非 CJK 用空格分隔。
    ``start_time`` / ``end_time`` 取自首尾词。
    """
    if not words:
        return None
    use_space = not _is_cjk_first(words[0].get("text", ""))
    text = " ".join(w["text"] for w in words) if use_space else "".join(w["text"] for w in words)
    return {
        "text": text,
        "start_time": words[0]["start_time"],
        "end_time": words[-1]["end_time"],
        "words": words,
    }


def _split_words_by_time_gap(
    words: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """按词间时间间隔切分。相邻词间隔 > ``_TIME_GAP_THRESHOLD_MS`` 时断句。"""
    if len(words) <= 1:
        return [words] if words else []

    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = [words[0]]
    for prev, cur in zip(words, words[1:]):
        gap = cur["start_time"] - prev["end_time"]
        if gap > _TIME_GAP_THRESHOLD_MS:
            chunks.append(current)
            current = [cur]
        else:
            current.append(cur)
    if current:
        chunks.append(current)
    return chunks


def _split_words_at_punctuation(
    words: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """一级防御：超过 ``_DEFENSE1_MAX_WORDS`` 词时在下一个含任意标点的词处断句。

    从第 ``_DEFENSE1_MAX_WORDS`` 个词开始向后扫描，找到第一个词文本中含
    任意标点（``_PROTECTION_PUNCT``）的词，在此处切分。递归处理剩余词。
    """
    if len(words) <= _DEFENSE1_MAX_WORDS:
        return [words]

    # 从阈值位置向后扫描，找含标点的词
    for i in range(_DEFENSE1_MAX_WORDS, len(words)):
        token = words[i].get("text", "")
        if any(ch in _PROTECTION_PUNCT for ch in token):
            # 在此处切分（标点词归入前半段）
            first = words[:i + 1]
            rest = words[i + 1:]
            return [first] + _split_words_at_punctuation(rest)

    # 没找到含标点的词 → 整体返回，留给二级防御
    return [words]


def _force_split_words(
    words: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """二级防御：超过 ``_DEFENSE2_MAX_WORDS`` 词时强制按步长切分。"""
    if len(words) <= _DEFENSE2_MAX_WORDS:
        return [words]

    chunks: list[list[dict[str, Any]]] = []
    for i in range(0, len(words), _DEFENSE2_MAX_WORDS):
        chunks.append(words[i:i + _DEFENSE2_MAX_WORDS])
    return chunks


def _apply_protection_rules(
    utterances: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """对每个 utterance 的 words 串行应用三重防御规则，展平后重建 utterance。

    规则顺序：时间间隔断句 → 一级标点防御 → 二级强制防御。
    后者兜底前者遗漏的情况。

    当防御规则未触发切分时，保留原始 utterance 不变（包含标点的原始 text）。
    仅当实际发生切分后，才通过词列表重建 text。
    """
    if not utterances:
        return utterances

    result: list[dict[str, Any]] = []
    for u in utterances:
        words: list[dict[str, Any]] = u.get("words", [])
        if not words:
            result.append(u)
            continue

        # 规则 1：时间间隔断句
        time_chunks = _split_words_by_time_gap(words)

        # 对所有时间块串行应用规则 2 + 3
        all_chunks: list[list[dict[str, Any]]] = []
        for chunk in time_chunks:
            punct_chunks = _split_words_at_punctuation(chunk)
            for pc in punct_chunks:
                forced = _force_split_words(pc)
                all_chunks.extend(forced)

        # 如果最终只有 1 个 chunk（未触发任何切分），保留原始 utterance
        if len(all_chunks) == 1:
            result.append(u)
            continue

        # 发生了切分 → 从词列表重建 utterance
        for fc in all_chunks:
            utt = _words_chunk_to_utterance(fc)
            if utt:
                result.append(utt)

    return result


def _split_full_text_to_sentence_ranges(full_text: str) -> list[tuple[int, int]]:
    """按标点将 *full_text* 拆分为句子字符范围。

    返回 ``(start_char, end_char)`` 元组列表，每个元组表示一个句子在
    *full_text* 中的闭区间字符范围。句子边界由句末标点决定，带小数点保护
    （``digit.digit`` 不视为句子边界）。

    末尾无句末标点的剩余文本作为最后一个范围返回。
    """
    _SENTENCE_PUNCT = set(".!?\u3002\uff01\uff1f\uff1b\uff1a")

    ranges: list[tuple[int, int]] = []
    sentence_start = 0

    for i, ch in enumerate(full_text):
        if ch in _SENTENCE_PUNCT:
            # 小数点保护：digit.digit 不视为句子边界
            if ch == "." and i > 0 and i + 1 < len(full_text):
                if full_text[i - 1].isdigit() and full_text[i + 1].isdigit():
                    continue
            ranges.append((sentence_start, i))
            sentence_start = i + 1

    # 结尾无句末标点的剩余文本
    if sentence_start < len(full_text):
        ranges.append((sentence_start, len(full_text) - 1))

    return ranges


def _group_words_to_utterances(
    words: list[dict[str, Any]],
    full_text: str,
) -> list[dict[str, Any]]:
    """Group word-level timestamp items into sentence-level utterances.

    The ForcedAligner returns word timestamps *without* punctuation tokens.
    We first strip all punctuation from ``full_text`` to create a *cleaned*
    text, then match each word token sequentially against the cleaned text to
    determine each word's character position.  Sentence boundaries are derived
    from sentence-ending punctuation in the original ``full_text`` (with
    decimal-point protection for numbers like ``4.5``).

    This approach works correctly for CJK languages (Japanese, Chinese, etc.)
    where words are NOT separated by whitespace, as well as for space-delimited
    languages like English.
    """
    if not words:
        return []

    _ALL_PUNCT = set(".!?,;:\u3002\uff0c\u3001\uff01\uff1f\uff1b\uff1a")
    _SENTENCE_PUNCT = set(".!?\u3002\uff01\uff1f\uff1b\uff1a")

    # ------------------------------------------------------------------
    # Step 1: Build cleaned text (no punctuation) and record the mapping
    #         from cleaned-text positions back to original-text positions.
    # ------------------------------------------------------------------
    cleaned_chars: list[str] = []
    clean_to_orig: list[int] = []  # clean_pos → orig_pos
    for oi, ch in enumerate(full_text):
        if ch not in _ALL_PUNCT:
            cleaned_chars.append(ch)
            clean_to_orig.append(oi)
    cleaned = "".join(cleaned_chars)

    # ------------------------------------------------------------------
    # Step 2: Split full_text into sentence character ranges by
    #         sentence-ending punctuation (pure text-driven, independent
    #          of word matching).
    # ------------------------------------------------------------------
    sentence_ranges = _split_full_text_to_sentence_ranges(full_text)

    # ------------------------------------------------------------------
    # Step 3: Match each word against the cleaned text sequentially,
    #         recording its position in the original text.
    # ------------------------------------------------------------------
    word_positions: list[tuple[dict[str, Any], int]] = []  # (word, orig_pos)
    ci = 0  # current position in cleaned text
    for word in words:
        token = word["text"]
        if token in _ALL_PUNCT:
            continue  # skip punctuation tokens from the aligner

        matched = False
        while ci < len(cleaned):
            remaining_clean = cleaned[ci:]
            if remaining_clean.startswith(token):
                orig_pos = clean_to_orig[ci]
                word_positions.append((word, orig_pos))
                ci += len(token)
                matched = True
                break
            # Fallback: token might be a sub-word fragment that doesn't
            # match the cleaned text exactly (e.g., Japanese morphemes
            # split differently by the aligner).  Skip one character
            # and try again.
            ci += 1

        if not matched:
            # Exhausted cleaned text — append word at the last known
            # position so it is not lost.
            last_pos = word_positions[-1][1] if word_positions else 0
            word_positions.append((word, last_pos))

    if not word_positions:
        return []

    # ------------------------------------------------------------------
    # Step 4: 按句子范围分配词（断句由 full_text 标点驱动，不依赖词匹配）。
    #
    # 核心设计：断句逻辑与 Step 3 的逐词匹配解耦。即使 Step 3 因
    # ForcedAligner tokenization 与模型 full_text 不一致而导致 ci 漂移、
    # orig_pos 错位，句子边界仍由 full_text 中的标点准确决定。
    #
    # 策略：当所有词的 orig_pos 都唯一（Step 3 完全成功）时，
    # 用精确的 orig_pos → 句子映射分配词；否则按各句非标点字符数
    # 比例分配词。两种路径都保证句子不丢失、不合并。
    # ------------------------------------------------------------------
    utterances: list[dict[str, Any]] = []

    # 检测 Step 3 匹配是否完全可靠（每个词有唯一的 orig_pos）
    orig_positions = [op for _, op in word_positions]
    all_reliable = len(set(orig_positions)) == len(orig_positions)

    if all_reliable:
        # ---- 精确路径：按 orig_pos 落入的句子范围分配词 ----
        current_words: list[dict[str, Any]] = []
        current_si = 0  # 当前句子在 sentence_ranges 中的索引

        for word, orig_pos in word_positions:
            # 找到该词 orig_pos 所属的句子范围
            word_si = current_si
            for si in range(current_si, len(sentence_ranges)):
                s_start, s_end = sentence_ranges[si]
                if s_start <= orig_pos <= s_end:
                    word_si = si
                    break

            if word_si != current_si and current_words:
                # 句子切换：输出当前句子
                s_start, s_end = sentence_ranges[current_si]
                sentence_text = full_text[s_start:s_end + 1].strip()
                if sentence_text:
                    utterances.append({
                        "text": sentence_text,
                        "start_time": current_words[0]["start_time"],
                        "end_time": current_words[-1]["end_time"],
                        "words": list(current_words),
                    })
                current_words = []
                current_si = word_si

            current_words.append(word)

        # 输出最后一个句子
        if current_words:
            s_start, s_end = sentence_ranges[current_si]
            sentence_text = full_text[s_start:s_end + 1].strip()
            if sentence_text:
                utterances.append({
                    "text": sentence_text,
                    "start_time": current_words[0]["start_time"],
                    "end_time": current_words[-1]["end_time"],
                    "words": list(current_words),
                })
    else:
        # ---- 比例路径：部分词的 orig_pos 不可靠，按字符比例分配 ----
        # 计算每个句子的非标点字符数
        sentence_chars: list[int] = []
        for s_start, s_end in sentence_ranges:
            chars = sum(
                1 for i in range(s_start, s_end + 1)
                if full_text[i] not in _ALL_PUNCT
            )
            sentence_chars.append(chars)
        total_chars = sum(sentence_chars)

        # 按字符比例计算每个句子的词配额（每句至少 1 个词）
        total_words = len(word_positions)
        quotas: list[int] = []
        remaining = total_words
        for i in range(len(sentence_ranges)):
            if i == len(sentence_ranges) - 1:
                quotas.append(remaining)
            else:
                if total_chars > 0:
                    prop = sentence_chars[i] / total_chars
                else:
                    prop = 1.0 / len(sentence_ranges)
                q = max(1, round(prop * total_words))
                # 不能超过剩余词数（为后续句子至少各留 1 个）
                q = min(q, remaining - (len(sentence_ranges) - i - 1))
                quotas.append(q)
                remaining -= q

        # 按配额分配词到各句子
        wi = 0
        for si, (s_start, s_end) in enumerate(sentence_ranges):
            bucket: list[dict[str, Any]] = []
            while wi < total_words and len(bucket) < quotas[si]:
                bucket.append(word_positions[wi][0])
                wi += 1

            if bucket:
                sentence_text = full_text[s_start:s_end + 1].strip()
                utterances.append({
                    "text": sentence_text,
                    "start_time": bucket[0]["start_time"],
                    "end_time": bucket[-1]["end_time"],
                    "words": bucket,
                })

        # 剩余未分配词追加到最后一个句子
        if wi < total_words:
            extra = [word_positions[i][0] for i in range(wi, total_words)]
            utterances[-1]["words"].extend(extra)
            utterances[-1]["end_time"] = extra[-1]["end_time"]

    # ---- 后处理：应用三重防御规则（时间间隔 / 标点 / 强制断句） ----
    utterances = _apply_protection_rules(utterances)

    return utterances


def _convert_result(result: Any, duration_ms: int) -> dict[str, Any]:
    """Convert a Qwen3-ASR transcribe result to YouDub asr.json payload."""
    text = getattr(result, "text", "") or ""
    language = getattr(result, "language", "") or ""
    time_stamps = getattr(result, "time_stamps", None) or []

    # Convert ForcedAligner timestamps to word items.
    words: list[dict[str, Any]] = []
    for ts in time_stamps:
        # Each ts has .text, .start_time, .end_time (seconds, float).
        token = str(getattr(ts, "text", "")).strip()
        if not token:
            continue
        start_s = float(getattr(ts, "start_time", 0.0) or 0.0)
        end_s = float(getattr(ts, "end_time", start_s) or start_s)
        words.append({
            "text": token,
            "start_time": _to_ms(start_s),
            "end_time": _to_ms(end_s),
        })

    utterances = _group_words_to_utterances(words, text)

    # Fallback: no timestamps at all.
    if not utterances:
        utterances = [{
            "text": text,
            "start_time": 0,
            "end_time": duration_ms,
            "words": [],
        }]

    return {
        "audio_info": {"duration": duration_ms},
        "result": {
            "text": text,
            "utterances": utterances,
        },
    }


def recognize_speech(
    vocals_file: Path, session: Path, language: str, *, model_id: str | None = None
) -> Path:
    """Recognize speech using Qwen3-ASR with transformers backend."""
    metadata_dir = session / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_file = metadata_dir / "asr.json"
    if output_file.exists():
        return output_file

    model = _load_model()
    qwen_lang = _LANG_MAP.get(language)  # None → auto-detect

    logger.info("Qwen3-ASR transcribing %s (language=%s)", vocals_file.name, qwen_lang)
    results = model.transcribe(
        audio=str(vocals_file),
        language=qwen_lang,
        return_time_stamps=True,
    )

    if not results:
        raise RuntimeError("Qwen3-ASR did not return any results.")

    result = results[0]
    del results  # release batch reference early

    # Save raw result for debugging.
    raw_output = metadata_dir / "asr.raw.qwen3asr.json"
    raw_data = {
        "text": getattr(result, "text", ""),
        "language": getattr(result, "language", ""),
        "time_stamps": [
            {
                "text": str(getattr(ts, "text", "")),
                "start_time": float(getattr(ts, "start_time", 0.0) or 0.0),
                "end_time": float(getattr(ts, "end_time", 0.0) or 0.0),
            }
            for ts in (getattr(result, "time_stamps", None) or [])
        ],
    }
    raw_output.write_text(
        json.dumps(raw_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    duration_ms = len(AudioSegment.from_file(vocals_file))
    payload = _convert_result(result, duration_ms)
    del result  # release ASR result (holds aligner output) early

    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_file
