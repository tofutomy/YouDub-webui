from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StageSpec:
    name: str
    label: str
    # Glob patterns (relative to session dir) for the files this stage
    # produces. Used by clear-stage (delete) and restore-cache (verify).
    # Empty tuple means the stage has no declarable file outputs.
    outputs: tuple[str, ...] = ()


STAGES: tuple[StageSpec, ...] = (
    StageSpec(
        "download",
        "Download",
        outputs=("media/video_source.mp4", "metadata/ytdlp_info.json", "metadata/local_info.json"),
    ),
    StageSpec(
        "separate",
        "Demucs",
        outputs=("media/audio_vocals.wav", "media/audio_bgm.wav"),
    ),
    StageSpec(
        "asr",
        "Whisper",
        outputs=("metadata/asr.json",),
    ),
    StageSpec(
        "asr_fix",
        "Split sentences",
        outputs=("metadata/asr_fixed.json",),
    ),
    StageSpec(
        "translate",
        "Translate",
        # translation/validation/subtitles use glob patterns; handled specially
        # in clear-stage because of the language suffix.
        outputs=(
            "metadata/translation.*.json",
            "metadata/translation.*.checkpoint.json",
            "metadata/subtitles.*.srt",
            "metadata/validation.json",
            "metadata/validation.*.checkpoint.json",
        ),
    ),
    StageSpec(
        "split_audio",
        "Split audio",
        outputs=("segments/vocals/*.wav",),
    ),
    StageSpec(
        "tts",
        "VoxCPM",
        outputs=("segments/tts/*.wav",),
    ),
    StageSpec(
        "merge_audio",
        "Merge audio",
        outputs=("tmp/audio_dubbing.wav", "metadata/timings.json"),
    ),
    StageSpec(
        "merge_video",
        "Merge video",
        outputs=("media/video_final.mp4",),
    ),
)

STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in STAGES)


def stage_outputs(stage_name: str) -> tuple[str, ...]:
    """Return the declared output paths (relative to session) for a stage."""
    for stage in STAGES:
        if stage.name == stage_name:
            return stage.outputs
    return ()


def resolve_stage_outputs(session: "Path", stage_name: str) -> list:
    """Expand a stage's output patterns to concrete Paths inside ``session``.

    Supports ``*`` globs (e.g. ``metadata/translation.*.json``). Patterns
    without a glob are returned as-is (the file may or may not exist).
    """
    from pathlib import Path

    targets: list[Path] = []
    for pattern in stage_outputs(stage_name):
        if "*" in pattern:
            parent, name = pattern.rsplit("/", 1) if "/" in pattern else (".", pattern)
            glob_dir = session / parent if parent != "." else session
            if glob_dir.exists():
                targets.extend(sorted(glob_dir.glob(name)))
        else:
            targets.append(session / pattern)
    return targets

