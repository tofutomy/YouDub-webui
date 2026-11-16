import io
import json
from unittest.mock import Mock

from backend.app.adapters.youtube_assets import CAPTION, CAPTION_INFO, COVER, apply_caption_corrections, download_youtube_assets
from backend.app.glossary import select_glossary
from backend.app.adapters.youtube_assets import read_cues, correct_terms


MANUAL = b"""WEBVTT

00:00:00.000 --> 00:00:03.000
Ground Slam into Sunder to be the smoothest.
"""

ROLLING = """WEBVTT
Kind: captions
Language: en

00:03:34.200 --> 00:03:36.792
you can use<00:03:35.000><c> ground slam</c>

00:03:36.792 --> 00:03:39.082
you can use ground slam
into<00:03:37.092><c> Sunder</c><00:03:37.492><c> to</c><00:03:37.692><c> be the smoothest</c>

00:03:39.082 --> 00:03:39.092
into Sunder to be the smoothest
"""


def test_download_youtube_assets_prefers_manual_and_preserves_original_jpeg(tmp_path):
    urls = {
        "https://example.invalid/manual": MANUAL,
        "https://example.invalid/cover.jpg": b"\xff\xd8jpeg\xff\xd9",
    }
    info = {
        "subtitles": {"en-US": [{"ext": "vtt", "url": "https://example.invalid/manual"}]},
        "automatic_captions": {
            "en": [{"ext": "vtt", "url": "https://example.invalid/translated?tlang=en"}],
            "en-orig": [{"ext": "vtt", "url": "https://example.invalid/automatic"}],
        },
        "thumbnails": [{"width": 1920, "height": 1080, "url": "https://example.invalid/cover.jpg"}],
    }
    downloader = Mock()
    downloader.urlopen = Mock(side_effect=lambda url: io.BytesIO(urls[url]))
    download_youtube_assets(downloader, info, tmp_path)
    assert (tmp_path / CAPTION).read_bytes() == MANUAL
    assert json.loads((tmp_path / CAPTION_INFO).read_text()) == {"kind": "manual", "language": "en-US"}
    assert (tmp_path / COVER).read_bytes() == urls["https://example.invalid/cover.jpg"]
    assert downloader.urlopen.call_count == 2


def test_rolling_native_caption_corrects_game_term_without_changing_timing(tmp_path):
    (tmp_path / "metadata").mkdir()
    (tmp_path / CAPTION).write_text(ROLLING, encoding="utf-8")
    (tmp_path / CAPTION_INFO).write_text('{"kind":"automatic","language":"en-orig"}', encoding="utf-8")
    (tmp_path / "metadata/ytdlp_info.json").write_text('{"title":"PoE 1 guide"}', encoding="utf-8")
    asr = tmp_path / "metadata/asr_fixed.json"
    utterance = {"start_time": 215000, "end_time": 219000, "text": "ground slam into sundar to be the smoothest",
                 "speaker": "1"}
    asr.write_text(json.dumps({"result": {"utterances": [utterance]}}), encoding="utf-8")

    assert apply_caption_corrections(asr, tmp_path, "poe1", "") == 1
    result = json.loads(asr.read_text(encoding="utf-8"))
    assert result["result"]["utterances"] == [{**utterance, "text": "ground slam into Sunder to be the smoothest"}]
    assert result["source_caption_corrections"] == [{"start_time": 215000, "from": "sundar", "to": "Sunder"}]

    other_game = select_glossary("none", {})
    assert not correct_terms([utterance], read_cues(ROLLING, automatic=True), automatic=True, glossary=other_game)
