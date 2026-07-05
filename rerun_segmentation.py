"""只跑断句：从 asr.raw.qwen3asr.json 重新生成 asr.json，不跑 ASR 推理。"""
import json, sys
sys.path.insert(0, ".")
from backend.app.adapters.qwen3_asr import _convert_result
from pathlib import Path

SESSION = Path(r"g:\FFOutput\youdub\硅谷101\美国AI研究员的中国之旅_年轻人_追赶者_算力焦虑与_AGI展示厅_专访Nathan_Lambert_101视频播客__BV1HXTs6bEzH")
RAW = SESSION / "metadata" / "asr.raw.qwen3asr.json"
OUT = SESSION / "metadata" / "asr.json"

raw = json.loads(RAW.read_text("utf-8"))

# 从 raw 构造 _convert_result 需要的 mock result 对象
class _TS:
    def __init__(self, d):
        self.text = d["text"]
        self.start_time = d["start_time"]
        self.end_time = d.get("end_time", d["start_time"])

class _Result:
    def __init__(self, raw):
        self.text = raw["text"]
        self.language = raw.get("language", "")
        self.time_stamps = [_TS(t) for t in raw["time_stamps"]]

result = _Result(raw)

# duration 从旧 asr.json 读取
old = json.loads(OUT.read_text("utf-8"))
duration_ms = old["audio_info"]["duration"]

payload = _convert_result(result, duration_ms)
OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# 打印前 5 句
utts = payload["result"]["utterances"]
print(f"Total: {len(utts)} utterances")
for i, u in enumerate(utts[:5]):
    wt = [w["text"] for w in u["words"]]
    t = u["text"]
    print(f"  [{i}] {t[:80]}")
    print(f"       time={u['start_time']}-{u['end_time']}  words({len(u['words'])}): {wt[:15]}")
