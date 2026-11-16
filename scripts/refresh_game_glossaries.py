"""从双语资料站和公开结构化数据生成 YouDub 本地术语库。"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen


class Sources:
    def __init__(self, cache: Path | None):
        self.cache = cache

    def get(self, url: str, cache_key: str) -> str:
        if self.cache and (self.cache / cache_key).is_file():
            return (self.cache / cache_key).read_text(encoding="utf-8-sig")
        request = Request(url, headers={"User-Agent": "YouDub glossary updater/1.0"})
        with urlopen(request, timeout=25) as response:
            raw = response.read(32_000_001)
        if len(raw) > 32_000_000:
            raise ValueError(f"资料超过大小限制：{url}")
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
            (self.cache / cache_key).write_bytes(raw)
        return raw.decode("utf-8-sig")


class Dictionary:
    def __init__(self):
        self.terms = defaultdict(set)
        self.effects = defaultdict(set)
        self.sources = []

    def add(self, category, urls, english, chinese, *, minimum, effects=False):
        # 每个源先按同一游戏记录的 ID 配对，再检查当前语言的字段。
        output = self.effects if effects else self.terms
        count = 0
        for key in english.keys() & chinese.keys():
            en = " ".join(english[key].split())
            zh = " ".join(chinese[key].split())
            if not (len(en) <= 420 and len(zh) <= 420 and re.search(r"[A-Za-z]", en)
                    and re.search(r"[\u3400-\u9fff]", zh)):
                continue
            if effects and en.count("#") != zh.count("#"):
                continue
            output[en].add(zh)
            count += 1
        if count < minimum:
            raise ValueError(f"{category} 只有 {count} 个可配对条目，检查源结构：{urls[0]}")
        self.sources.append({"category": category, "urls": [*urls], "matched_records": count})

    def build(self):
        # 同名异译没有足够上下文，留给上层结合完整句子处理。
        return {
            "schema_version": 1,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "locale": "zh-CN",
            "sources": self.sources,
            "terms": {key: next(iter(values)) for key, values in self.terms.items() if len(values) == 1},
            "effects": {key: next(iter(values)) for key, values in self.effects.items() if len(values) == 1},
        }


class PoEEntries(HTMLParser):
    def __init__(self, category):
        super().__init__()
        self.category = category
        self.current = None
        self.records = defaultdict(set)
        self.in_type_line = False

    def handle_starttag(self, tag, items):
        attrs = dict(items)
        if tag == "span" and self.current and "uniqueTypeLine" in (attrs.get("class") or "").split():
            self.in_type_line = True
        if tag != "a":
            return
        self.in_type_line = False
        classes = set((attrs.get("class") or "").split())
        hover = unquote(attrs.get("data-hover") or "")
        valid = (
            (self.category in ("gems", "support") and "GemEffects" in hover)
            or (self.category == "unique" and "UniqueItem" in classes)
            or (self.category == "currency" and classes & {"currencyitem", "item_currency"})
            or (self.category == "keystones" and "PassiveSkills" in hover)
        )
        path = urlsplit(attrs.get("href") or "")
        parts = path.path.strip("/").split("/")
        if len(parts) == 2 and parts[0] in {"cn", "us"}:
            parts = parts[1:]
        self.current = (unquote(parts[0]), []) if valid and len(parts) == 1 and parts[0] and not path.netloc else None

    def handle_data(self, data):
        if self.current and not self.in_type_line:
            self.current[1].append(data)

    def handle_endtag(self, tag):
        if tag == "span" and self.in_type_line:
            self.in_type_line = False
        if tag == "a" and self.current:
            slug, words = self.current
            text = " ".join("".join(words).split())
            if text:
                self.records[slug].add(text)
            self.current = None

    def result(self):
        return {key: next(iter(values)) for key, values in self.records.items() if len(values) == 1}


def poe_entries(page, category):
    parser = PoEEntries(category)
    parser.feed(page)
    parser.close()
    return parser.result()


def trade_stats(payload):
    return {
        entry["id"]: entry["text"]
        for group in json.loads(payload)["result"] if group["id"] != "pseudo"
        for entry in group["entries"] if not entry.get("option")
    }


def poe(source: Sources, number: int):
    site = "https://poedb.tw" if number == 1 else "https://poe2db.tw"
    trade = "trade" if number == 1 else "trade2"
    pack = Dictionary()
    for category, page, minimum in (
        ("gems", "Skill_Gems", 100),
        ("support", "Support_Gems", 70),
        ("unique", "Unique_item", 100),
        ("currency", "Currency", 15),
        ("keystones", "Keystone", 10),
    ):
        en, zh = f"{site}/us/{page}", f"{site}/cn/{page}"
        en_entries = poe_entries(source.get(en, f"poe{number}_en_{category}"), category)
        zh_entries = poe_entries(source.get(zh, f"poe{number}_zh_{category}"), category)
        pack.add(category, (en, zh), en_entries, zh_entries, minimum=minimum)
    en = f"https://www.pathofexile.com/api/{trade}/data/stats"
    zh = f"https://poe.game.qq.com/api/{trade}/data/stats"
    en_stats = trade_stats(source.get(en, f"poe{number}_en_stats"))
    zh_stats = trade_stats(source.get(zh, f"poe{number}_zh_stats"))
    pack.add("trade_stats", (en, zh), en_stats, zh_stats, minimum=1000, effects=True)
    return pack


def d4_page(html: str, category: str):
    # 资料站的公开页面附带渲染列表用的游戏记录，包含跨语言不变的 ID。
    for match in re.finditer(r'<script[^>]*>self\.__next_f\.push\((\[1,.*?\])\)</script>', html, re.S):
        payload = json.loads(match[1])[1]
        marker = re.search(r'"pageKey"\s*:\s*"' + re.escape(category) + r'"[^{}]*"records"\s*:\s*(?=\[)', payload)
        if marker:
            return json.JSONDecoder().raw_decode(payload[marker.end():])[0]
    raise ValueError(f"OP.GG 的 {category} 页面没有可识别的游戏记录")


def d4(source: Sources):
    from concurrent.futures import ThreadPoolExecutor

    pack = Dictionary()
    for category, minimum in (
        ("skills", 100), ("unique-items", 100), ("aspects", 100), ("affixes", 100),
        ("glyphs", 50), ("runes", 15), ("runewords", 15), ("paragon-boards", 15),
        ("tempering", 50), ("gems", 20), ("charms", 100), ("soul-thorns", 15),
    ):
        en = f"https://op.gg/diablo4/{category}"
        zh = f"https://op.gg/zh-cn/diablo4/{category}"
        first = source.get(en, f"d4_en_{category}")
        pages = re.search(r'aria-label="Page 1 of (\d+)"', first)
        pages = int(pages[1]) if pages else 1
        season = re.search(r'season=(\d+)', first)
        if not 1 <= pages <= 40:
            raise ValueError(f"OP.GG 的 {category} 分页格式已变更")

        def load_page(task):
            lang, url, number = task
            if lang == "en" and number == 1:
                html = first
            else:
                paged = f"{url}?page={number}" if number > 1 else url
                if number > 1 and season:
                    paged += f"&season={season[1]}"
                suffix = f"_{number}" if number > 1 else ""
                html = source.get(paged, f"d4_{lang}_{category}{suffix}")
            return lang, d4_page(html, category)

        first_rows = {"en": {}, "zh": {}}
        tasks = [(lang, url, i) for i in range(1, pages + 1) for lang, url in (("en", en), ("zh", zh))]
        with ThreadPoolExecutor(max_workers=3) as pool:
            for lang, rows in pool.map(load_page, tasks):
                for row in rows:
                    if row["id"] in first_rows[lang]:
                        raise ValueError(f"OP.GG 的 {category} 分页包含重复 ID")
                    first_rows[lang][row["id"]] = row
        english, chinese = first_rows["en"], first_rows["zh"]
        if len(english.keys() & chinese.keys()) < max(len(english), len(chinese)) * 0.90:
            raise ValueError(f"OP.GG 的 {category} 两种语言记录没有对齐")
        if category != "affixes":
            pack.add(category, (en, zh), {k: v["name"] for k, v in english.items()},
                     {k: v["name"] for k, v in chinese.items()}, minimum=minimum)
        if category == "tempering":
            continue
        en_effect = {k: effect_text(v) for k, v in english.items()}
        zh_effect = {k: effect_text(v) for k, v in chinese.items()}
        pack.add(f"{category}_effects", (en, zh), en_effect, zh_effect,
                 minimum=minimum if category == "affixes" else 1, effects=True)
    return pack


def effect_text(record):
    summary = record.get("summary") or ""
    return "" if summary.startswith("$") else re.sub(r"<[^>]+>", "", summary)


def last_epoch(source: Sources):
    site = "https://www.lastepochtools.com"
    page = source.get(f"{site}/skills/zh/", "le_skills_zh")
    version = re.search(r"/data/(version\d+)/skills/", page)
    marker = re.search(r"\['langGameCacheMarker'\]='(\d+)'", page)
    if not version or not marker:
        raise ValueError("最后纪元资料站的本地化版本格式已变更")
    root = f"{site}/data/{version[1]}/i18n/full"
    en, zh = f"{root}/en.json?{marker[1]}", f"{root}/zh.json?{marker[1]}"
    english = json.loads(source.get(en, "le_full_en"))
    chinese = json.loads(source.get(zh, "le_full_zh"))
    pack = Dictionary()
    for category, expression, minimum, effects in (
        ("skills", r"Abilities\..+_Name$", 100, False),
        ("skill_nodes", r"Skills\..+_Name$", 100, False),
        ("items", r"Item_Names\.(?:Unique_Name_|Item_(?:SubType|BaseType)_|ItemSet_\d+_Name$)", 100, False),
        ("affix_names", r"Item_Affixes\.Item_Affix_\d+_DisplayName$", 100, False),
        ("properties", r"Properties\.Property_Master_.+_Name$", 25, False),
        ("ailments", r"Ailments\..+_Name$", 25, False),
        ("endgame_names", r"(?:Monolith_Timelines\..+_Name_|MapObjects\.Dungeon_ID_)", 20, False),
        ("ability_effects", r"Abilities\..+_Description$", 100, True),
        ("ailment_effects", r"Ailments\..+_Description$", 25, True),
        ("skill_modifiers", r"Skills\..+_Stat$", 100, True),
        ("unique_effects", r"Item_Names\.Unique_Tooltip_", 25, True),
    ):
        pattern = re.compile(expression)
        selected_en = {key: clean_le(value) for key, value in english.items() if pattern.search(key) and isinstance(value, str)}
        selected_zh = {key: clean_le(chinese[key]) for key in selected_en if isinstance(chinese.get(key), str)}
        pack.add(category, (en, zh), selected_en, selected_zh, minimum=minimum, effects=effects)
    return pack


def clean_le(value):
    value = value.replace("''", "'")
    return re.sub(r"(?:<[^>]+>|'?\{|\}'?)", "", value)


def save_pack(path: Path, payload: dict):
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        # 保留人工核对的译名、别名和来源，只刷新资料站数据。
        for field in ("name", "aliases", "fallback_terms", "overrides", "curated_sources", "curated_locale"):
            if field in previous:
                payload[field] = previous[field]
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".glossary-", delete=False) as output:
        temporary = Path(output.name)
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    import sys
    from functools import partial

    importers = {"poe1": partial(poe, number=1), "poe2": partial(poe, number=2),
                 "last_epoch": last_epoch, "diablo4": d4}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game", nargs="*", metavar="游戏", help="poe1 / poe2 / last_epoch / diablo4，默认全部")
    parser.add_argument("--cache-dir", type=Path, help="可选的原始资料快照目录，已有缓存会被复用")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "backend" / "app" / "glossaries")
    args = parser.parse_args()
    games = args.game or list(importers)
    if unknown := set(games) - importers.keys():
        parser.error("未知游戏：" + ", ".join(sorted(unknown)))
    source = Sources(args.cache_dir)
    failed = False
    for game in dict.fromkeys(games):
        try:
            pack = importers[game](source).build()
            save_pack(args.output_dir / f"{game}.json", pack)
            print(f"{game}：{len(pack['terms'])} 项译名、{len(pack['effects'])} 条效果", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{game} 更新失败，已保留旧词库：{exc}", file=sys.stderr, flush=True)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
