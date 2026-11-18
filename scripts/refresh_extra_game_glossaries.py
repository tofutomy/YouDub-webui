"""从公开游戏资料刷新扩展词库，需要 beautifulsoup4。"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
from struct import unpack_from
import unicodedata
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen
import zlib

from bs4 import BeautifulSoup

from refresh_game_glossaries import Dictionary, Sources, save_pack


def page(source, url, cache):
    return BeautifulSoup(source.get(url, cache), "html.parser")


def binary_source(source, url, name, limit):
    cached = source.cache / name if source.cache else None
    if cached and cached.is_file():
        return cached.read_bytes()
    with urlopen(Request(url, headers={"User-Agent": "YouDub glossary updater/1.0"}), timeout=25) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError(f"资料超过大小限制：{url}")
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(content)
    return content


def names(rows, field):
    return {row["id"]: row[field] for row in rows if row.get(field)}


def diablo2(source):
    site = "https://wiki.projectdiablo2.cn/wiki/"
    pack = Dictionary()
    for suffix, cache, minimum in (("Skillname", "pd2_Skill", 170), ("Itemname", "pd2_Item", 450),
                                   ("RWname", "pd2_RW", 75), ("Areaname", "pd2_Area", 65),
                                   ("Monname", "pd2_Mon", 40), ("Statename", "pd2names", 40)):
        url = site + "D2" + suffix
        english, chinese = {}, {}
        for i, table in enumerate(page(source, url, cache).select("table.wikitable")):
            rows = table.select("tr")
            if not rows:
                continue
            header = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"], recursive=False)]
            if not {"英文名", "D2R简"}.issubset(header):
                continue
            en, zh = header.index("英文名"), header.index("D2R简")
            for j, row in enumerate(rows[1:]):
                cells = row.find_all("td", recursive=False)
                if len(cells) <= max(en, zh) or any(cell.get("colspan") for cell in cells):
                    continue
                en_name = cells[en].get_text(" ", strip=True)
                if "(PD2 wiki)" in en_name:
                    en_name = en_name.split("(PD2 wiki)", 1)[1].strip() or en_name.split("(PD2 wiki)", 1)[0].strip()
                english[i, j], chinese[i, j] = en_name, cells[zh].get_text(" ", strip=True)
        pack.add(suffix, [url], english, chinese, minimum=minimum)
    return pack


def d3_entries(soup, hero, kind):
    entries = {}
    for a in soup.select("a[href]"):
        path = urlsplit(a["href"]).path
        if re.search(fr"/class/{hero}/{kind}/[^/]+/?$", path):
            text = a.get_text(" ", strip=True)
            if text:
                entries[path.rstrip("/").rsplit("/", 1)[-1]] = text
    for rune in soup.select("span[data-d3tooltip^='rune/']"):
        entries[rune["data-d3tooltip"]] = rune.get_text(" ", strip=True)
    return entries


def diablo3(source):
    pack = Dictionary()
    for hero in ("barbarian", "crusader", "demon-hunter", "monk", "necromancer", "witch-doctor", "wizard"):
        for kind in ("active", "passive"):
            urls = [f"https://eu.diablo3.blizzard.com/{lang}/class/{hero}/{kind}/" for lang in ("en-us", "zh-tw")]
            en, zh = [d3_entries(page(source, url, f"d3_{lang}_{hero}_{kind}"), hero, kind)
                      for url, lang in zip(urls, ("en-us", "zh-tw"))]
            pack.add(f"{hero}_{kind}", urls, en, zh, minimum=12)
    return pack


def diablo_immortal(source):
    pack = Dictionary()
    for category, minimum in (("skills", 100), ("equip-items", 150)):
        urls = ["https://www.wowhead.com/diablo-immortal/" + lang + category for lang in ("", "cn/")]
        found = []
        for lang, url in zip(("en", "cn"), urls):
            soup = page(source, url, f"di_{lang}_{category}")
            rows = json.loads(soup.find("script", id="data.page.listPage.listviews").string)
            found.append(names([row for group in rows for row in group.get("data", [])], "name"))
        pack.add(category, urls, *found, minimum=minimum)
    return pack


def grim_dawn(source):
    base = "https://raw.githubusercontent.com/tednaleid/grimdawn-devotions/main/data/i18n/game"
    urls = [f"{base}.{lang}.json" for lang in ("en", "zh")]
    en, zh = [json.loads(source.get(url, f"grim_{lang}")) for lang, url in zip(("en", "zh"), urls)]
    pack = Dictionary()
    for category, expr, minimum in (
        ("skills_and_devotions", r"^tag(?:.*Skill(?:Name|Class)|Devotion)|^tagClass\d+$", 200),
        ("equipment", r"^tag(?:GDX\d+)?(?:Shield|Dagger|Sword|Axe|Mace|Spear|Ring|Amulet|Medal|Relic|Chest|Weapon|Head|Feet|Legs|Hands|Waist|Shoulder|Focus|ItemSet|Component|Offhand|Armor)", 400),
        ("properties", r"^(?:Damage|Resistance|OffensiveAbility|DefensiveAbility|Cooldown|AttackSpeed|CastingSpeed|MovementSpeed)", 15),
    ):
        keys = {key for key in en.keys() & zh.keys() if re.search(expr, key) and "Description" not in key
                and isinstance(en[key], str) and isinstance(zh[key], str)}
        if category == "properties":
            en_map = {key: re.sub(r"\{[^{}]+\}", "#", en[key]) for key in keys}
            zh_map = {key: re.sub(r"\{[^{}]+\}", "#", zh[key]) for key in keys}
            pack.add(category, urls, en_map, zh_map, minimum=minimum, effects=True)
        else:
            pack.add(category, urls, {key: en[key] for key in keys if len(en[key]) < 95},
                     {key: zh[key] for key in keys if len(zh[key]) < 95}, minimum=minimum)
    return pack


def torchlight_infinite(source):
    pack = Dictionary()
    for category, minimum in (
        ("Active_Skill", 150), ("Support_Skill", 100), ("Passive_Skill", 30),
        ("Activation_Medium_Skill", 15), ("Magnificent_Support_Skill", 55), ("Noble_Support_Skill", 55),
        ("Modularization_Skill", 8), ("Tartarus_Skill", 30), ("Legendary_Gear", 240), ("Pactspirit", 80),
    ):
        urls = [f"https://tlidb.com/{lang}/{category}" for lang in ("en", "cn")]
        found = []
        for lang, url in zip(("en", "cn"), urls):
            soup = page(source, url, f"tli_{lang}_{category}")
            entries = {}
            for link in soup.select("a[data-hover][href]"):
                path = unquote(urlsplit(link["href"]).path).strip("/")
                slug = re.sub(rf"^{lang}/", "", path)
                if slug and "/" not in slug:
                    entries[slug] = link.get_text(" ", strip=True)
            found.append(entries)
        pack.add(category, urls, *found, minimum=minimum)
    return pack


def hero_siege(source):
    base = "https://www.herosiegedata.com/cn/"
    pack = Dictionary()
    roster = page(source, base + "heroes", "hero_zh")
    urls = sorted({urljoin(base, a["href"]) for a in roster.select("a[href^='hero_'][href$='.html']")})

    def read_hero(url):
        slug = url.rsplit("/", 1)[-1]
        raw = source.get(url, "hero_cn_" + slug)
        groups = json.JSONDecoder().raw_decode(raw.split("var SPECS=", 1)[1])[0]
        english, chinese = {}, {}
        for spec in groups:
            for row in spec["skills"]:
                if row.get("e2"):
                    key = row["id"]
                    english[key], chinese[key] = row["e2"], row["n"]
                for sub in row.get("sub", []):
                    if isinstance(sub, dict) and sub.get("e2") and sub.get("n"):
                        key = row["id"], sub.get("id", sub["e2"])
                        english[key], chinese[key] = sub["e2"], sub["n"]
        return url, english, chinese

    if len(urls) < 20:
        raise ValueError("Hero Siege 职业目录缺少条目")
    with ThreadPoolExecutor(max_workers=4) as pool:
        for url, en, zh in pool.map(read_hero, urls):
            pack.add("skills", [url], en, zh, minimum=8)
    url = base + "items_data.json"
    rows = json.loads(source.get(url, "hero_cn_items_data.json"))
    pack.add("equipment", [url], {i: row.get("e", "") for i, row in enumerate(rows)},
             {i: row.get("n", "") for i, row in enumerate(rows)}, minimum=700)
    return pack


def slormancer(source):
    base = "https://raw.githubusercontent.com/cayrac/slorm-planner/master/data/"
    pack = Dictionary()
    for category, minimum in (("cla_0", 100), ("cla_1", 100), ("cla_2", 100), ("leg", 100),
                              ("act", 30)):
        url = f"{base}dat_{category}.json"
        rows = json.loads(base64.b64decode(source.get(url, f"slorm_dat_{category}")))
        english, chinese = {}, {}
        for i, row in enumerate(rows):
            en, zh = row.get("EN_NAME"), row.get("CH_NAME")
            if isinstance(en, str) and isinstance(zh, str) and not any(marker in en or marker in zh for marker in ("$", "%", "{", "<")):
                english[i], chinese[i] = en, zh
        pack.add(category, [url], english, chinese, minimum=minimum)
    return pack


def titan_quest2(source):
    url = "https://www.tq2items.com/site-language.js"
    raw = source.get(url, "tq2_site-language.js")
    data = json.JSONDecoder().raw_decode(raw.split("window.TQ2_NATIVE_TRANSLATIONS=", 1)[1])[0]
    catalog = data["byKey"]["zh-Hans"]
    labels = {value for key, value in catalog.items() if not re.search(r"(?:Description|Tooltip|Lore|Subtitle|Dialogue)", key, re.I)}
    pairs = {en: zh for en, zh in data["bySource"]["zh-Hans"].items() if zh in labels
             and len(en) <= 90 and len(zh) <= 65 and not re.search(r"[<>{}\n]", en + zh)}
    pack = Dictionary()
    pack.add("game_labels", [url], {en: en for en in pairs}, pairs, minimum=350)
    return pack


def read_torchlight2_pairs(raw):
    version, count = unpack_from("<II", raw)
    if version != 2 or not 20000 <= count <= 50000:
        raise ValueError("火炬之光 II 译名文件格式变化")
    strings = {}
    offset = 8
    for _ in range(count):
        key, length = unpack_from("<IH", raw, offset)
        offset += 6
        end = offset + 2 * length
        strings[key] = raw[offset:end].decode("utf-16le")
        offset = end
    root, attributes, entries = unpack_from("<III", raw, offset)
    if (root, attributes) != (0x6C25D918, 0) or not 10000 <= entries <= count:
        raise ValueError("火炬之光 II 译名表根节点格式变化")
    offset += 12
    pairs = []
    for _ in range(entries):
        node, attributes, original, original_type, en, translation, translation_type, zh, children = (
            unpack_from("<9I", raw, offset)
        )
        if (node, attributes, original, original_type, translation, translation_type, children) != (
            0x58E12ECA, 2, 0x1655BA00, 5, 0x58E12ECA, 5, 0
        ):
            raise ValueError("火炬之光 II 译名表条目格式变化")
        pairs.append((strings[en], strings[zh]))
        offset += 36
    return pairs


def torchlight2(source):
    workshop = "https://steamcommunity.com/sharedfiles/filedetails/?id=405160259"
    url = "https://cdn.steamusercontent.com/ugc/496888905826645202/8BE82A132B0767A639FC5CBC8E6DC4855596FC46/"
    archive = binary_source(source, url, "tl2_official.mod", 32_000_000)
    data_offset, manifest_offset = unpack_from("<II", archive, 12)
    offset = data_offset + 8
    # 固定工作坊版本的第 40 个数据块是 Mandarin 的 TRANSLATION.DAT.BINDAT。
    for index in range(41):
        size, compressed = unpack_from("<II", archive, offset)
        if size > 5_000_000 or offset + 8 + compressed > manifest_offset:
            raise ValueError("火炬之光 II 本地化包格式变化")
        offset += 8
        if index == 40:
            pairs = read_torchlight2_pairs(zlib.decompress(archive[offset:offset + compressed]))
        offset += compressed
    if ("Magma Spear", "岩浆长矛") not in pairs:
        raise ValueError("火炬之光 II 简体中文校验失败")
    pack = Dictionary()
    labels, effects = {}, {}
    for i, (en, zh) in enumerate(pairs):
        if re.fullmatch(r"[A-Z][A-Za-z0-9 '&’:+/\-]{3,69}", en) and len(zh) <= 35 and not re.search(r"[\n|。，！；]", zh):
            labels[i] = en, zh
        elif en.count("[VALUE]") == zh.count("[VALUE]") == 1 and len(en) <= 95 and len(zh) <= 65:
            if not re.search(r"[<>{}|\n]", en + zh):
                effects[i] = en.replace("[VALUE]", "#"), zh.replace("[VALUE]", "#")
    for group, rows, minimum in (("skill_item_and_game_labels", labels, 500), ("numeric_effects", effects, 20)):
        pack.add(group, [workshop, url], {i: pair[0] for i, pair in rows.items()}, {i: pair[1] for i, pair in rows.items()},
                 minimum=minimum, effects=group == "numeric_effects")
    return pack


def chronicon(source):
    base = "https://raw.githubusercontent.com/gabriel-dehan/chronicondb-client/main/src/engine/data/1.52.0/sources/locale/"
    pack = Dictionary()
    for category, pattern, minimum in (
        ("skills", r"skill_\d+_name", 650), ("items", r"item_\d+_name", 500),
        ("mastery", r"mastery_\d+_name", 225), ("stats", r"stat_\w+", 40),
        ("enchants", r"enchant_\d+_name", 100), ("buffs", r"buff_\d+_name", 100),
    ):
        urls = [f"{base}{lang}/{category}" for lang in ("EN", "CN")]
        entries = [dict(re.findall(r'^([a-zA-Z0-9_]+)="([^\r\n]*)"\s*$', source.get(url, f"chron_152_{lang}_{category}"), re.M))
                   for lang, url in zip(("EN", "CN"), urls)]
        labels = [{key: value for key, value in data.items() if re.fullmatch(pattern, key) and len(value) <= 90}
                  for data in entries]
        pack.add(category, urls, *labels, minimum=minimum)
        if category in {"enchants", "buffs"}:
            # 保留同一数值占位符的短效果，跳过引用技能、长描述和多变量句子。
            effects = [{key: value.replace("AMOUNT", "#") for key, value in data.items()
                        if key.endswith("_txt") and value.count("AMOUNT") == 1 and len(value) <= 100
                        and not re.search(r"[<>|\d]", value)}
                       for data in entries]
            pack.add(category + "_effects", urls, *effects, minimum=10, effects=True)
    return pack


IMPORTERS = {"diablo2": diablo2, "diablo3": diablo3, "diablo_immortal": diablo_immortal,
             "grim_dawn": grim_dawn, "torchlight_infinite": torchlight_infinite, "hero_siege": hero_siege,
             "slormancer": slormancer, "titan_quest2": titan_quest2, "torchlight2": torchlight2, "chronicon": chronicon}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("games", nargs="*", help=", ".join(IMPORTERS))
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "backend/app/glossaries")
    args = parser.parse_args()
    requested = args.games or list(IMPORTERS)
    if unknown := set(requested) - IMPORTERS.keys():
        parser.error("未知游戏：" + ", ".join(sorted(unknown)))
    source = Sources(args.cache_dir)
    failed = False
    for game in dict.fromkeys(requested):
        try:
            payload = IMPORTERS[game](source).build()
            # 跳过只有大小写或引号写法不同、中文却不一致的条目。
            terms = payload["terms"]
            normalized = {}
            for word, translated in terms.items():
                key = unicodedata.normalize("NFKC", word).casefold().replace("’", "'")
                normalized.setdefault(key, set()).add(translated)
            payload["terms"] = {
                word: translated for word, translated in terms.items()
                if len(normalized[unicodedata.normalize("NFKC", word).casefold().replace("’", "'")]) == 1
            }
            if game == "diablo3":
                payload["locale"] = "zh-TW"
            save_pack(args.output_dir / f"{game}.json", payload)
            print(f"{game}: {len(payload['terms'])} 项术语，{len(payload['effects'])} 条效果", flush=True)
        except (OSError, ValueError, KeyError, AttributeError, IndexError) as error:
            print(f"{game} 刷新失败，保留已有词库：{error}", flush=True)
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
