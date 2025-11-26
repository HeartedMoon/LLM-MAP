# core/query_builder.py
import os, re, yaml
from typing import List, Tuple, Dict

# ---------- 文本归一化（含全角->半角、中英标点空白化） ----------
_ZH_WS = r"\u3000"
_FULL2HALF = str.maketrans({
    '，':',', '。':'.', '；':';', '：':':', '（':'(', '）':')', '【':'[', '】':']', '！':'!', '？':'?',
    '“':'"', '”':'"', '、':',', '—':'-', '－':'-', '～':'~', '·':'-', '　':' '  # 全角空格
})
def normalize_text(s: str) -> str:
    if not s: return ""
    s = s.translate(_FULL2HALF)
    s = re.sub(rf"[{_ZH_WS}\t\r]+", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

# ---------- 默认内置（若无 YAML 时使用） ----------
_DEFAULT_STOP_EN = set("""
the a an of in on to for with into from by at as about over under and or if then else
is are was were be being been can could should would may might will shall do does did
this that these those it its their there here such using use via method methods based
study studies approach approaches result results effect effects analysis evaluation
data figure table supplementary note notes discussion conclusion
""".split())

_DEFAULT_STOP_ZH = set("""
的 了 和 与 及 或 而 但 被 在 从 对 于 将 为 因 因为 因此 所以 由于 根据 其中 同时 以及 等 等等
这些 那些 该 此 本 我们 他们 其 主要 一种 一类 一般 该文 该方法 该工艺
可以 能够 需要 必须 进行 采用 使用 利用 给出 给定 得到 实现 提出
方法 研究 实验 数据 结果 分析 影响 优化 讨论 综述 概述
通过 对比 比较 测试 验证 建模 模型 参数 条件
例如 比如 如 如下 上述 下述 分别
""".split())

_DEFAULT_DOMAIN_MUST = {
    "LiFePO4","LFP","FeCl3","FeCl2","Li2CO3","Li2CO₃","LiOH","LiCl","Li2SO4","HCl","H2SO4","HNO3",
    "Na2CO3","NaHCO3","NH4HCO3","NH3","NaOH","K2CO3","KOH","EDTA","citric acid","oxalic acid",
    "ferric chloride","ferrous chloride","iron phosphate",
    "hydrometallurgy","leaching","selective leaching","oxidative leaching","precipitation",
    "carbonation","crystallization","washing","solid-liquid separation",
    "Li2CO3 purification","recrystallization","re-precipitation",
    "循环浸出","盐浸","氧化","还原","固液分离","结晶","沉淀","洗涤","再沉淀","重结晶","冷却结晶","蒸发结晶",
    "磷酸铁锂", "磷酸铁", "碳酸锂","氢氧化锂","碳酸钠","碳酸氢钠","碳酸氢铵","氨","草酸","柠檬酸","络合","萃取",
    "pH","温度","浓度","料液比","停留时间","搅拌","过饱和","溶度积","Ksp",
    "impurity","impurities","purity","selectivity","yield","recovery",
    "Fe","P","Al","Cu","Ni","Mn","Co","Mg","Ca","Na","K","S","Cl","F","Si",
    "ion exchange","membrane","nanofiltration","ultrafiltration","reverse osmosis",
    "kinetics","thermodynamics","diffusion","mass transfer","shrinking core model","activation energy",
    "动力学","热力学","传质","扩散","收缩核模型","表观活化能",
    "reactor","crystallizer","centrifuge","filter","dryer","rotary evaporator",
    "反应釜","结晶器","离心机","过滤机","干燥器","旋蒸",
    "cost","energy consumption","environmental","wastewater","waste gas","solid waste","E-factor",
    "成本","能耗","环保","废水","废气","固废","碳足迹"
}

_DEFAULT_DOMAIN_ANY = {
    "battery","batteries","cathode","recycling","regeneration","process","process optimization",
    "kinetics analysis","thermodynamic analysis","phase","solubility","soda ash","ammonia","seed",
    "filtration","rinsing","mother liquor","liquor recycle","bleed","impurity purge",
    "正极","回收","再生","工艺","优化","工艺集成","多目标优化","多级逆流","洗涤比","夹带","滞留",
    "复杂体系","相平衡","活度","溶解度","晶型","晶核","籽晶","诱导期",
    "过氧化氢","高锰酸钾","次氯酸钠","臭氧","氯气","氧气","二氧化碳","空气"
}

def _load_yaml_terms() -> Tuple[set, set, set, set]:
    """
    从 core/keywords.yaml 加载；若不存在则返回默认集
    每行允许用 ; 分隔多个词条
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "keywords.yaml")
    if not os.path.exists(path):
        return _DEFAULT_STOP_EN, _DEFAULT_STOP_ZH, _DEFAULT_DOMAIN_MUST, _DEFAULT_DOMAIN_ANY

    def _splitline(line: str) -> List[str]:
        # 支持 “a; b; c” 多词一行
        parts = [p.strip() for p in str(line).replace("；",";").split(";")]
        return [p for p in parts if p]

    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}
    se = set(); sz = set(); dm = set(); da = set()
    for line in (y.get("stopwords_en") or []):
        se.update(_splitline(line))
    for line in (y.get("stopwords_zh") or []):
        sz.update(_splitline(line))
    for line in (y.get("domain_must") or []):
        dm.update(_splitline(line))
    for line in (y.get("domain_any") or []):
        da.update(_splitline(line))

    # 防止空载导致全清空
    se = se or _DEFAULT_STOP_EN
    sz = sz or _DEFAULT_STOP_ZH
    dm = dm or _DEFAULT_DOMAIN_MUST
    da = da or _DEFAULT_DOMAIN_ANY
    return se, sz, dm, da

STOP_EN, STOP_ZH, DOMAIN_MUST, DOMAIN_ANY = _load_yaml_terms()

# ---------- Token 规则：英数/化学式/中文逐字 ----------
CHEM_PATTERN = re.compile(r"(?:(?:[A-Z][a-z]?)(?:\d+|₀|₁|₂|₃|₄|₅|₆|₇|₈|₉)?){1,6}")
PHRASE_IN_QUOTES = re.compile(r'["“](.+?)["”]')
TOKEN = re.compile(r"[A-Za-z0-9₀-₉\-\+\./₊₋₍₎]+|[\u4e00-\u9fa5]{1,}")

def _normalize_token(t: str) -> str:
    t = normalize_text(t)
    return t.strip(",.;:，；。()（）").lower()

def extract_keywords(text: str) -> Tuple[List[str], List[str], Dict]:
    """
    返回 (must_terms, any_terms, debug)
    - 支持 YAML 可配置词表
    - 更强的中文停用词过滤
    - 全角/半角、标点归一化
    """
    if not text:
        return [], [], {"raw":"","phrases":[],"chems":[],"must":[],"any":[]}

    raw = normalize_text(text)

    phrases = [p.strip() for p in PHRASE_IN_QUOTES.findall(raw) if len(p.strip()) >= 2]
    toks = [m.group(0) for m in TOKEN.finditer(raw)]
    toks_norm = [_normalize_token(t) for t in toks if _normalize_token(t)]

    chems = set()
    for t in toks:
        if CHEM_PATTERN.fullmatch(t):
            chems.add(t)

    dm = {x.lower() for x in DOMAIN_MUST}
    da = {x.lower() for x in DOMAIN_ANY}

    must, anyt = set(), set()
    for t in toks:
        if (t in DOMAIN_MUST) or (t.lower() in dm):
            must.add(t)
    for t in toks:
        tl = t.lower()
        if (t in DOMAIN_ANY) or (tl in da):
            if t not in must:
                anyt.add(t)

    # 英文/中文停用词
    for t in toks_norm:
        if len(t) <= 1: 
            continue
        if (t in STOP_EN) or (t in STOP_ZH):
            continue
        if t not in {x.lower() for x in must} and t not in {x.lower() for x in anyt}:
            anyt.add(t)

    # 引号短语 + 化学式 强制进入 must
    must |= set(phrases) | chems

    # 去重保序 & 裁剪
    def _dedup(seq):
        seen=set(); out=[]
        for x in seq:
            xl = x.lower()
            if xl in seen: continue
            seen.add(xl); out.append(x)
        return out
    must_list = _dedup(list(must))[:8]
    any_list  = _dedup(list(anyt))[:12]

    return must_list, any_list, {"raw":raw,"phrases":phrases,"chems":list(chems),"must":must_list,"any":any_list}

# def build_boolean_query(must: List[str], anyt: List[str]) -> str:
#     def qword(w: str) -> str:
#         return f'"{w}"' if (" " in w or "-" in w or "/" in w) else w
#     parts=[]
#     if must:
#         parts.append(" AND ".join(qword(w) for w in must))
#     if anyt:
#         parts.append("(" + " OR ".join(qword(w) for w in anyt) + ")")
#     return " AND ".join(parts) if parts else ""


def build_boolean_query(must: List[str], anyt: List[str], mode: str = "combined") -> str:
    """
    mode 说明：
      - "combined": (must) OR (must AND (any))  —— 推荐：一条查询覆盖“严格+放宽”
      - "strict":   must AND (any)               —— 仅在你明确需要强约束 any 时使用
      - "must":     must                         —— 不需要 boolean 时只用 must
      - "any":      (any)                        —— 仅用 any（极少需要）
    """

    def qword(w: str) -> str:
        # 含空格/连字符/斜杠的词加引号，其他原样返回
        return f'"{w}"' if (" " in w or "-" in w or "/" in w) else w

    must_terms = [qword(w) for w in must if w and w.strip()]
    any_terms  = [qword(w) for w in anyt if w and w.strip()]

    must_expr = " AND ".join(must_terms) if must_terms else ""
    any_expr  = "(" + " OR ".join(any_terms) + ")" if any_terms else ""

    if mode == "strict":
        # must AND (any)；若 any 为空，则退化为 must
        return f"{must_expr} AND {any_expr}" if must_expr and any_expr else (must_expr or any_expr)

    if mode == "must":
        return must_expr

    if mode == "any":
        return any_expr

    # 默认 "combined"： (must) OR (must AND (any))
    if must_expr and any_expr:
        return f"({must_expr}) OR ({must_expr} AND {any_expr})"
    elif must_expr:
        return must_expr
    elif any_expr:
        return any_expr
    else:
        return ""
