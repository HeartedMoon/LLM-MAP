# core/query_refiner.py
import json, re
from typing import Dict, List, Tuple
from .llm_client import chat_complete
from .query_builder import extract_keywords, build_boolean_query

# ROLE = (
#     "You are a query refining assistant for academic literature search. "
#     "You MUST respond with a single, valid JSON object only. No prose, no code fences."
# )

# USER_TMPL = """Convert the following question to English (if the input is not in English, the language conversion is ignored for English input), extract the keyword set for academic literature search, and return only JSON (without any extra text and code block fences).

# Return the JSON fields (strictly use these keys):
# - need_online_search: bool  # Whether online search is required (true/false) according to the user's intent
# - must_terms: list[str]     # Key mandatory terms (<=2)
# - any_terms:  list[str]     # Related expansion terms (<=4) 
# - boolean:    string        # AND/OR boolean query expression (English AND/OR)
# - english_query: string     # English query for relevance calculation (terms/phrases only, space separated, no AND/OR operators)

# question:
# {q}
# """

# =========================
# 1) ROLE：更明确的行为规范
# =========================
ROLE = (
    "You are a query refining assistant for academic literature search. "
    "You MUST respond with a single, valid JSON object only (no prose, no code fences). "
    "Your job is to: "
    "(1) decide if online search is needed; "
    "(2) extract compact MUST terms and ANY terms; "
    "(3) produce a robust Boolean query that uses **base forms (no wildcard '*')** to cover inflections; enumerate key variants only when necessary;"
    "(4) output an english_query for semantic ranking (no AND/OR). "
    "Rules:\n"
    "- **Do NOT use '*'**. Prefer base forms/lemmas (e.g., leach, crystal, precipitate, purify) so the engine can match inflections (leached, leaching, purification, etc.). If a variant is commonly distinct (e.g., selective vs selectivity), you may enumerate (selective OR selectivity).\n"
    "- Map chemical names and formulas into an OR group when appropriate (e.g., (\"lithium carbonate\" OR Li2CO3)).\n"
    "- For multi-word phrases, use quotes in Boolean (e.g., \"lithium carbonate\").\n"
    "- Keep must_terms ≤ 2 and any_terms ≤ 4; each can be a base form or a short quoted phrase (no '*').\n"
    "- The Boolean must be concise and executable (use AND/OR and parentheses; avoid NOT unless necessary). Prefer the combined pattern: (must) OR (must AND (any)).\n"
    "- english_query is a space-separated bag of base forms/phrases (no AND/OR), concise.\n"
    "- If the user intent is a follow-up or conceptual and doesn’t require new retrieval, set need_online_search=false."
)

# =========================
# 2) USER_TMPL：强约束输出字段
# =========================
USER_TMPL = """Convert the question to English (if already English, keep it). 
Extract robust search terms and produce a Boolean query.

Return ONLY JSON (no extra text, no code fences) with exactly these keys:
- need_online_search: bool
- must_terms: list[str]   # ≤2; base forms or short phrases; **no '*'**
- any_terms:  list[str]   # ≤4; base forms or common variants; **no '*'**
- boolean:    string      # Use AND/OR and parentheses; prefer combined pattern: (must) OR (must AND (any)); **no '*'**
- english_query: string   # Space-separated base forms/phrases (no AND/OR), concise; **no '*'**

Guidelines:
- Prefer base forms (e.g., leach) so engines match inflections (leached, leaching, leachate).
- Use quotes only for multi-word phrases.
- Keep the Boolean minimal but expressive.

question:
{q}
"""

# ============================================
# 3) FEW_SHOTS：教会“词根通配 + OR 组合”的风格
# ============================================
FEW_SHOTS = [
    {
        "user": USER_TMPL.format(q="在LFP回收中如何提升碳酸锂结晶的纯度与收率？有没有最新的工艺？"),
        "assistant": (
            '{'
            '"need_online_search": true, '
            '"must_terms": ["lithium carbonate", "Li2CO3", "crystal"], '
            '"any_terms":  ["purif", "impurit", "yield", "recrystalliz", "selectiv", "wash"], '
            '"boolean": "( ("lithium carbonate" OR Li2CO3) AND crystal AND (purif OR impurit OR yield OR recrystalliz OR selectiv OR wash) )", '
            '"english_query": "lithium carbonate Li2CO3 crystal purif impurit yield recrystalliz selectiv wash"'
            '}'
        )
    },
    {
        "user": USER_TMPL.format(q="Effect of washing on removal of Mg/Ca impurities during Li2CO3 precipitation"),
        "assistant": (
            '{'
            '"need_online_search": true, '
            '"must_terms": ["Li2CO3", "precipitat"], '
            '"any_terms":  ["lithium carbonate", "wash", "impurit", "magnesium", "calcium", "remov"], '
            '"boolean": "( (Li2CO3 OR "lithium carbonate") AND precipitat AND (wash OR impurit OR magnesium OR calcium OR remov) )", '
            '"english_query": "Li2CO3 lithium carbonate precipitat wash impurit magnesium calcium remov"'
            '}'
        )
    },
    {
        "user": USER_TMPL.format(q="We already tested a washing sequence; just compare against known recrystallization routes."),
        "assistant": (
            '{'
            '"need_online_search": false, '
            '"must_terms": ["recrystalliz"], '
            '"any_terms":  ["purif", "impurit", "route", "compar"], '
            '"boolean": "( recrystalliz AND (purif OR impurit OR route OR compar) )", '
            '"english_query": "recrystalliz purif impurit route compar"'
            '}'
        )
    }
]

def _safe_json(text: str) -> Dict:
    """
    鲁棒 JSON 提取：
    - 去掉 ```json ...``` 或 ```...``` 围栏
    - 支持 <JSON>...</JSON>
    - 扫描第一个平衡的 {...} 块
    """
    import json, re
    if not text: return {}
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence: s = fence.group(1).strip()
    tag = re.search(r"<JSON>\s*([\s\S]*?)\s*</JSON>", s, re.IGNORECASE)
    if tag: s = tag.group(1).strip()

    try:
        j = json.loads(s)
        if isinstance(j, dict): return j
    except Exception:
        pass

    depth = 0; start = -1
    for i, ch in enumerate(s):
        if ch == '{':
            if depth == 0: start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    cand = s[start:i+1]
                    try:
                        j = json.loads(cand)
                        if isinstance(j, dict): return j
                    except Exception:
                        pass
    return {}


def merge_terms(llm_must: List[str], llm_any: List[str],
                rule_must: List[str], rule_any: List[str],
                max_must=8, max_any=12) -> Tuple[List[str], List[str]]:
    """
    合并 + 去重 + 去停用（规则法内部已做停用词过滤）
    策略：LLM > 规则；化学式/引号短语优先进 must。
    """
    seen = set()
    out_must = []
    out_any  = []

    def push(lst, tgt):
        for w in lst:
            w = (w or "").strip()
            if not w: continue
            lw = w.lower()
            if lw in seen: continue
            seen.add(lw)
            tgt.append(w)

    # LLM 优先
    push(llm_must or [], out_must)
    push(llm_any or [], out_any)

    # 规则兜底
    push(rule_must or [], out_must)
    push(rule_any or [], out_any)

    # 裁剪
    return out_must[:max_must], out_any[:max_any]

def _strip_star(s: str) -> str:
    # 去掉词尾或词中的星号，保留词根
    return re.sub(r"\*", "", s or "")

def sanitize_terms(terms):
    out = []
    for w in terms or []:
        w2 = _strip_star(w.strip())
        if w2:
            out.append(w2)
    return out


def refine_query_with_llm(question: str, model: str, temperature: float=0.2) -> Tuple[List[str], List[str], str, Dict]:
    q = (question or "").strip()

    # 规则法兜底（先算好，后面要合并）
    rule_must, rule_any, rule_dbg = extract_keywords(q)

    if not q:
        boolean_q = build_boolean_query(rule_must, rule_any)
        return rule_must, rule_any, boolean_q, {
            "llm_raw": "", "llm_json": {}, "rule_terms": rule_dbg,
            "merged": {"must": rule_must, "any": rule_any, "boolean": boolean_q},
            "reason": "empty_question"
        }

    # 1) LLM 提取（尽量强约束 JSON）
    llm_raw = ""; llm_js = {}
    try:
        msgs = [{"role":"system","content": ROLE}]
        for fs in FEW_SHOTS:
            msgs.append({"role":"user","content": fs["user"]})
            msgs.append({"role":"assistant","content": fs["assistant"]})
        msgs.append({"role":"user","content": USER_TMPL.format(q=q)})
        llm_raw = chat_complete(
            model=model,
            temperature=temperature,
            messages=msgs
            # max_tokens=1200
        )
        llm_js = _safe_json(llm_raw) or {}
    except Exception:
        llm_js = {}

    llm_need_search = bool(llm_js.get("need_online_search"))
    llm_must = llm_js.get("must_terms") or []
    llm_any  = llm_js.get("any_terms")  or []
    llm_must = sanitize_terms(llm_must)
    llm_any = sanitize_terms(llm_any)
    # llm_bool = (llm_js.get("boolean")   or "").strip()
    llm_bool = build_boolean_query(llm_must, llm_any, mode="combined")
    llm_eq   = (llm_js.get("english_query") or "").strip()

    if llm_need_search:
        def _is_term_ok(w: str) -> bool:
            w = (w or "").strip()
            return bool(w) and len(w) <= 64 and not w.endswith(("。","!","?","."))

        llm_terms_ok = any(_is_term_ok(x) for x in (llm_must + llm_any))

        # 2) 合并（LLM 优先，规则补强）
        def _merge_terms(a_must, a_any, b_must, b_any, max_must=2, max_any=4):
            seen=set(); out_must=[]; out_any=[]
            def push(src, tgt):
                for w in src or []:
                    w = (w or "").strip()
                    if not w: continue
                    lw = w.lower()
                    if lw in seen: continue
                    seen.add(lw); tgt.append(w)
            push(a_must, out_must); push(a_any, out_any)
            push(b_must, out_must); push(b_any, out_any)
            return out_must[:max_must], out_any[:max_any]

        if llm_terms_ok:
            must, anyt = _merge_terms(llm_must, llm_any, rule_must, rule_any)
            # 生成 rank_query：优先 LLM 的 english_query；否则用 must+any 拼接；再兜底到原始 q
            rank_query = llm_eq or " ".join(must + anyt) or q
            def _boolean_valid(s: str) -> bool:
                s = (s or "").strip()
                if len(s) < 3: return False
                return (" AND " in s) or (" OR " in s) or (" " in s)
            boolean_q = llm_bool if _boolean_valid(llm_bool) else build_boolean_query(must, anyt)
            reason = "llm_first_then_rule_merged"
        else:
            must, anyt = rule_must, rule_any
            boolean_q = build_boolean_query(must, anyt)
            reason = "fallback_to_rules"

        debug = {
            "llm_raw": llm_raw,
            "llm_json": llm_js,
            "rule_terms": rule_dbg,
            "merged": {"must": must, "any": anyt, "boolean": boolean_q},
            "rank_query": rank_query,
            "reason": reason
        }
        return must, anyt, boolean_q, debug
    else:
        # 不需要在线搜，直接返回空
        # boolean_q = build_boolean_query([], [])
        return [], [], "", {
            "llm_raw": llm_raw,
            "llm_json": llm_js,
            "rule_terms": rule_dbg,
            "merged": {"must": [], "any": [], "boolean": ""},
            "reason": "llm_says_no_search"
        }
