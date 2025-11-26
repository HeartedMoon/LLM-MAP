# core/lit_search.py
from typing import List, Dict, Tuple
from .lit_api import search_online_sources
from .query_builder import extract_keywords, build_boolean_query

# ========== 新增：TF-IDF 排序 ==========
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

def _norm_text(s: str) -> str:
    if not s: return ""
    # 统一空白 + 去掉过多标点
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _pack_doc(s: Dict) -> str:
    """把 title + abstract + journal 拼成一个 doc，最大化重合机会"""
    return _norm_text(" ".join([
        s.get("title","") or "",
        s.get("abstract","") or "",
        s.get("journal","") or ""
    ]))

def rank_sources_by_relevance(query_text: str, sources: List[Dict], top_k: int = 8) -> Tuple[List[Dict], List[Dict]]:
    """
    用 TF-IDF 排序（字符 n-gram，兼容中英混合）。
    回退策略：
      - 若字符 n-gram 得分全 0，再试词粒度（带中文 token_pattern）
      - 若仍 0，则按是否含摘要/标题长度做一个弱排序
    """
    if not sources:
        return [], []

    docs = [_pack_doc(s) for s in sources]
    query = _norm_text(query_text or "")

    # ---------- 1) 首选：字符 n-gram（中英都稳） ----------
    try:
        vect = TfidfVectorizer(
            analyzer="char_wb",        # 词内字符，边界感知
            ngram_range=(3, 5),        # 3~5 字符片段
            min_df=1,
            max_features=20000,
            lowercase=True
        )
        X = vect.fit_transform(docs + [query])
        D, q = X[:-1], X[-1]
        scores = cosine_similarity(D, q).ravel()
    except Exception:
        scores = np.zeros(len(docs), dtype=float)

    def _rank_from_scores(scores_arr: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        order = np.argsort(-scores_arr)
        top_idx = order[:max(1, top_k)]
        ranked = [sources[int(i)] for i in top_idx]
        debug = [{
            "idx": int(i),
            "score": float(scores_arr[int(i)]),
            "title": sources[int(i)].get("title",""),
            "doi": sources[int(i)].get("doi",""),
            "year": sources[int(i)].get("year",""),
            "journal": sources[int(i)].get("journal","")
        } for i in top_idx]
        return ranked, debug

    ranked, debug = _rank_from_scores(scores)

    # 如果全部为 0，说明字符重合很弱 → 回退到“词粒度 + 中文 token_pattern”
    if not np.any(scores > 0):
        try:
            vect2 = TfidfVectorizer(
                analyzer="word",
                token_pattern=r"(?u)\b[\w\u4e00-\u9fff]{2,}\b",  # 英文词 + 连续中文 >=2
                ngram_range=(1, 2),
                min_df=1,
                max_features=20000,
                lowercase=True
            )
            X2 = vect2.fit_transform(docs + [query])
            D2, q2 = X2[:-1], X2[-1]
            scores2 = cosine_similarity(D2, q2).ravel()
            if np.any(scores2 > 0):
                ranked, debug = _rank_from_scores(scores2)
        except Exception:
            pass

    # 若仍然全 0，做一个弱排序：优先有摘要的、标题更长的
    if all(abs(d["score"]) < 1e-12 for d in debug):
        def _weak_key(i: int) -> tuple:
            s = sources[i]
            has_abs = 1 if (s.get("abstract") and len(s["abstract"]) > 30) else 0
            title_len = len(s.get("title") or "")
            year = int(s.get("year") or 0)
            return (has_abs, year, title_len)
        idxs = sorted(range(len(sources)), key=_weak_key, reverse=True)[:max(1, top_k)]
        ranked = [sources[i] for i in idxs]
        debug = [{
            "idx": int(i),
            "score": 0.0,
            "title": sources[i].get("title",""),
            "doi": sources[i].get("doi",""),
            "year": sources[i].get("year",""),
            "journal": sources[i].get("journal","")
        } for i in idxs]

    return ranked, debug
# ========== /新增 ==========


def unified_search(query:str, year_from:int=2010, year_to:int=2100, limit:int=8)->List[Dict]:
    """
    关键词抽取 → 布尔查询（先 AND，少则 OR 回退） → 在线检索
    """
    must, anyt, _ = extract_keywords(query)
    strict_q = build_boolean_query(must, anyt)
    res = search_online_sources(query=strict_q or query, year_from=year_from, year_to=year_to, limit=limit)
    if len(res) >= max(1, limit // 2) or not anyt:
        return res
    # 放宽：仅 OR
    loose_q = "(" + " OR ".join(anyt) + ")"
    res2 = search_online_sources(query=loose_q, year_from=year_from, year_to=year_to, limit=limit)
    # 合并去重（按 DOI/标题）
    seen_doi = set((x.get("doi") or "").lower() for x in res if x.get("doi"))
    seen_title = set((x.get("title") or "").strip().lower() for x in res if x.get("title"))
    for x in res2:
        d = (x.get("doi") or "").lower()
        t = (x.get("title") or "").strip().lower()
        if d and d in seen_doi: continue
        if t and t in seen_title: continue
        res.append(x)
        if len(res) >= limit: break
    return res


def unified_search_with_boolean(boolean_query: str, year_from:int=2010, year_to:int=2100, limit:int=8) -> List[Dict]:
    """
    直接使用布尔查询字符串调用在线检索（Crossref/S2 封装）
    """
    q = (boolean_query or "").strip()
    if not q:
        return []
    return search_online_sources(query=q, year_from=year_from, year_to=year_to, limit=limit)



def make_online_snippets(sources:List[Dict], max_each:int=1200, max_total:int=40000)->str:
    """
    把文献（已排序/截断）压成 RAG 片段
    """
    out=[]; total=0
    for i, s in enumerate(sources, 1):
        title = (s.get("title","") or "").strip()
        jour  = (s.get("journal","") or "").strip()
        year  = s.get("year","") or ""
        doi   = s.get("doi","") or ""
        url   = f"https://doi.org/{doi}" if doi else (s.get("url","") or "")
        abstr = (s.get("abstract","") or "").strip()
        header = f"[{i}] {title} — {jour} ({year})"
        if url: header += f"\nURL: {url}"
        if abstr:
            seg = header + "\nABSTRACT: " + (abstr[:max_each] + (" …" if len(abstr) > max_each else ""))
        else:
            seg = header
        if total + len(seg) > max_total: break
        out.append(seg); total += len(seg)
    return "\n\n".join(out)
