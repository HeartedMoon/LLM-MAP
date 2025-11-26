from __future__ import annotations
from typing import List, Dict, Any
import re, collections

def refine_query_prf(initial_query: str, top_docs: List[Dict[str, Any]], k: int = 5, top_terms: int = 6) -> str:
    text = " ".join([(d.get("title","")+" "+d.get("abstract","")+" "+d.get("snippet","")+" "+ " ".join(d.get("keywords") or [])) for d in top_docs[:k]])
    toks = re.findall(r"[A-Za-z][A-Za-z0-9+\-]{2,}", text.lower())
    stop = set(["the","and","with","for","from","this","that","into","using","based","study","results","data","high","low","rate","type","over","more"])
    cnt = collections.Counter([t for t in toks if t not in stop])
    top = [w for w,_ in cnt.most_common(top_terms)]
    if not top:
        return initial_query
    extra = " OR ".join(sorted(set(top)))
    return f"({initial_query}) AND ({extra})"

def score_snippet_goal_aware(snippet: str, kpi: Dict[str, Any]) -> float:
    s = snippet.lower()
    score = 0.0
    for k, v in (kpi or {}).items():
        key = re.sub(r"[_\-]+", " ", str(k).lower())
        if key in s:
            score += 0.6
        if re.search(r"\b\d+(\.\d+)?\s*%?\b", s):
            score += 0.2
    if re.search(r"(extraction|stripping|precipitation|crystallization|leaching|raffinate|kerosene|p204|n235|fe\(?iii\)?)", s):
        score += 0.3
    return round(min(score, 1.5), 3)
