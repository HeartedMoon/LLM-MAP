# core/lit_api.py
import re, time, requests
from typing import List, Dict, Optional

CR_BASE = "https://api.crossref.org/works"
S2_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"

def _strip_html(x: str) -> str:
    if not x: return ""
    x = re.sub(r"<\/?jats:[^>]*>", "", x)          # 去 Crossref JATS
    x = re.sub(r"<[^>]+>", "", x)                  # 其它标签
    return re.sub(r"\s+", " ", x).strip()

def _authors_from_crossref(item) -> str:
    au = item.get("author") or []
    names=[]
    for a in au:
        g=a.get("given",""); f=a.get("family","")
        nm = " ".join([g,f]).strip() or a.get("name","")
        if nm: names.append(nm)
    return "; ".join(names)

def _authors_from_s2(item) -> str:
    au = item.get("authors") or []
    names = [a.get("name","") for a in au if a.get("name")]
    return "; ".join(names)

def query_crossref(query:str, limit:int=8, year_from:Optional[int]=None, year_to:Optional[int]=None)->List[Dict]:
    params = {"query": query, "rows": limit, "select":"DOI,title,author,container-title,issued,abstract,type"}
    filt=[]
    if year_from: filt.append(f"from-pub-date:{year_from}-01-01")
    if year_to:   filt.append(f"until-pub-date:{year_to}-12-31")
    filt.append("type:journal-article")
    params["filter"]=",".join(filt)
    try:
        r = requests.get(CR_BASE, params=params, timeout=15)
        r.raise_for_status()
        items = (r.json().get("message") or {}).get("items") or []
    except Exception:
        return []
    out=[]
    for it in items:
        title = " ".join(it.get("title") or []) or ""
        jour  = " ".join(it.get("container-title") or []) or ""
        year  = ""
        if "issued" in it:
            d = it["issued"].get("date-parts")
            if isinstance(d,list) and d and isinstance(d[0],list) and d[0]:
                year = str(d[0][0])
        doi   = it.get("DOI","")
        url   = f"http://doi.org/{doi}" if doi else ""
        abstr = _strip_html(it.get("abstract","") or "")
        authors = _authors_from_crossref(it)
        out.append({
            "title": title, "journal": jour, "year": year, "authors": authors,
            "doi": doi, "url": url, "abstract": abstr, "source": "crossref"
        })
    return out

def query_semanticscholar(query:str, limit:int=8, year_from:Optional[int]=None, year_to:Optional[int]=None)->List[Dict]:
    params={"query":query,"limit":limit,"fields":"title,year,venue,authors,abstract,doi,url"}
    try:
        r = requests.get(S2_BASE, params=params, timeout=15)
        r.raise_for_status()
        items = (r.json() or {}).get("data") or []
    except Exception:
        return []
    out=[]
    for it in items:
        title = it.get("title","")
        jour  = it.get("venue","")
        year  = str(it.get("year","") or "")
        doi   = it.get("doi","") or ""
        url   = f"http://doi.org/{doi}" if doi else (it.get("url","") or "")
        abstr = _strip_html(it.get("abstract","") or "")
        authors = _authors_from_s2(it)
        out.append({
            "title": title, "journal": jour, "year": year, "authors": authors,
            "doi": doi, "url": url, "abstract": abstr, "source": "semanticscholar"
        })
    return out

def search_online_sources(query:str, limit:int=20, year_from:Optional[int]=2010, year_to:Optional[int]=None)->List[Dict]:
    """
    先 Crossref，再用 S2 填补缺失摘要/DOI；按 DOI 或 标题去重
    """
    cr = query_crossref(query, limit=limit, year_from=year_from, year_to=year_to)
    s2 = query_semanticscholar(query, limit=limit, year_from=year_from, year_to=year_to)
    # 索引现有
    by_doi = { (i["doi"] or "").lower(): i for i in cr if i.get("doi") }
    by_title = { (i["title"] or "").strip().lower(): i for i in cr if i.get("title") }
    # 合并
    for x in s2:
        key_d = (x.get("doi") or "").lower()
        key_t = (x.get("title") or "").strip().lower()
        target = None
        if key_d and key_d in by_doi: target = by_doi[key_d]
        elif key_t and key_t in by_title: target = by_title[key_t]
        if target:
            # 填补缺字段
            for k in ["abstract","journal","year","authors","url","doi"]:
                if not target.get(k) and x.get(k): target[k]=x[k]
        else:
            cr.append(x)
            if key_d: by_doi[key_d]=x
            if key_t: by_title[key_t]=x
    return cr[:limit]
