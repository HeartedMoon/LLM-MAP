
from __future__ import annotations
from typing import List, Dict, Any, Tuple
from pathlib import Path
import re, json, os
from .orchestrator import run_with_role
# 依赖 session_gc 来获取本会话的 variables.json 路径
from core.session_gc import variables_path

# 统一路径：<repo_root>/project_state/variables.json
# VAR_FILE = Path(__file__).resolve().parents[1] / "project_state" / "variables.json"

PROMPT_ROLE = """You are a meticulous scientific variable curator.
Given the ChemProcess Modeler reply, extract or propose variables needed for equations and balances.
Return a JSON with key 'variables': list of {name, symbol, unit, description, default_value}.
-name: the name of the variable
-symbol: the symbol used in equations
-unit: the unit of measurement
-description: a brief description of the variable's meaning
-default_value: the default value of the variable, if applicable, otherwise null
Respond only with JSON.
"""

PROMPT_USER = """ChemProcess Modeler reply:
----------------
{reply}
----------------
If reply defines or needs variables, extract or propose them.
Output JSON format:
{{
  "variables":[{{"name":"string","symbol":"string","unit":"string","description":"string","default_value":null}}]
}}
"""

def _best_effort_parse_json(text: str) -> dict:
    import re, json
    if not text: return {}
    for m in re.findall(r'\{[\s\S]*?\}', text):
        try:
            return json.loads(m)
        except Exception:
            continue
    return {}

# 若模型未给布尔位，做轻量兜底（关键词/正则）
def _maybe_changed_by_text(txt: str) -> bool:
    if not isinstance(txt, str) or not txt.strip():
        return False
    txt_l = txt.lower()
    # 关键词：你可按需扩展
    keywords = [
        "define variable", "new variable", "update variable", "modify variable",
        "变量", "新变量", "更新变量", "变量定义", "符号", "symbol", "notation"
    ]
    if any(k in txt_l for k in keywords):
        return True
    # 一些简单的符号模式，如 X := ... 或 X = f(...)
    import re
    if re.search(r"\b[A-Za-z][A-Za-z0-9_]*\s*[:=]\s*", txt):
        return True
    return False

def extract_variables(modeler_reply: str, model: str, temperature: float=0.2) -> List[Dict[str, Any]]:
    res = run_with_role(
        # agent_name="Variable Definition & Editing",
        role_text=PROMPT_ROLE,
        model=model, temperature=temperature,
        task_spec=PROMPT_USER.format(reply=modeler_reply),
        examples="", cot="", question="",
        output_format='{"variables":[{"name":"string","symbol":"string","unit":"string","description":"string","default_value":null}]}',
        json_schema='{"variables":[{"name":"string","symbol":"string","unit":"string","description":"string","default_value":null}]}',
        # rag_enabled=False, 
        rag_snippets="",
        sources_list=None, citations_parentheses=False
    )
    obj = res.get("conclusion") or _best_effort_parse_json(res.get("reply",""))
    items = (obj or {}).get("variables") or []
    norm = []
    for it in items:
        if not isinstance(it, dict): continue
        name = str(it.get("name","")).strip()
        # symbol = re.sub(r"[^A-Za-z0-9_]", "_", str(it.get("symbol","")).strip())[:32]
        symbol = str(it.get("symbol","")).strip()
        if not name or not symbol: 
            continue
        norm.append({
            "name": name,
            "symbol": symbol,
            "unit": str(it.get("unit","")).strip(),
            "description": str(it.get("description","")).strip(),
            # "formula": str(it.get("formula","")).strip(),
            "default_value": it.get("default_value", None)
        })
    return norm

# def load_variables() -> List[Dict[str, Any]]:
#     try:
#         if VAR_FILE.exists():
#             return json.loads(VAR_FILE.read_text(encoding="utf-8"))
#     except Exception:
#         pass
#     return []

def variables_path(st_session_state=None) -> Path:
    from core.session_gc import session_dir
    return session_dir(st_session_state) / "variables.json"

def load_variables(st_session_state=None) -> List[Dict]:
    try:
        p = variables_path(st_session_state)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

# def save_variables(vars_list: List[Dict[str, Any]]) -> None:
#     os.makedirs(os.path.dirname(VAR_FILE), exist_ok=True)
#     payload = {"variables": vars_list}
#     with open(VAR_FILE, "w", encoding="utf-8") as f:
#         json.dump(payload, f, ensure_ascii=False, indent=2)

# def save_variables(rows):
#     VAR_FILE.parent.mkdir(parents=True, exist_ok=True)
#     VAR_FILE.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

def save_variables(rows: List[Dict], st_session_state=None) -> None:
    p = variables_path(st_session_state)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

# def merge_variables(existing: List[Dict[str, Any]], new_vars: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
#     idx = {v.get("symbol"): i for i, v in enumerate(existing) if isinstance(v, dict) and v.get("symbol")}
#     changed = []
#     for nv in new_vars:
#         sym = nv.get("symbol")
#         if not sym: 
#             continue
#         if sym in idx:
#             i = idx[sym]
#             base = existing[i]
#             upd = dict(base)
#             for k in ["name","unit","description","default_value"]:
#                 if nv.get(k) and (not base.get(k) or len(str(nv.get(k))) > len(str(base.get(k)))):
#                     upd[k] = nv[k]
#             if upd != base:
#                 existing[i] = upd
#                 changed.append(upd)
#         else:
#             existing.append(nv)
#             changed.append(nv)
#     return existing, changed

def merge_variables(existing: List[Dict[str, Any]], new_vars: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    合并变量列表，返回合并后的列表和发生变化的变量列表
    
    Args:
        existing: 现有变量列表
        new_vars: 新变量列表
        
    Returns:
        Tuple[合并后的所有变量, 发生变化的变量列表]
    """
    # 创建符号索引，同时过滤无效数据
    by_symbol = {}
    for i, v in enumerate(existing or []):
        if isinstance(v, dict) and v.get("symbol"):
            sym = (v["symbol"] or "").strip()
            if sym:
                by_symbol[sym] = v
    changed = []
    for new_var in (new_vars or []):
        if not isinstance(new_var, dict) or not new_var.get("symbol"):
            continue  
        sym = (new_var["symbol"] or "").strip()
        if not sym:
            continue
        old_var = by_symbol.get(sym)
        if old_var is None:
            # 新变量，直接添加
            by_symbol[sym] = new_var
            changed.append(new_var)
        else:
            # 合并现有变量
            merged_var = dict(old_var)
            updated = False
            for key in ["name", "unit", "description", "formula", "default_value"]:
                new_value = new_var.get(key)
                old_value = old_var.get(key)
                # 只有当新值存在且与旧值不同时才更新
                if new_value is not None and new_value != old_value:
                    merged_var[key] = new_value
                    updated = True
            if updated:
                by_symbol[sym] = merged_var
                changed.append(merged_var)
    return list(by_symbol.values()), changed

# def clear_variables():
#     """
#     删除变量表文件。返回 (ok: bool, info: str)
#     """
#     try:
#         if VAR_FILE.exists():
#             VAR_FILE.unlink()
#             return True, f"deleted: {VAR_FILE}"
#         else:
#             return False, f"not found: {VAR_FILE}"
#     except Exception as e:
#         return False, f"failed: {VAR_FILE} :: {e}"
def clear_variables(st_session_state=None) -> Tuple[bool, str]:
    p = variables_path(st_session_state)
    try:
        if p.exists():
            p.unlink()
            return True, f"deleted: {p.name}"
        return False, "not found"
    except Exception as e:
        return False, f"failed: {e}"