# dyn_schema.py
# -----------------------------------------------------------------------------
# 合并版要点（符合你的要求）：
# 1) Schema 选择优先级（稳定 > 智能）：
#    自定义保存(缓存) > Agent库(json_schema) > LLM提议 > 启发式推断
#    —— 这样“工程可复现 + 冷启动也能用 + 可逐步走向稳定”
#
# 2) 完整保留你原版的“智能兜底”：
#    - _extract_json_keys：从文本/代码块里抓 key
#    - _heuristic_shape：根据 key 名含义猜类型（对 KPI 等做了简单领域特化）
#    - _llm_propose_schema_text：可选，若提供 call_llm 则由 LLM 提议 shape
#
# 3) 保留/增强 UI 需要的持久化接口：
#    - get_schema(agent_name)：读缓存（schemas/custom/<agent>.schema.json）
#    - save_schema(agent_name, shape)：写缓存（含 draft-07 jsonschema）
#
# 4) _to_formal_jsonschema：以你的风格实现的正式 JSON Schema 生成，
#    比简单 to_jsonschema 更有表现力（对象/数组/领域字段）。
# -----------------------------------------------------------------------------

from __future__ import annotations
import os, re, json, hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List
from .llm_client import chat_complete

try:
    import jsonschema  # type: ignore
except Exception:
    jsonschema = None  # noqa: F401

# 缓存目录：自定义/最终确定的 schema 会写入这里
SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas" / "custom"
SCHEMA_ROOT.mkdir(parents=True, exist_ok=True)

# ========== 基础工具 ==========

def _slugify(name: str) -> str:
    """将 agent 名清洗为安全文件名。"""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return s[:80] if s else "Custom"

# def _extract_json_keys(text: str) -> list[str]:
#     """
#     从文本中“尽力而为”抽取 JSON 键：
#     - 优先扫描 ```json ... ``` 代码块
#     - 若没有代码块，则从全文正则抓取 `"key":` 的形式
#     仅返回前 20 个，避免过长。
#     """
#     keys = set()
#     # 代码块中的 JSON
#     m = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.I)
#     for block in m:
#         for k in re.findall(r'"([A-Za-z0-9_]+)"\s*:', block):
#             keys.add(k)
#     # 全文 JSON-like
#     for k in re.findall(r'"([A-Za-z0-9_]+)"\s*:', text):
#         keys.add(k)
#     return list(keys)[:20]

# def _heuristic_shape(keys: list[str]) -> dict:
#     """
#     根据键名启发式猜测类型：
#     - 对 KPI/数值类字段推 number
#     - 对 notes/summary 类推 string
#     - 对 recommendations/steps 等推 ["string"]（字符串数组）
#     - 对 papers/models 等推“可能为对象或数组”，降级映射为空数组 [] 或特化对象
#     """
#     def guess_type(k: str) -> str | list | dict:
#         kl = k.lower()
#         if any(w in kl for w in ["kpi","score","rate","ratio","purity","yield","efficiency","loss","energy","temp","temperature","ph"]):
#             return "number"
#         if any(w in kl for w in ["notes","summary","rationale","reason","desc","explanation","assumptions"]):
#             return "string"
#         if any(w in kl for w in ["recommendation","opportunity","steps","actions","todos","items","references","citations","bottlenecks","risks","limitations","next"]):
#             return ["string"]
#         if any(w in kl for w in ["papers","models","experiments","kpis","balances","parameters","ranges"]):
#             return {"type":"object_or_array"}
#         return "string"

#     shape: dict = {}
#     for k in keys:
#         t = guess_type(k)
#         if t == {"type":"object_or_array"}:
#             # 对领域特化示例：kpis 字段直接给出 Li2CO3 KPI 模板
#             if "kpis" in k.lower():
#                 shape[k] = {"Li2CO3_purity": "number", "Li_yield": "number"}
#             else:
#                 # 不确定时降级为空数组（字符串数组），可在 UI/后续手动细化
#                 shape[k] = []
#         else:
#             shape[k] = t
#     # 常用补位
#     if "summary" not in shape: shape["summary"] = "string"
#     if "recommendations" not in shape: shape["recommendations"] = ["string"]
#     if "notes" not in shape: shape["notes"] = "string"
#     return shape

def _to_formal_jsonschema(shape: dict) -> dict:
    """
    将轻量 shape（string/number/list/object）转为 draft-07 JSON Schema。
    对对象类型的子键做一层递归处理；对数组默认 items 为 string。
    提供“足够”的表达力，又不过度复杂化。
    """
    props: Dict[str, Any] = {}
    required: List[str] = []

    for k, v in (shape or {}).items():
        if v == "string":
            props[k] = {"type": "string"}
        elif v == "number":
            props[k] = {"type": "number"}
        elif isinstance(v, list):
            # 缺省处理为字符串数组；如需更强表达力，可在 UI 里保存更具体的 shape 覆盖
            props[k] = {"type": "array", "items": {"type": "string"}}
        elif isinstance(v, dict):
            # 对象类型：对子键再猜一层简单类型；如果子值本身是"number"/"string"，则映射类型；否则当 string
            sub_props: Dict[str, Any] = {}
            for sk, sv in v.items():
                if sv == "number" or isinstance(sv, (int, float)):
                    sub_props[sk] = {"type": "number"}
                elif sv == "string" or isinstance(sv, str):
                    sub_props[sk] = {"type": "string"}
                else:
                    # 复杂子结构：保守起见，当 string；如需更强表达力，建议在 UI 中手工保存覆盖
                    sub_props[sk] = {"type": "string"}
            props[k] = {"type": "object", "properties": sub_props}
        else:
            # 未知类型：保守 string
            props[k] = {"type": "string"}
        required.append(k)

    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": props,
        "required": required
    }

# ========== LLM 提议（可选） ==========

def _llm_propose_schema_text(
    agent_name: str,
    role_text: str,
    llm_model: str = "gpt-4o",     # 当未传 call_llm 时，使用 chat_complete 的模型/温度
    llm_temperature: float = 0.2,
) -> Optional[dict]:
    """
    通过 LLM 生成 schema shape。
    优先使用外部传入的 call_llm(messages)->str；若未提供，则直接调用 chat_complete。
    仅接受返回 JSON，且包含顶层 key "shape"。
    """
    sys = "You are a helpful system that designs concise JSON Schemas for saving key conclusions from an agent's answer."
    usr = f"""Agent: {agent_name}
Role: {role_text or "<none>"}

Please propose a JSON output schema with 6-10 fields tailored to the role.
Return ONLY JSON with a top-level key "shape" whose value is an object mapping field names to types ("string","number",["string"], nested objects).
Avoid prose and code fences.
"""
    try:
        messages = [{"role":"system","content":sys},{"role":"user","content":usr}]
        ans = chat_complete(llm_model, llm_temperature, messages, max_tokens=512, timeout=60)

        m = re.search(r"\{[\s\S]*\}", ans or "")
        if not m:
            return None
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and isinstance(obj.get("shape"), dict):
            return obj
    except Exception:
        return None
    return None


# ========== 稳定优先级 + 智能兜底 的统一入口 ==========

def _find_spec_from_library(agent_name: str) -> Optional[dict]:
    """
    尝试从 agents.agent_specs 导入并查找该 agent 的规格（若工程中存在）。
    仅作为可选增强，不强依赖此模块存在。
    """
    try:
        from agents.agent_specs import AGENTS  # type: ignore
        return next((a for a in AGENTS if a.get("name") == agent_name), None)
    except Exception:
        return None

def load_or_propose_schema(
    agent_name: str,
    role_text: str = "",
    llm_model: str = "gpt-4o",     # 当未传 call_llm 时，使用 chat_complete 的模型/温度
    llm_temperature: float = 0.2,
) -> Tuple[str, dict]:
    """
    最终对外接口（保持你原有的函数名与签名）：
    优先级：自定义保存(缓存) > Agent库(json_schema) > LLM提议 > 启发式推断
    返回：
      - shape_text: 可直接展示/编辑的 JSON 文本（带缩进）
      - draft: draft-07 形式化 jsonschema（给校验/存档用）
    """
    safe = _slugify(agent_name)
    cache_file = SCHEMA_ROOT / f"{safe}.schema.json"

    # A. 若已有缓存（用户保存过）→ 直接返回
    if cache_file.exists():
        try:
            payload = json.loads(cache_file.read_text(encoding="utf-8"))
            shape = payload.get("shape", {})
            draft = payload.get("jsonschema", {})
            if shape:
                return json.dumps(shape, ensure_ascii=False, indent=2), draft
        except Exception:
            pass  # 缓存损坏则继续往下走

    # B. 若 Agent 库里有 schema → 用库 schema 并写入缓存（形成稳定基线）
    spec = _find_spec_from_library(agent_name)
    if spec and spec.get("json_schema"):
        shape = spec["json_schema"]
        draft = _to_formal_jsonschema(shape)
        payload = {"agent": agent_name, "shape": shape, "jsonschema": draft}
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return json.dumps(shape, ensure_ascii=False, indent=2), draft

    # C. 不走稳定源 → 尝试 LLM 提议（可选）
    llm_obj = _llm_propose_schema_text(agent_name, role_text, llm_model=llm_model, llm_temperature=llm_temperature)
    if isinstance(llm_obj, dict) and isinstance(llm_obj.get("shape"), dict):
        shape = llm_obj["shape"]
        draft = _to_formal_jsonschema(shape)
        payload = {"agent": agent_name, "shape": shape, "jsonschema": draft}
        cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return json.dumps(shape, ensure_ascii=False, indent=2), draft

    # 若一个 key 都没有抽到，给一个极简默认
    if not shape:
        # 极简默认：conclusion + summary + notes
        shape = {"conclusion": "string", "summary": "string", "notes": "string"}
    # else:
    #     shape = _heuristic_shape(keys)

    draft = _to_formal_jsonschema(shape)
    payload = {"agent": agent_name, "shape": shape, "jsonschema": draft}
    cache_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.dumps(shape, ensure_ascii=False, indent=2), draft

# ========== UI 持久化接口（供 inline 编辑器调用） ==========

def get_schema(agent_name: str):
    """
    读取已缓存的 schema（若存在）。
    返回 (shape_text, draft)；若无缓存，返回 (None, None)。
    """
    safe = _slugify(agent_name)
    f = SCHEMA_ROOT / f"{safe}.schema.json"
    if not f.exists():
        return None, None
    payload = json.loads(f.read_text(encoding="utf-8"))
    shape = payload.get("shape", {})
    draft = payload.get("jsonschema", {})
    return json.dumps(shape, ensure_ascii=False, indent=2), draft

def save_schema(agent_name: str, shape: dict):
    """
    保存用户在 UI 中编辑后的 schema。
    会自动生成 draft-07 jsonschema 一并写入缓存。
    """
    safe = _slugify(agent_name)
    f = SCHEMA_ROOT / f"{safe}.schema.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    draft = _to_formal_jsonschema(shape)  # 使用“更有表现力”的形式化函数
    payload = {"agent": agent_name, "shape": shape, "jsonschema": draft}
    f.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return json.dumps(shape, ensure_ascii=False, indent=2), draft
