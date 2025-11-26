# core/utils/json_extract.py
from __future__ import annotations
import re, json

def extract_json_best_effort(text: str):
    """尽力从文本里提取一个 JSON（优先 {…}，其次 […]）。失败则 {}。"""
    if not text:
        return {}
    # 优先对象
    objs = re.findall(r'\{[\s\S]*?\}', text)
    # 没对象就试数组
    if not objs:
        objs = re.findall(r'\[[\s\S]*?\]', text)
    # 倒序尝试（很多模型把最终 JSON 放在最后）
    for blob in reversed(objs):
        try:
            parsed = json.loads(blob)
            if isinstance(parsed, dict):
                return parsed
            if isinstance(parsed, list):
                return {"items": parsed}  # 避免 schema 报错
        except Exception:
            continue
    return {}
