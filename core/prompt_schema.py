# core/prompt_schema.py
import re
from typing import Dict, List

# 统一的五段式键
CANON_KEYS = ["task", "examples", "cot", "question", "outfmt"]

# 各段可接受的“标题”同义词（大小写不敏感）
LABEL_ALIASES: Dict[str, List[str]] = {
    "task": [
        r"task(?:\s*spec(?:ification)?)?",  # Task / Task spec / Task specification
        r"任务(?:说明|规格)?"
    ],
    "examples": [
        r"examples?",                       # Example / Examples
        r"示例"
    ],
    "cot": [
        r"cot",                             # CoT
        r"chain[-\s]?of[-\s]?thought",      # Chain-of-Thought / Chain of Thought
        r"思路|推理"
    ],
    "question": [
        r"question",                        # Question
        r"problem",                         # Problem
        r"问题"
    ],
    "outfmt": [
        r"output\s*format",                 # Output format / Outputformat
        r"out\s*format",
        r"outformat",
        r"输出\s*格式"
    ],
}

# 生成用于整体匹配的正则：(?P<label>...):  后接内容直到下一个 label
# 说明：只要用户写了 “标题: 内容”，就能被识别；段与段之间可换行/逗号/空格/分号等任意符号
LABEL_GROUP = "|".join(
    [f"(?:{pat})" for pats in LABEL_ALIASES.values() for pat in pats]
)
# 如： (task|task spec|examples|cot|chain-of-thought|problem|question|output format|outformat) \s*:
LABEL_REGEX = re.compile(
    rf"(?i)(?P<label>{LABEL_GROUP})\s*:",
    flags=re.I | re.S
)

def _label_to_key(label: str) -> str:
    """把匹配到的标签文本映射为规范键（task/examples/cot/question/outfmt）"""
    low = label.lower().strip()
    for key, patterns in LABEL_ALIASES.items():
        for pat in patterns:
            if re.fullmatch(pat, low, flags=re.I):
                return key
    return ""  # 未知标签（正常情况下不会到达）

def parse_five_parts(user_text: str) -> Dict[str, str]:
    """
    解析五段式输入：
    - 允许任意顺序，允许只提供子集；
    - 各段标题可用多种同义（大小写不敏感，含中英文常用写法）；
    - 段与段之间可用换行/逗号/空格/分号等任意内容分隔；
    - 如果完全没有匹配到任何“标题:”，则整段作为 Question，其余置空。
    - 同一段标题若出现多次，将内容按换行拼接。
    返回 dict: {"task","examples","cot","question","outfmt"}
    """
    parts = {k: "" for k in CANON_KEYS}
    if not user_text or not user_text.strip():
        return parts

    text = user_text.strip()

    # 找到所有“标题:”的位置
    matches = list(LABEL_REGEX.finditer(text))
    if not matches:
        # 没有任何“标题:” → 全文作为 Question
        parts["question"] = text
        return parts

    # 有标题：按出现顺序切片
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        seg = text[start:end].strip().strip("，,;； \n\t")
        key = _label_to_key(m.group("label"))
        if not key:
            continue
        if parts[key]:
            parts[key] = parts[key].rstrip() + "\n" + seg
        else:
            parts[key] = seg

    return parts

def build_prompt_with_role(role_text: str, parts: Dict[str, str], rag_snippets: str = "", sources_list=None) -> list:
    """
    构造完整 prompt (system+user)，仅包含非空段落。
    """
    sys_msg = {"role": "system", "content": role_text}
    user_msg = "\n"

    if parts.get("task"):
        user_msg += f"\nTask specification:\n{parts['task']}\n"
    if parts.get("examples"):
        user_msg += f"\nExamples:\n{parts['examples']}\n"
    if parts.get("cot"):
        user_msg += f"\nCoT:\n{parts['cot']}\n"
    if parts.get("question"):
        user_msg += f"\nQuestion:\n{parts['question']}\n"
    if parts.get("outfmt"):
        user_msg += f"\nOutput format:\n{parts['outfmt']}\n"

    if rag_snippets:
        user_msg += f"\n---\n以下为可用的参考资料片段，请综合利用：\n{rag_snippets}\n"

    if sources_list:
        user_msg += "\n---\n自动检索的文献资料：\n"
        for i, s in enumerate(sources_list, 1):
            title = s.get("title") or ""
            year = s.get("year") or ""
            abstract = s.get("abstract") or ""
            user_msg += f"[{i}] {title} ({year} main content: {abstract})\n"

    return [sys_msg, {"role": "user", "content": user_msg}]

def build_prompt_no_role(parts: Dict[str, str], rag_snippets: str = "", sources_list=None) -> List:
    """
    构造完整 prompt (system+user)，仅包含非空段落。
    """
    user_msg = "\n"

    if parts.get("task"):
        user_msg += f"\nTask specification:\n{parts['task']}\n"
    if parts.get("examples"):
        user_msg += f"\nExamples:\n{parts['examples']}\n"
    if parts.get("cot"):
        user_msg += f"\nCoT:\n{parts['cot']}\n"
    if parts.get("question"):
        user_msg += f"\nQuestion:\n{parts['question']}\n"
    if parts.get("outfmt"):
        user_msg += f"\nOutput format:\n{parts['outfmt']}\n"

    if rag_snippets:
        user_msg += f"\n---\n以下为可用的参考资料片段，请综合利用：\n{rag_snippets}\n"

    if sources_list:
        user_msg += "\n---\n自动检索的文献资料：\n"
        for i, s in enumerate(sources_list, 1):
            title = s.get("title") or ""
            year = s.get("year") or ""
            abstract = s.get("abstract") or ""
            user_msg += f"[{i}] {title} ({year} main content: {abstract})\n"

    return [{"role": "user", "content": user_msg}]