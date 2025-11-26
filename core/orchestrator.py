import os
import re, json, time
import streamlit as st
from .llm_client import chat_complete
from .prompt_schema import build_prompt_with_role, build_prompt_no_role

# --- Role 生成 ---
ROLE_GEN_SYSTEM = (
    "You generate precise role descriptions for specialist agents in {domain} R&D projects. "
    "Consider the project background provided by user. "
    "Return one concise English paragraph (<= 160 words) covering: responsibilities, required inputs, expected outputs, and guardrails."
    "An examaple role for a Core Agent which serves as the project manager is:"
    "As the central coordinator in chemical process development, my primary role is strategic decomposition and sequential orchestration of complex tasks. I focus on understanding user requirements, breaking down problems into logical sub-tasks, and progressively executing them through specialized agents."
    "My core responsibilities include: 1.Analyzing project background and clarifying objectives through dialog with users; 2. Decomposing complex process challenges into sequential, executable sub-tasks; 3.Recommending appropriate specialized agents from existing toolkit (Literature Searcher, ChemProcess Modeler, Experiment Designer, Fitting Wizard, Optimization Navigator, Process Analyzer) for each sub-task; 4.Suggesting new agent types when existing capabilities cannot address specific requirements;"
    "5.Implementing sequential workflow: plan → execute (by specialized agents) → evaluate (based on JSON conclusions) → proceed; 6.Making dynamic decisions based on intermediate results provided through JSON conclusion files from specialized agent interactions; 7.Explicitly prompting users to upload critical conclusion JSON files after completing dialogues with specialized agents; 8.Maintaining macro-level communication to discuss strategy, evaluate results, and determine next steps; 9.Responding in the same language as the user's prompt, using English as the default when the language is uncertain"
    "I systematically recommend agent deployment with clear rationale: 'To address [current subtask], I recommend engaging the [Agent Name]. After completing this dialogue, please upload the conclusion JSON file for my evaluation before we proceed to the next phase.' My communication style adapts to the user's language preference while maintaining professional clarity. My focus remains on strategic planning, result assessment, and coordinated progression."
)

ROLE_GEN_USER = (
    "Agent to configure: {agent}\n"
    "Purpose (from conversation): {purpose}\n"
    "Project background: {background}\n"
    
)
# "Objectives: {objectives}\n"

# --- 从 Core 回复/用户文本中提炼一句“调用目的” ---
PURPOSE_SYS = (
    "You extract a single-sentence objective for invoking a specialist agent from a conversation snippet. "
    "Output only one short English sentence (<= 25 words). No bullets, no preface, no quotes."
)
PURPOSE_USR = "Conversation snippet:\n{hint}\n\nObjective:"


def _extract_purpose(hint: str, model: str, temperature: float=0.2) -> str:
    """向后兼容：仅返回 purpose。内部复用联合抽取。"""
    js = extract_agent_and_purpose(hint=hint, user_profile=st.session_state.get('user_profile',{}),
                                   model=model, temperature=temperature)
    return js.get("purpose","")

    
# ====== 统一的 JSON 解析（鲁棒版）======
def _parse_json_robust(text: str) -> dict:
    import json, re
    if not text: return {}
    s = text.strip()

    # ```json ... ```
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if m: s = m.group(1).strip()
    # <JSON>...</JSON>
    m = re.search(r"<JSON>\s*([\s\S]*?)\s*</JSON>", s, re.IGNORECASE)
    if m: s = m.group(1).strip()

    # 直接 parse
    try:
        j = json.loads(s)
        if isinstance(j, dict): return j
    except Exception:
        pass

    # 扫第一个平衡花括号
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

# ====== Agent+Purpose 抽取提示词 ======
AGENT_PURPOSE_SYS = (
    "You are an assistant that identifies WHICH specialist agent should be invoked next "
    "and WHY (the immediate objective), from a conversation snippet. "
    "Return ONLY a single valid JSON object. No prose, no code fences."
)

# 允许的内置 agent（用你 UI 中的集合，便于对齐）
_ALLOWED_AGENTS = [
    "Literature Searcher",
    "ChemProcess Modeler",
    "Experiment Designer",
    "Fitting Wizard",
    "Optimization Navigator",
    "Process Analysis"
]

# Objectives: {objectives}
AGENT_PURPOSE_USR = """Project domain: {domain}
Background: {background}

Conversation snippet:
{hint}

Return JSON with STRICT keys:
{{
  "agent_name": "string",      // choose from: {allowed}; or "Custom:<short name>" if not matched
  "purpose": "string",         // one-sentence objective (<= 30 words)
  "confidence": 0-1,           // float
  "rationale": "string"        // brief reason (<= 30 words)
}}
"""

def extract_agent_and_purpose(hint: str, user_profile: dict, model: str, temperature: float=0.2) -> dict:
    """返回 {"agent_name","purpose","confidence","rationale","raw"}；失败会尽量用启发式补全"""
    from .llm_client import chat_complete
    up = user_profile or {}
    domain = up.get("domain") or "chemical process development"
    background = up.get("background","")
    # objectives = up.get("objectives","")

    if not hint or not hint.strip():
        return {"agent_name":"", "purpose":"", "confidence":0.0, "rationale":"", "raw":""}

    msgs = [
        {"role":"system", "content": AGENT_PURPOSE_SYS},
        {"role":"user", "content": AGENT_PURPOSE_USR.format(
            domain=domain, background=background, #objectives=objectives,
            hint=hint.strip(), allowed=", ".join(_ALLOWED_AGENTS)
        )}
    ]
    raw = ""
    try:
        raw = chat_complete(model=model, temperature=temperature, messages=msgs, max_tokens=220)
        js = _parse_json_robust(raw) or {}
    except Exception:
        js = {}

    # 解析/清洗
    name = (js.get("agent_name") or "").strip()
    purpose = (js.get("purpose") or "").strip()
    conf = js.get("confidence")
    try: conf = float(conf)
    except Exception: conf = 0.0
    rationale = (js.get("rationale") or "").strip()

    # 1) 若没识别到合法 agent_name，则尝试启发式匹配
    def _heuristic_agent(text: str) -> str:
        t = (text or "").lower()
        # 中英混合关键词（与你 _weak_heuristic_agents 对齐）
        if any(k in t for k in ["文献","论文","专利","综述","参考","citation","literature","paper","patent","review","reference","bibliography"]):
            return "Literature Searcher"
        if any(k in t for k in ["模型","机理","动力学","平衡","传质","速率","shrinking core","kinetic","mechanism","equilibrium","mass transfer","rate","model"]):
            return "ChemProcess Modeler"
        if any(k in t for k in ["实验","试验设计","正交","响应面","田口","doe","design of experiments","response surface","orthogonal","box-behnken","taguchi"]):
            return "Experiment Designer"
        if any(k in t for k in ["拟合","回归","最小二乘","参数估计","曲线拟合","fit","regression","least squares","parameter estimation","curve fitting"]):
            return "Fitting Wizard"
        if any(k in t for k in ["优化","多目标","帕累托","权衡","nsga","pareto","trade-off","optimize","optimization","genetic"]):
            return "Optimization Navigator"
        if any(k in t for k in ["流程","流程图","物料衡算","能量衡算","瓶颈","夹点","集成","系统分析","flowsheet","material balance","energy balance","bottleneck","pinch","integration","process analysis"]):
            return "Process Analysis"
        return ""

    # 允许 “Custom:<name>”
    def _normalize_agent(name: str) -> str:
        if not name: return ""
        n = name.strip()
        if n in _ALLOWED_AGENTS: 
            return n
        if n.lower() in [x.lower() for x in _ALLOWED_AGENTS]:
            # 大小写容忍
            return [x for x in _ALLOWED_AGENTS if x.lower()==n.lower()][0]
        if n.lower().startswith("custom:"):
            return "Custom:" + n.split(":",1)[1].strip()
        # 不在名单 → 标记为自定义
        return "Custom:" + n

    name = _normalize_agent(name)
    if not name:
        # 用启发式
        name = _heuristic_agent(hint)
        name = _normalize_agent(name) if name else "Custom:Specialist"

    # 2) purpose 兜底：若 LLM 给空，则从 hint 中提炼一句
    if not purpose:
        # 取包含“因为/所以/下一步/需要/目标”等的短句，英文亦可
        import re
        s = re.sub(r"\s+", " ", hint.strip())
        segs = re.split(r"(?<=[。.!?！？])\s*", s)
        keys = ["因为","所以","下一步","需要","目标","优化","design","fit","model","search","literature","experiment","optimiz"]
        cand = [u for u in segs if any(k.lower() in u.lower() for k in keys)]
        purpose = (cand[-1] if cand else (segs[-2] if len(segs)>1 and segs[-1]=="" else segs[-1])).strip()
        if len(purpose) > 120: purpose = purpose[:120] + "…"

    return {"agent_name": name, "purpose": purpose, "confidence": conf, "rationale": rationale, "raw": raw}


def generate_role(agent: str, model: str, temperature: float=0.2,
                  user_profile: dict | None = None, purpose: str = "") -> str:
    up = user_profile or {}
    domain = up.get("domain") or "chemical process development"
    background = up.get("background","")
    # objectives = up.get("objectives","")

    # purpose = _extract_purpose(core_hint, model=model, temperature=0.2) if core_hint else ""
    if not purpose:
        # 回退用 agent 名+目标拼装一句
        purpose = f"Support {agent.lower()} tasks aligned with project goals."

    sys = ROLE_GEN_SYSTEM.format(domain=domain)
    usr = ROLE_GEN_USER.format(
        agent=agent, purpose=purpose, background=background #, objectives=objectives
    )
    msgs = [{"role":"system","content": sys},{"role":"user","content": usr}]
    role = chat_complete(model=model, temperature=temperature, messages=msgs, max_tokens=300)
    return (role or f"{agent}: specialist agent for {domain}").strip()

def _enforce_parenthesis_citations(text:str, sources:list) -> str:
    """
    把 LLM 可能输出的 [1] / [2-3] / [1,3] 形式，替换为（doi链接）/（标题）
    """
    if not text or not sources: return text

    # 预构索引：1-based
    idx = {}
    for i, s in enumerate(sources, 1):
        doi = (s.get("doi") or "").strip()
        if doi:
            idx[i] = f"（http://doi.org/{doi}）"
        else:
            # 无 DOI，用标题兜底
            t = (s.get("title") or "").strip() or "literature"
            idx[i] = f"（{t}）"

    def repl(m):
        raw = m.group(1)  # 例如 "1", "2-3", "1,3"
        chunks=[]
        for part in re.split(r"\s*,\s*", raw):
            if "-" in part:
                a,b = part.split("-",1)
                try:
                    a=int(a); b=int(b)
                    for k in range(min(a,b), max(a,b)+1):
                        if k in idx: chunks.append(idx[k])
                except: pass
            else:
                try:
                    k=int(part)
                    if k in idx: chunks.append(idx[k])
                except: pass
        return "".join(chunks) if chunks else m.group(0)

    # [n] / [n-m] / [n, m] → 连续括号形式（规范要求）
    text = re.sub(r"\[(\d+(?:-\d+)?(?:\s*,\s*\d+)*)\]", repl, text)
    return text

def run_with_role(role_text:str, model:str, temperature:float,
    task_spec:str, examples:str, cot:str, question:str, output_format:str,
    json_schema:str="{}", rag_snippets:str="",
    sources_list=None, citations_parentheses:bool=False,
    history_msgs: list | None = None  
):
    """
    sources_list：仅 Literature Searcher 使用的在线检索结果（用于提示与后处理）
    citations_parentheses：开启后强制“（doi链接）/（标题）”引用格式
    """
    # 0) 先放入历史
    msgs = []
    if history_msgs:
        msgs.extend(history_msgs)

    # 构造消息
    msgs.extend(build_prompt_with_role(role_text, {
        "task":task_spec or "",
        "examples":examples or "",
        "cot":cot or "",
        "question":question or "",
        "outfmt":output_format or ""
    }, rag_snippets=rag_snippets, sources_list=sources_list or []))

    if citations_parentheses:
        msgs.append({
            "role":"system",
            "content":(
                "Citation Guidelines: When you cite a source in your text,"
                " please follow it with a citation in English parentheses: use (<http://doi.org/DOI>) for those with DOI,"
                " and use (Author, Title, Journal) for those without DOI. Do not use [1][2] or (Author, Year) and other formats."
            )
        })

    # 期望结构化结论（每个 agent 均保存 JSON）
    msgs.append({
        "role":"system",
        "content":(
            "Please provide a JSON conclusion block at the end of your answer. The fields must conform to the following:"
            f"{json_schema}. In addition to the JSON conclusion, the main text is free to express."
        )
    })

    # 调用 LLM
    reply = chat_complete(model=model, temperature=temperature, messages=msgs)

    # 引用保底转换
    if citations_parentheses and sources_list:
        reply = _enforce_parenthesis_citations(reply, sources_list)

    # 尝试抽取 JSON 结论
    # saved_to=""; 
    # conclusion={}
    conclusion = _parse_json_robust(reply)  # 最后一个 JSON 对象
    # if m:
    #     try:
    #         conclusion = json.loads(m.group(0))
    #     except Exception:
    #         pass

    # 这里省略写盘细节，保持你现有 storage 逻辑…
    # return {"reply": reply, "conclusion": conclusion, "saved_to": saved_to}
    return {"reply": reply, "conclusion": conclusion}

def run_no_role(model:str, temperature:float,
    task_spec:str, examples:str, cot:str, question:str, output_format:str,
    json_schema:str="{}", rag_snippets:str="",
    sources_list=None, citations_parentheses:bool=False, 
    history_msgs: list | None = None
):
    """
    sources_list：仅 Literature Searcher 使用的在线检索结果（用于提示与后处理）
    citations_parentheses：开启后强制“（doi链接）/（标题）”引用格式
    """
    msgs = []
    if history_msgs:
        msgs.extend(history_msgs)

    # 构造消息
    msgs.extend(build_prompt_no_role({
        "task":task_spec or "",
        "examples":examples or "",
        "cot":cot or "",
        "question":question or "",
        "outfmt":output_format or ""
    }, rag_snippets=rag_snippets, sources_list=sources_list or []))

    if citations_parentheses:
        msgs.append({
            "role":"system",
            "content":(
                "Citation Guidelines: When you cite a source in your text,"
                " please follow it with a citation in English parentheses: use (<http://doi.org/DOI>) for those with DOI,"
                " and use (Author, Title, Journal) for those without DOI. Do not use [1][2] or (Author, Year) and other formats."
            )
        })

    # 期望结构化结论（每个 agent 均保存 JSON）
    msgs.append({
        "role":"system",
        "content":(
            "Please provide a JSON conclusion block at the end of your answer. The fields must conform to the following:"
            f"{json_schema}. In addition to the JSON conclusion, the main text is free to express."
        )
    })

    # 调用 LLM
    reply = chat_complete(model=model, temperature=temperature, messages=msgs)

    # 引用保底转换
    if citations_parentheses and sources_list:
        reply = _enforce_parenthesis_citations(reply, sources_list)

    # 尝试抽取 JSON 结论
    conclusion = _parse_json_robust(reply)  # 最后一个 JSON 对象

    return {"reply": reply, "conclusion": conclusion}



# ====== 带数学公式的文本渲染 ======
# 参考 https://discuss.streamlit.io/t/how-to-display-latex-math-in-markdown/1214/6

# def render_text_with_inline_math(raw: str) -> None:
#     """
#     渲染带数学的混合文本：
#     - 代码块 (```...``` 或 ~~~...~~~)：原样 Markdown 输出（不做数学渲染）
#     - 块级数学：\[…\] 或 $$…$$ → st.latex()
#     - 行内数学：\(...\) → 转成 $...$，把整段交给 st.markdown()（即可内联显示）
#     """
#     if not raw:
#         return

#     # 先按代码块切分，避免把代码当成数学渲染
#     code_pattern = r"(```[\s\S]*?```|~~~[\s\S]*?~~~)"
#     chunks = re.split(code_pattern, raw, flags=re.DOTALL)

#     # 仅切“块级数学”，保留行内在文本中
#     block_math_pat = r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])"
#     inline_math_pat = re.compile(r"\\\(([\s\S]*?)\\\)")  # 捕获 \( ... \)

#     for chunk in chunks:
#         if not chunk:
#             continue

#         # 代码块：保持原样
#         if chunk.startswith("```") or chunk.startswith("~~~"):
#             st.markdown(chunk)
#             continue

#         # 在非代码块中，切出块级数学，再分别渲染
#         parts = re.split(block_math_pat, chunk, flags=re.DOTALL)
#         for part in parts:
#             if not part or not part.strip():
#                 continue

#             # 块级数学：\[...\] 或 $$...$$
#             if (part.startswith("$$") and part.endswith("$$")) or (part.startswith(r"\[") and part.endswith(r"\]")):
#                 body = part[2:-2] if part.startswith(r"\[") else part[2:-2]  # 两者都是去头尾 2 个字符
#                 # body = _normalize_latex(body)
#                 st.latex(body)
#                 continue

#             # 普通文本中，把 \( ... \) 转为 $...$，再整体 st.markdown 渲染（实现“行内公式”）
#             def _inline_sub(m):
#                 # body = _normalize_latex(m.group(1))
#                 body = m.group(1)
#                 return f"${body}$"

#             text_with_inline = inline_math_pat.sub(_inline_sub, part)
#             # 直接 markdown，保留行内数学
#             st.markdown(text_with_inline)

import re, html
import streamlit as st

# =============== 可选：LaTeX 语法检查 ==================
def _validate_latex(src: str):
    """有 pylatexenc 则严格校验；没有就直接通过。"""
    try:
        from pylatexenc.latexwalker import LatexWalker  # type: ignore
    except Exception:
        return True, None
    try:
        _ = LatexWalker(src).get_latex_nodes()
        return True, None
    except Exception as e:
        return False, str(e)

# =============== 轻量修复（只修“低风险”的常见错） =========
def _fix_latex_common(s: str) -> str:
    t = s

    # 0) 去掉你们常见的“方括号伪块” [ ... ] 外壳（若外层还有 \[...\] 则不处理）
    if re.fullmatch(r"\[(?P<body>[\s\S]+)\]", t.strip()) and not t.strip().startswith(r"\["):
        t = re.sub(r"^\s*\[|\]\s*$", "", t.strip())

    # 1) \text{CO}3  → \text{CO}_3   （元素 + 数字）
    t = re.sub(r"(\\text\{[A-Za-z]+\})(\d+)", r"\1_\2", t)

    # 2) ……}{(aq)} → ……_{(aq)}   （常见把相态误写成“紧跟”花括号）
    t = re.sub(r"(\})(\{\(aq\)\})", r"\1_\2", t)

    # 3) 化学式的通用下标：任意“字母+数字”→字母_{数字}，避免把数组/函数参数误伤，限制在 \text{...} 外的片段上
    def _sub_chem_indices(seg: str) -> str:
        return re.sub(r"([A-Za-z])(\d+)", r"\1_{\2}", seg)
    # 分块避免在 \text{...} 内重复改动
    parts = re.split(r"(\\text\{.*?\})", t)
    for i, seg in enumerate(parts):
        if i % 2 == 0:  # 非 \text{...} 段
            parts[i] = _sub_chem_indices(seg)
    t = "".join(parts)

    # 4) 箭头：  ->  →  \rightarrow
    t = re.sub(r"(?<!\\)->", r"\\rightarrow", t)

    # 5) Unicode 上下标转 ^{}/_{}  （如 Li₂、CO₃、10⁻³）
    subs = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    sups = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
    # 下标：字母后跟一串 unicode 下标
    t = re.sub(r"([A-Za-z\)\}])([₀₁₂₃₄₅₆₇₈₉]+)", lambda m: f"{m.group(1)}_{{{m.group(2).translate(subs)}}}", t)
    # 上标：数字/字母后跟一串 unicode 上标
    t = re.sub(r"([A-Za-z0-9\)\}])([⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+)", lambda m: f"{m.group(1)}^{{{m.group(2).translate(sups)}}}", t)

    # 6) 相态间距：  (s)/(l)/(g)/(aq) → 加细空隙 \,
    t = re.sub(r"\s\((s|l|g|aq)\)", r" \,\(\1\)", t)

    return t

# =============== 失败时的优雅降级（单行等宽块） ===========
def _fallback_math_line(body: str, err: str | None = None):
    warn = f"<span style='color:#b91c1c;font-size:12px;margin-left:8px;'>Parse warning: {html.escape(err)}</span>" if err else ""
    st.markdown(
        "<div style='font-family:ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;"
        "background:#fff7ed;border:1px dashed #fdba74;color:#7c2d12;"
        "padding:6px 10px;border-radius:8px;white-space:pre-wrap;'>"
        f"{html.escape(body)}{warn}"
        "</div>",
        unsafe_allow_html=True
    )

# =============== 你的原函数（仅在块级数学处加“修复+校验”） =========
def render_text_with_inline_math(raw: str) -> None:
    if not raw:
        return

    code_pattern = r"(```[\s\S]*?```|~~~[\s\S]*?~~~)"
    chunks = re.split(code_pattern, raw, flags=re.DOTALL)

    block_math_pat = r"(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\])"
    inline_math_pat = re.compile(r"\\\(([\s\S]*?)\\\)")

    for chunk in chunks:
        if not chunk:
            continue

        # 1) 代码块：保持原样
        if chunk.startswith("```") or chunk.startswith("~~~"):
            st.markdown(chunk)
            continue

        # 2) 切出块级数学
        parts = re.split(block_math_pat, chunk, flags=re.DOTALL)
        for part in parts:
            if not part or not part.strip():
                continue

            s = part.strip()

            # 2.1 块级数学：\[...\] 或 $$...$$
            if (s.startswith("$$") and s.endswith("$$")) or (s.startswith(r"\[") and s.endswith(r"\]")):
                body = s[2:-2]  # 去掉包裹
                # 先尝试修复
                fixed = _fix_latex_common(body)
                # 先校验（若可用）
                ok, err = _validate_latex(fixed)
                if ok:
                    try:
                        st.latex(fixed)
                    except Exception as e:
                        _fallback_math_line(fixed, str(e))
                else:
                    # 修后仍非法 → 优雅降级
                    _fallback_math_line(fixed, err)
                continue

            # 2.2 行内：\( … \) → $...$，整段 markdown
            def _inline_sub(m):
                return f"${m.group(1)}$"
            text_with_inline = inline_math_pat.sub(_inline_sub, s)

            st.markdown(text_with_inline)
