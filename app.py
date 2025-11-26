import io, os, sys, json, time, tempfile, zipfile
from pathlib import Path
import pandas as pd
import streamlit as st
from datetime import datetime
from core.dyn_schema import get_schema, save_schema, load_or_propose_schema
from streamlit_extras.stylable_container import stylable_container

# local imports
sys.path.append(str(Path(__file__).parent))
from core.models import load_model_list
from core.orchestrator import run_with_role, run_no_role, generate_role, extract_agent_and_purpose, _extract_purpose, render_text_with_inline_math
from core.storage import latest_entry_path, save_session
from core.prompt_schema import parse_five_parts
from core.rag import build_rag_ctx_and_log
from agents.agent_specs import AGENTS
from core.lit_search import unified_search, make_online_snippets, rank_sources_by_relevance, unified_search_with_boolean
from core.query_builder import extract_keywords, build_boolean_query
from core.query_refiner import refine_query_with_llm
from core.variable_agent import _maybe_changed_by_text, extract_variables, load_variables, save_variables, merge_variables, clear_variables
from core.session_gc import get_session_id, touch_heartbeat, gc_stale_sessions, save_history_json, list_history_files, resolve_history_paths, history_name_to_path, variables_path
st.set_page_config(
    page_title='LLM4PDG: One-stop auxiliary design',
    layout='wide',
    initial_sidebar_state='expanded'
)

# --- Global UI polish: buttons & chat bg ---
st.markdown("""
<style>
/* 1) 整体基调：白色聊天背景 */
section.main > div.block-container { background: #ffffff; }

/* 2) 主按钮（type="primary"）：蓝色渐变、悬停暗一点 */
div[data-testid="stButton"] > button[kind="primary"],
div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
button[kind="primary"], 
button[data-testid="baseButton-primary"] {
  background-color: #27a5ff !important;  /* 覆盖内联 background-color */
  background-image: linear-gradient(135deg, #27a5ff 0%, #0997f6 100%) !important; /* 真正的渐变 */
  border-color: #0d8ae8 !important;      /* 覆盖边框颜色（部分版本会用到） */
  color: #fff !important;
  border: none !important;
  box-shadow: 0 2px 10px rgba(9,151,246,.25) !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-primary"]:hover,
button[kind="primary"]:hover,
button[data-testid="baseButton-primary"]:hover {
  filter: brightness(0.95) !important;
  transform: translateY(-1px) !important;
}

/* 3) 次按钮（type="secondary"）：灰色描边、浅背景 */
div[data-testid="stButton"] > button[kind="secondary"],
div[data-testid="stButton"] > button[data-testid="baseButton-secondary"],
button[kind="secondary"],
button[data-testid="baseButton-secondary"] {
  background: #f7f7f9 !important; color: #1d2939 !important;
  border: 1px solid #e5e7eb !important;
}

div[data-testid="stButton"] > button[kind="secondary"]:hover,
div[data-testid="stButton"] > button[data-testid="baseButton-secondary"]:hover,
button[kind="secondary"]:hover,
button[data-testid="baseButton-secondary"]:hover {
  background: #eef2f7 !important;
}           

.pill {
  display: inline-block;
  padding: 0.35em 1em;
  border-radius: 10px;   /* 胶囊形状 */
  font-size: 0.95em;
  font-weight: 600;
  color: white;
  box-shadow: 0 2px 6px rgba(0,0,0,0.15);  /* 柔和阴影 */
}
.pill-core { background: linear-gradient(135deg,#27a5ff,#0997f6); }
.pill-agent { background: linear-gradient(135deg,#7c3aed,#6d28d9); }
.pill-exp { background: linear-gradient(135deg,#10b981,#059669); }
.pill-role{ background: linear-gradient(135deg, #f77f81 0%, #e75480 100%) }

.gem-pill {
  display: inline-block;
  padding: 0.45em 1.5em;
  border-radius: 999px;
  font-size: 1.05em;
  font-weight: 600;
  color: #ffffff;
  background: radial-gradient(circle at center,
    #86efac 0%,   /* 中浅绿色（比 #6ee7b7 稍浅） */
    #60a5fa 42%,  /* 中浅蓝色（比 #3b82f6 浅一些，但比 #93c5fd 深） */
    #a78bfa 100%  /* 中浅紫色（比 #9333ea 浅，但比 #c4b5fd 深） */
  );
  box-shadow: 0 3px 10px rgba(0,0,0,0.14);
  letter-spacing: 0.4px;
  text-shadow: 0 1px 2px rgba(0,0,0,0.22);
}
            
.gem-pill-sub {
  display: inline-block;
  padding: 0.4em 1.3em;
  border-radius: 999px;
  font-size: 0.95em;
  font-weight: 600;
  color: #ffffff;
  background: radial-gradient(circle at center,
    #f472b6 0%,   /* 粉红中心 (Pink 400) */
    #fdba74 40%,  /* 橙色过渡 (Orange 500) */
    #c084fc 100%  /* 浅紫收尾 (Violet 400) */
  );
  box-shadow: 0 3px 9px rgba(0,0,0,0.12);
  letter-spacing: 0.3px;
  text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
            
</style>
""", unsafe_allow_html=True)

# ---------- Session init ----------
def _init_state():
    ss = st.session_state
    ss.setdefault('page', 'core')
    ss.setdefault('current_agent', None)
    ss.setdefault('chat', {'core': [], 'agents': {}})
    ss.setdefault('agent_roles', {})
    ss.setdefault('agent_schema', {})
    ss.setdefault('core_auto_rag', '')
    ss.setdefault('turn_core', 0)
    ss.setdefault('turn_agents', {})
    ss.setdefault('pending_core', None)
    ss.setdefault('pending_agent', None)
    ss.setdefault('core_suggest', None)
    ss.setdefault('proc_log', [])
    ss.setdefault('variables', {'isextracted': False, 'needextracted': False, 'table': []})
    ss.setdefault('agent_role_sent', {})   # 每个 agent 是否已经发送过 role：{name: True/False}
    ss.setdefault('core_role_sent', False) # Core 是否已发送过 role

_init_state()

# 初始化/心跳/GC —— 放在最顶层（每次 rerun 都会执行）
sid = get_session_id(st.session_state)
touch_heartbeat(st.session_state)
gc_stale_sessions(max_idle_minutes=60)  # 你可以改成 10 或 30

AV_CORE = "🧠"
AVMAP = {
    'Literature Searcher':'📚',
    'ChemProcess Modeler':'⚗️',
    'Experiment Designer':'🧪',
    'Fitting Wizard':'📈',
    'Optimization Navigator':'🧭',
    'Process Analyzer': '🗂️'   # ← 新增
}

CORE_ROLE_DEFAULT = """As the central coordinator in chemical process development, my primary role is strategic decomposition and sequential orchestration of complex tasks. I focus on understanding user requirements, breaking down problems into logical sub-tasks, and progressively executing them through specialized agents.
                    My core responsibilities include: 1.Analyzing project background and clarifying objectives through dialog with users; 2. Decomposing complex process challenges into sequential, executable sub-tasks; 3.Recommending appropriate specialized agents from existing toolkit (Literature Searcher, ChemProcess Modeler, Experiment Designer, Fitting Wizard, Optimization Navigator, Process Analyzer) for each sub-task; 4.Suggesting new agent types when existing capabilities cannot address specific requirements;
                    5.Implementing sequential workflow: plan → execute (by specialized agents) → evaluate (based on JSON conclusions) → proceed; 6.Making dynamic decisions based on intermediate results provided through JSON conclusion files from specialized agent interactions; 7.Explicitly prompting users to upload critical conclusion JSON files after completing dialogues with specialized agents; 8.Maintaining macro-level communication to discuss strategy, evaluate results, and determine next steps; 9.Responding in the same language as the user's prompt, using English as the default when the language is uncertain
                    I systematically recommend agent deployment with clear rationale: "To address [current subtask], I recommend engaging the [Agent Name]. After completing this dialogue, please upload the conclusion JSON file for my evaluation before we proceed to the next phase." My communication style adapts to the user's language preference while maintaining professional clarity. My focus remains on strategic planning, result assessment, and coordinated progression. Note that if the called agents from existing toolkit, their names must be exactly the same as the corresponding agent names when written into the json file."""

# ---Sidebar Helpers---
# --- (A) 生成 ZIP ---
# def _build_zip(paths:list) -> bytes:
#     buf = io.BytesIO()
#     with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
#         for p in paths:
#             try:
#                 # 归档名只保留文件名，避免泄露本地路径
#                 arcname = os.path.basename(p)
#                 zf.write(p, arcname=arcname)
#             except Exception as e:
#                 # 给一个错误说明文件，不影响整体下载
#                 zf.writestr(f"__error__/{os.path.basename(p)}.txt", f"Failed to add: {p}\nError: {e}")
#     buf.seek(0)
#     return buf.getvalue()

def _build_zip(paths: list) -> bytes:
    buf = io.BytesIO()

    # 1) 选择压缩方式：优先 DEFLATED，缺 zlib 时回退到 STORED
    compression = zipfile.ZIP_DEFLATED
    try:
        import zlib  # noqa
    except Exception:
        compression = zipfile.ZIP_STORED

    files_added = 0

    with zipfile.ZipFile(buf, mode="w", compression=compression) as zf:

        for raw in paths:
            p = Path(raw)
            try:
                if p.is_file():
                    # 只保留文件名，避免泄露完整路径
                    zf.write(str(p), arcname=p.name)
                    files_added += 1

                elif p.is_dir():
                    base = p
                    wrote_any_in_dir = False

                    # 递归写入目录内的所有文件
                    for f in base.rglob("*"):
                        if f.is_file():
                            rel = f.relative_to(base)                 # a/b.txt
                            arcname = str(Path(base.name) / rel)     # 目录名/a/b.txt
                            zf.write(str(f), arcname=arcname)
                            wrote_any_in_dir = True
                            files_added += 1

                    # 空目录：写一个占位目录项，避免“看起来是空包”
                    if not wrote_any_in_dir:
                        zinfo = zipfile.ZipInfo(str(Path(base.name)) + "/")
                        zf.writestr(zinfo, "")

                else:
                    # 路径不存在或不是常规文件/目录
                    zf.writestr(f"__error__/{p.name}.txt",
                                f"Path not found or not a regular file/dir:\n{p}")
                    files_added += 1

            except Exception as e:
                # 写入失败时，记录错误说明文件
                zf.writestr(f"__error__/{p.name}.txt",
                            f"Failed to add: {p}\nError: {repr(e)}")
                files_added += 1

        # 3) 如果一个条目都没加进去，写个提示文件，避免空包
        if files_added == 0:
            zf.writestr("__error__/EMPTY.txt",
                        "No files were added. "
                        "The provided paths may be empty directories or not exist.")

    buf.seek(0)
    return buf.getvalue()

# --- (B) 生成 JSONL（每行一个 JSON，附带文件名元数据） ---
def _build_jsonl(paths:list) -> bytes:
    lines = []
    for p in paths:
        try:
            txt = open(p, "r", encoding="utf-8").read()
            obj = json.loads(txt) if txt.strip() else {}
        except Exception:
            obj = {"_error": f"failed to load {p}"}
        # 附加元数据便于回溯
        obj_meta = {
            "_file": os.path.basename(p),
            "_saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
        }
        # 如果是 dict，合并；否则包装
        if isinstance(obj, dict):
            obj.update(obj_meta)
            lines.append(json.dumps(obj, ensure_ascii=False))
        else:
            lines.append(json.dumps({"_content": obj, **obj_meta}, ensure_ascii=False))
    jsonl = "\n".join(lines)
    return jsonl.encode("utf-8")

# ---------- Sidebar ----------
# --- logo on top ---
st.sidebar.image("./ui/logo.svg", use_container_width=True)
if not os.environ.get("OPENAI_API_KEY"):
    with st.sidebar:
        api_key = st.text_input(
            "🔑 OpenAI API Key",
            type="password",
            help="Get from: https://platform.openai.com/api-keys"
        )
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.rerun()
        else:
            st.info("👆 Enter your API key to start")

models = load_model_list()
model = st.sidebar.selectbox("Model", models or ["gpt-5-chat-latest"], index=0)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
if st.sidebar.button("🧹 Clean stale sessions now"):
    gc_stale_sessions(max_idle_minutes=0.1)   # 立刻清理短时间内所有无心跳或过期会话
    st.sidebar.success("Cleaned.")
if st.sidebar.button('Back to Core', type='primary', use_container_width=True):
    st.session_state['page']='core'
    st.session_state['current_agent']=None

# --- Download history files ---
with st.sidebar.expander("📦 Download Area", expanded=False):
    # 1) 列出本 session 的历史结论文件（相对文件名）
    file_names = list_history_files(st.session_state)  # e.g. ["2025-09-05T10-22-31_core.json", "T2_....json", ...]
    name2path = history_name_to_path(st.session_state)
    options = sorted(name2path.keys())   # 纯文件名，比如 2025-09-22_1512.json
    if not file_names:
        st.caption("No conclusion JSON files yet.")
    else:
        st.write(f"Found **{len(file_names)}** files.")
        # 可选：允许用户挑子集；默认全选
        selected = st.multiselect(
            "Select files to include",
            options=options,
            default=options, #默认全选，全不选为None
            help="Choose which conclusion JSON files to include in the archive."
        )

        if selected:
            # 2) 解析为绝对路径
            abs_paths = resolve_history_paths(selected, st.session_state)

            # # 预检 1：确认确实存在
            # for p in abs_paths:
            #     pp = Path(p)
            #     st.write("exists=", pp.exists(), "is_file=", pp.is_file(), "path=", str(pp))

            # # 预检 2：zip 里预览条目名
            # zip_bytes = _build_zip(abs_paths)  # 你的 _build_zip 修正版
            # import zipfile, io
            # with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            #     st.write("zip entries (preview):", zf.namelist()[:50])
            
            sid = get_session_id(st.session_state)  # 用于命名
            zip_bytes   = _build_zip(abs_paths)
            jsonl_bytes = _build_jsonl(abs_paths)

            # c1, c2 = st.columns(2)
            # with c1:
            st.download_button(
                label="⬇️ Download ZIP",
                data=zip_bytes,
                file_name=f"{sid}_conclusions.zip",
                mime="application/zip",
                use_container_width=True
            )
            # with c2:
            #     st.download_button(
            #         label="⬇️ JSONL",
            #         data=jsonl_bytes,
            #         file_name=f"{sid}_conclusions.jsonl",
            #         mime="application/json",
            #         use_container_width=True
            #     )

            # 可选：展示将被打包的文件清单
            # with st.expander("Show file list"):
            #     for fn in selected:
            #         st.write("• ", fn)


# with st.sidebar.expander("🗂️ Development process records", expanded=True):
#     logs = st.session_state.get('proc_log', [])
#     if not logs:
#         st.caption("No records yet")
#     else:
#         for i, rec in enumerate(logs):
#             if rec['type']=='core':
#                 st.write(f"{i+1}. [Core] {rec.get('ts','')}")
#                 if rec.get('suggest'):
#                     st.caption("Suggested call: " + ", ".join(rec['suggest']))
#             else:
#                 st.write(f"{i+1}. [{rec.get('agent')}] {rec.get('ts','')}")
#                 if rec.get('saved_to'):
#                     st.caption(f"Output saved: {rec['saved_to']}")

# ============ 替换原来的 expander 部分 ============
# 漂亮的标题容器
st.sidebar.markdown("""
<div style='text-align: center; font-weight: 700; font-size: 16px; 
            padding: 12px 0; background: linear-gradient(135deg, #e0f2fe 0%, #7dd3fc 100%);
            color: #0369a1; border-radius: 10px; margin: 15px 0; 
            box-shadow: 0 4px 12px rgba(125, 211, 252, 0.2);
            border: 1px solid #bae6fd;'>
    🗂️ Development Records
</div>
""", unsafe_allow_html=True)

# 内容区域
with st.sidebar.container():
    logs = st.session_state.get('proc_log', [])
    if not logs:
        st.caption("📝 No records yet")
    else:
        for i, rec in enumerate(logs):
            # 为每条记录添加卡片样式
            bg_color = "#f0f5ff" if rec['type'] == 'core' else "#f0f9ff"
            border_color = "#2563eb" if rec['type'] == 'core' else "#0891b2"
            emoji = "🧠" if rec['type'] == 'core' else "🤖"
            
            if rec['type']=='core':
                # st.write(f"{i+1}. [Core] {rec.get('ts','')}")
                st.markdown(f"""
                    <div style='background: {bg_color}; padding: 12px; border-radius: 10px; 
                                border-left: 5px solid {border_color}; margin: 10px 0;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <b>{emoji} {i+1}. [Core] {rec.get('ts', '')}</b>
                    </div>
                    """, unsafe_allow_html=True)
                if rec.get('saved_to'):
                    st.caption(f"💾 Conclusion Saved: {os.path.basename(rec['saved_to'])}")
                if rec.get('suggest'):
                    st.caption("🔄 Suggested: " + ", ".join(rec['suggest']))
            else:
                # st.write(f"{i+1}. [{rec.get('agent')}] {rec.get('ts','')}")
                st.markdown(f"""
                    <div style='background: {bg_color}; padding: 12px; border-radius: 10px; 
                                border-left: 5px solid {border_color}; margin: 10px 0;
                                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <b>{emoji} {i+1}. [{rec.get('agent')}] {rec.get('ts', '')}</b>
                    </div>
                    """, unsafe_allow_html=True)
                if rec.get('saved_to'):
                    st.caption(f"💾 Conclusion saved: {os.path.basename(rec['saved_to'])}")

            # 添加分隔线（最后一条不添加）
            if i < len(logs) - 1:
                st.markdown("<hr style='margin: 8px 0; opacity: 0.3;'>", unsafe_allow_html=True)
# ============ 替换结束 ============

# ---------- Helpers ----------
def _render_user_msg(m, avatar):
    with st.chat_message('user', avatar=avatar):
        st.write(m.get('content',''))
        rb = m.get('rag_brief')
        if rb:
            j = rb.get('json_files') or []
            u = rb.get('upload_files') or []
            w = rb.get('urls') or []
            st.caption("RAG of this round："
                       + (f" JSON:{', '.join(j)} " if j else "")
                       + (f" Upload:{', '.join(u)} " if u else "")
                       + (f" URL:{', '.join(w)} " if w else ""))

def _render_assistant_msg(m, avatar):
    with st.chat_message(m.get('role', 'assistant'), avatar=avatar):
        content = m.get('content', '') or ''
        render_text_with_inline_math(content)
        rag_log = m.get('rag_log') or {}
        if rag_log:
            with st.expander("RAG process", expanded=False):
                split_chunks = rag_log.get('split_chunks', 0)
                retrieval = rag_log.get('retrieval') or {}
                sources = rag_log.get('sources') or {}
                st.markdown("**Uploaded data locally**")
                st.markdown(f"- Split Chunks:**{split_chunks}**")
                st.markdown(f"- Search Hits:**{retrieval.get('hits',0)} / k={retrieval.get('k',8)}；预算={retrieval.get('budget_tokens',0)}**")
                j = sources.get('json_files') or []
                u = sources.get('upload_files') or []
                w = sources.get('urls') or []
                if j: st.markdown(f"- Historical JSON：{', '.join(j)}")
                if u: st.markdown(f"- Uploaded files：{', '.join(u)}")
                if w: st.markdown(f"- URLs：{', '.join(w)}")
                detail = retrieval.get('hits_detail') or []
                if detail:
                    st.markdown("**Hit details (first 60 characters preview)**")
                    for d in detail:
                        src = f"{d.get('source_type','')}/{d.get('source_name','')}".strip('/')
                        st.write(f"• [{d.get('mode')}] rank {d.get('rank')} | chunk {d.get('chunk_id')} | {d.get('chars')} chars | {src}")
                        st.caption(d.get('preview',''))

                # —— 显示“最终布尔查询”（由 Query Refiner 合成）——
                src = rag_log.get('sources', {})  # 你上文已有 src 定义就复用
                qd  = src.get('query_refiner_debug') or {}
                fq = src.get('final_booleanq') or ''
                reason = qd.get('reason') or '(noe)'
                st.markdown(f"**QueryRefiner path:** {reason}")
                st.markdown(f"**Angent name:** {st.session_state['current_agent']}")

                # 最终布尔查询（来自 merged）
                merged = qd.get('merged') or {}
                boolean_q = merged.get('boolean')
                if boolean_q:
                    st.markdown("**Final Boolean Query (Query Refiner)**")
                    st.code(fq, language='text')

                if qd.get('rank_query'):
                    st.markdown("**Query for similarity calculation (rank_query)**")
                    st.code(qd['rank_query'])

                # 可选：查看 LLM 原始输出
                if qd.get('llm_raw'):
                    with st.expander('Display the output of LLM', expanded=False):
                        st.code(qd['llm_raw'])
                # if qd:
                #     st.caption(f"QueryRefiner 路径：{qd.get('reason','')}")
                #     if st.checkbox("显示 LLM 原始输出", value=False, key="show_llm_raw"):
                #         st.code(qd.get("llm_raw",""))

                # —— 显示“文献相关性排序 Top-K（TF-IDF 余弦）”——
                rk = src.get('rank_debug') or []
                if rk:
                    st.markdown("**Literature relevance ranking (Top-K)**")
                    for i, it in enumerate(rk, 1):
                        score = it.get("score", 0.0)
                        title = it.get("title", "") or "(untitled)"
                        doi   = (it.get("doi") or "").strip()
                        link  = f"http://doi.org/{doi}" if doi else ""
                        line  = f"{i}. score={score:.3f} · {title}"
                        if link:
                            line += f"  \n{link}"
                        st.markdown(line)
            # ol = rag_log.get('sources',{}).get('online_libraries') or []
            # if ol:
            #     st.markdown("**检索的文献列表**")
            #     for it in ol:
            #         _url = it.get("doi") and f"http://doi.org/{it['doi']}" or (it.get("url") or "")
            #         line = f"- {it.get('title','(untitled)')} ({it.get('year','')}) · {it.get('journal','')}"
            #         if _url: line += f"  \n  {_url}"
            #         st.markdown(line)

def _persist_uploads(uploads):
    paths=[]
    if uploads:
        tmpd = tempfile.mkdtemp(prefix='rag_up_')
        for uf in uploads:
            p = os.path.join(tmpd, uf.name)
            with open(p, 'wb') as w: w.write(uf.getbuffer())
            paths.append(p)
    return paths

def _now(): return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

# ========= 工具：确保有用户画像（领域/背景/目标） =========
def _ensure_user_profile():
    up = st.session_state.setdefault('user_profile', {})
    up.setdefault('domain', '')
    up.setdefault('background', '')
    up.setdefault('objectives', '')
    return up

# 1.1 规范化读取 variables_changed（可能是 bool/str/number）
def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        return v in ("true", "1", "yes", "y", "是", "有", "需要")
    return None  # 无法判断 -> None

# 实现LLM若干轮的记忆功能
def _compress(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    t = text.strip()
    return t if len(t) <= max_chars else (t[:max_chars] + " …[truncated]")

def build_history_msgs(chat_list: list, max_pairs: int = 6, max_chars_each: int = 1200) -> list:
    """
    将当前 Agent 的历史对话转换为 Chat Completions API 所需的 messages 片段。
    只拿“最近 max_pairs 轮”的 (user, assistant) 配对，避免爆 token。
    """
    # 收集成 (user, assistant) 对
    pairs = []
    u_buf, a_buf = None, None
    for m in chat_list:
        if m.get("role") == "user":
            u_buf = m.get("content", "")
        elif m.get("role") == "assistant":
            # 跳过占位回答
            if m.get("pending"):  
                continue
            a_buf = m.get("content", "")
        # 凑成一对就推入
        if u_buf is not None and a_buf is not None:
            pairs.append((u_buf, a_buf))
            u_buf, a_buf = None, None

    # 只取最近 N 轮
    pairs = pairs[-max_pairs:]

    # 展开为 messages
    msgs = []
    for u, a in pairs:
        if u:
            msgs.append({"role": "user", "content": _compress(u, max_chars_each)})
        if a:
            msgs.append({"role": "assistant", "content": _compress(a, max_chars_each)})
    return msgs

# ========= 工具：检测用户是否主动点名 agent，并提取粗略目的 =========
AGENT_ALIASES = {
    "Literature Searcher": ["literature", "papers", "patent", "review", "reference",
                            "文献", "论文", "专利", "综述", "参考"],
    "ChemProcess Modeler": ["model", "mechanism", "kinetic", "equilibrium", "mass transfer",
                            "模型", "机理", "动力学", "平衡", "传质"],
    "Experiment Designer": ["DOE", "design of experiments", "response surface", "orthogonal", "taguchi",
                            "实验", "试验设计", "响应面", "正交", "田口"],
    "Fitting Wizard": ["fit", "least squares", "regression", "parameter estimation", "curve fitting",
                       "拟合", "回归", "最小二乘", "参数估计"],
    "Optimization Navigator": ["optimize", "optimization", "multi-objective", "pareto", "nsga", "trade-off",
                               "优化", "多目标", "帕累托", "权衡"],
    "Process Analyzer": ["flowsheet", "material balance", "energy balance", "bottleneck", "pinch", "integration",
                         "流程", "流程图", "物料衡算", "能量衡算", "瓶颈", "夹点", "集成", "系统分析"]
}

def detect_agent_request(text: str):
    """返回 {'agents': [...], 'purpose': '...'}；purpose 先用用户原话粗提炼（20字/15词内）"""
    if not text: 
        return {"agents": [], "purpose": ""}
    t = text.lower()
    # trigger 动词
    trig = any(k in t for k in ["use ", "call ", "invoke ", "open ", "switch to ", "start ",
                                "调用", "使用", "启用", "打开", "进入", "切到", "启动"])
    agents = []
    for name, keys in AGENT_ALIASES.items():
        if any(k in t for k in [k.lower() for k in keys] + [name.lower()]):
            agents.append(name)
    agents = list(dict.fromkeys(agents))[:3]
    # 粗略“目的”：截取用户最后一句或关键词后的一小段
    purpose = ""
    try:
        s = text.strip().replace("\n", " ")
        # 取最后一句
        import re
        segs = re.split(r"[。！？!?\.]", s)
        purpose = segs[-2].strip() if len(segs) >= 2 and segs[-1]=="" else segs[-1].strip()
        if len(purpose) > 80:
            purpose = purpose[:80] + "…"
    except Exception:
        purpose = ""
    if not trig:
        # 没触发词则认为不是“显式请求”
        agents = []
    return {"agents": agents, "purpose": purpose}


def _weak_heuristic_agents(text: str):
    if not text: 
        return []
    t = text.lower()

    def hit(keys): 
        return any(k in t for k in keys)

    cands = []
    if hit(["文献","论文","专利","综述","参考","citation","literature","paper","patent","review","reference","bibliography"]):
        cands.append('Literature Searcher')
    if hit(["模型","机理","动力学","平衡","传质","速率","shrinking core","kinetic","mechanism","equilibrium","mass transfer","rate","model"]):
        cands.append('ChemProcess Modeler')
    if hit(["实验","设计","试验","正交","响应面","田口","doe","design of experiments","response surface","orthogonal","box-behnken","taguchi"]):
        cands.append('Experiment Designer')
    if hit(["拟合","回归","最小二乘","参数估计","曲线拟合","fit","regression","least squares","parameter estimation","curve fitting"]):
        cands.append('Fitting Wizard')
    if hit(["优化","多目标","帕累托","权衡","nsga","pareto","trade-off","optimize","optimization","genetic"]):
        cands.append('Optimization Navigator')
    if hit(["流程","流程图","物料衡算","能量衡算","瓶颈","夹点","集成","系统分析",
            "flowsheet","material balance","energy balance","bottleneck","pinch","integration","process analysis"]):
        cands.append('Process Analyzer')

    # 如果 Core 回复里出现“下一步/建议/推荐/接下来/需要”也可弱触发
    if hit(["下一步","建议","推荐","接下来","需要","next step","recommend","suggest","proceed"]):
        pass  # 已被规则命中时优先展示
    return list(dict.fromkeys(cands))[:3]

def rag_for_this_turn(default_selected:str='', scope_key:str='core', turn:int=0):
    with st.expander("RAG (this question)", expanded=bool(default_selected)):
        enable = st.checkbox("Enable RAG", value=bool(default_selected), key=f"rag_enable_{scope_key}_{turn}")
        selected=[]; selected_paths=[]; urls=[]; uploads=[]; k=8; budget=2000
        if enable:
            existing = [fp.name for fp in list_history_files(st_session_state=st.session_state)]
            pre=[default_selected] if (default_selected and default_selected in existing) else []
            selected = st.multiselect("Select historical Agent output JSON", existing, default=pre, key=f"rag_sel_{scope_key}_{turn}")
            # 把文件名 -> 绝对路径（字符串）
            selected_paths = resolve_history_paths(selected, st_session_state=st.session_state)
            urls_text = st.text_area("URL (one per line)", height=60, placeholder="http(s)://...", key=f"rag_urls_{scope_key}_{turn}")
            urls = [u.strip() for u in (urls_text.splitlines() if urls_text else []) if u.strip()]
            uploads = st.file_uploader("Or upload documents (pdf/docx/md/txt/json/zip)", type=['pdf','docx','md','txt', 'json', 'zip'], accept_multiple_files=True, key=f"rag_up_{scope_key}_{turn}")
            c1, c2 = st.columns(2)
            with c1: k = st.slider("Search top-k", 2, 20, 8, 1, key=f"rag_k_{scope_key}_{turn}")
            with c2: budget = st.slider("RAG Token budget", 500, 4000, 2000, 100, key=f"rag_budget_{scope_key}_{turn}")
        return enable, selected_paths, urls, uploads, k, budget


# ---------- Core Page ----------
def core_page():
    st.markdown("""### <span class="gem-pill">🧠 Core Agent (Main dialogue)</span>""", unsafe_allow_html=True)
    chat = st.session_state['chat']['core']
    for m in chat:
        if m['role']=='user': _render_user_msg(m, "🙂")
        else: _render_assistant_msg(m, AV_CORE)
        
    # 历史消息渲染后，如果没有任何对话，展示欢迎/引导 + 采集用户画像
    if len(chat) == 0:
        up = _ensure_user_profile()
        st.markdown("""
    <div style="padding:14px 16px;border-radius:14px;
        background:linear-gradient(135deg,#f0f5ff 0%,#eef7ff 40%,#ffffff 100%);
        border:1px solid #e5e7eb;">
    <div style="font-weight:700;font-size:18px;margin-bottom:6px;">
        👋 Welcome to <span style="color:#2563eb">Core Agent</span>!
    </div>
    <div style="color:#334155;line-height:1.6;">
        This is a <b>LLM-assisted chemical process development</b> workstation. You can use the "five-part prompt"
        to describe tasks. Core will analyze your tasks and determine whether to invoke different specialized agents to help you in the main dialogue.
    </div>
    </div>
    """, unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown("#### 📝 Opening Configuration (Affects Subsequent Roles and Recommendations)")
            c1, c2 = st.columns([1,1])
            with c1:
                up_domain = st.text_input("Your Process Domain/Object (Required)", 
                                        value=up.get('domain',''), 
                                        placeholder="Example: LFP Recycling / NCM Wet Recycling / Saline Water Treatment / Dyeing Wastewater / ...")
            with c2:
                # up_objectives = st.text_input("Project Objectives (Optional)", 
                #                             value=up.get('objectives',''),
                #                             placeholder="Example: High Purity/High Yield Li2CO3 -> 99.5% / 95% …")
                up_background = st.text_input("Background Information (Optional)", 
                                            value=up.get('background',''),
                                            # height=90,
                                            placeholder="Project stage, existing data/equipment constraints, environmental/cost boundaries, etc.")

            # 第二行：Core Role + 保存按钮
            c3, c4 = st.columns([6,1])   # 左边宽一些给文本框
            with c3:
                core_role_ui = st.text_area(
                    "Core Agent Role (editable)", 
                    value=st.session_state.get('core_role_text', CORE_ROLE_DEFAULT),
                    height=120,
                    key="core_role_ui"
                )
            with c4:
                    st.markdown(
                        """
                        <style>
                        div[data-testid="stVerticalBlock"] > div:nth-child(1) {
                            display: flex;
                            flex-direction: column;
                            justify-content: center; /* 垂直居中 */
                            height: 100%;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True
                    )
                    if st.button("Save Opening Configuration", type="primary"):
                        st.session_state['user_profile'] = {
                            "domain": up_domain.strip(),
                            "background": up_background.strip()
                        }
                        st.session_state['core_role_text'] = core_role_ui.strip()
                        st.session_state['core_role_sent'] = False
                        st.success("Configuration & Core Role saved.")
        
            
            
            
            # # 先留两列空列，中间列宽度固定，让按钮居中
            # left, center, right = st.columns([1.5, 1, 1.5])   # 比例可自己调

            # with center:
            #     if st.button("Save Opening Configuration", type="primary", use_container_width=True):
            #         st.session_state['user_profile'] = {
            #             "domain": up_domain.strip(),
            #             # "objectives": up_objectives.strip(),
            #             "background": up_background.strip()
            #         }
            #         st.success("Configuration saved.")
        st.caption("Tip: You can now enter your questions below. You can also directly enter: Task/Examples/CoT/Question/Output format in five-part format.")

    if st.session_state['pending_core']:
        with stylable_container(
        key="danger_cancel_core",
        css_styles="""
        div[data-testid="stButton"] > button {
            background: #ef4444 !important;              /* 用 background 覆盖 background-image */
            background-image: none !important;           /* 防主题渐变/灰覆盖 */
            color: #fff !important;
            border: 1px solid #dc2626 !important;
            box-shadow: 0 2px 10px rgba(239,68,68,.25) !important;
            filter: none !important;                     /* 关掉可能的滤镜 */
            transition: background-color .15s ease, box-shadow .15s ease, transform .15s ease;
        }

        /* Hover / Focus / Active（红 600）——把所有可能态都重写，防发灰 */
        div[data-testid="stButton"] > button:not(:disabled):hover,
        div[data-testid="stButton"] > button:not(:disabled):focus,
        div[data-testid="stButton"] > button:not(:disabled):focus-visible,
        div[data-testid="stButton"] > button:not(:disabled):active {
            background: #dc2626 !important;              /* 更深一档而不是灰 */
            background-image: none !important;
            color: #fff !important;
            border-color: #b91c1c !important;            /* 边框也跟着加深 */
            box-shadow: 0 3px 12px rgba(220,38,38,.28) !important;
            filter: none !important;                     /* 禁止任何灰化滤镜 */
            transform: translateY(-1px);
            outline: none !important;
        }
        """
        ):
            if st.button("Cancel Pending Processing", key="cancel_pending"):
                st.session_state['pending_core'] = None
                chat.append({'role':'assistant','content':'(This round of requests has been cancelled)'})
                st.rerun()

    placeholder = "Natural language input; for structured input, please use five-part tags:\nTask: ...\nExamples: ...\nChain of thought: ...\nQuestion: ...\nOutput format: ..."
    user_text = st.chat_input(placeholder)

    rag_enable, rag_sel, rag_urls, rag_uploads, rag_k, rag_budget = rag_for_this_turn(
        default_selected=st.session_state['core_auto_rag'], scope_key='core', turn=st.session_state['turn_core'])

    # Core agent 的调用建议
    if st.session_state['core_suggest']:
        sg = st.session_state['core_suggest']
        agents = sg.get('agents') or []
        up = _ensure_user_profile()
        purpose = (sg.get('purpose') or '').strip()
        reason = (sg.get('reason') or '').strip()
        with st.container(border=True):
            st.markdown("**Core Suggestion:** It may be necessary to call a specific working agent to continue.")
            # 1) Let the user select an already "recommended" agent (these were obtained through LLM/heuristics in the previous step)
            pick = st.selectbox("Select the agent to switch to", agents, index=0 if agents else None)

            # 2) Allow the user to fine-tune the "call purpose" before switching (default fills in the core hint)
            purpose_edit = st.text_area(
                "Call Purpose (Editable)",
                value=purpose + f'(refering to {reason})',
                height=80,
                help="This text will be used to generate the role description for the dedicated Agent. You can leave it blank, and the system will automatically extract it from the hint."
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Switch to Selected Agent", type="primary", use_container_width=True, key="go_suggested_agent"):
                    # 1) 先根据 Core 的建议生成/更新 role_text
                    try:
                        # 只把“用户画像 + hint/编辑后的目的 + agent 名”传给 role 生成器
                        role_text = generate_role(
                            agent=pick,
                            model=model,
                            temperature=temperature,
                            user_profile=up,
                            purpose=purpose_edit
                        )
                        if pick.lower().startswith("chemprocess"):
                            role_text += "After answering, explicitly state whether new variables are defined or existing variables are modified for equations/mass balances. In the final JSON, set `variables_changed` to true or false accordingly; explicitly state whether variables are used in reply. In the final JSON, set `variables_used` to true or false accordingly."
                        else:
                            role_text=role_text
                    except Exception:
                        domain = (up or {}).get('domain') or "your process"
                        role_text = f"{pick}: specialist agent for {domain}"

                    # 2) 保存 role 到会话状态
                    st.session_state['agent_roles'][pick] = role_text

                    # 3) 仅基于 role_text + agent_specs 的先验 生成“默认 schema”，并缓存
                    try:
                        # 先验：来自 agent_specs（json_schema 优先，其次 outfmt）
                        try:
                            from agents.agent_specs import AGENTS as _SPEC_FOR_SCHEMA
                            _spec = next((a for a in _SPEC_FOR_SCHEMA if a['name'] == pick), {})
                            _prior_schema = _spec.get('json_schema') or {}
                            if isinstance(_prior_schema, dict):
                                _prior_schema = json.dumps(_prior_schema, ensure_ascii=False, indent=2)
                        except Exception:
                            _prior_schema = "{}"

                        if _prior_schema == "{}":
                            # 生成/加载（命中缓存就不调 LLM；无缓存且库缺失时，用 role_text 真调 LLM 生成首版 shape）
                            shape_text, draft = load_or_propose_schema(
                                agent_name=pick,
                                role_text=role_text,         # 仅 role_text
                                llm_model=model,
                                llm_temperature=temperature
                            )
                            st.session_state['agent_schema'][pick] = shape_text
                        else:
                            st.session_state['agent_schema'][pick] = _prior_schema

                    except Exception as e:
                        st.warning(f"Auto-generate default schema failed: {e}")

                    # 4) 跳转到 Agent 页
                    st.session_state['page'] = 'agent'
                    st.session_state['current_agent'] = pick

                    from core.storage import latest_entry_path
                    st.session_state['core_auto_rag'] = latest_entry_path('Core agent')

                    # 用完这次建议就清掉，避免反复弹窗
                    st.session_state['core_suggest'] = None
                    st.rerun()
            with c2:
                if st.button("Ignore Suggestion", type='secondary', use_container_width=True, key="ignore_suggested_agent"):
                    st.session_state['core_suggest'] = None
                    st.rerun()

    # Core agent pending 请求
    if st.session_state['pending_core']:
        job = st.session_state['pending_core']; cchat = st.session_state['chat']['core']; parts = job['parts']
        history_msgs = build_history_msgs(cchat, max_pairs=6, max_chars_each=1200)
        upload_paths = job.get('rag_upload_paths') or []
        rag_ctx, rag_log = build_rag_ctx_and_log(
            enable=job['rag_enable'], selected=job['rag_sel'], urls=job['rag_urls'],
            upload_paths=upload_paths, query_text=parts['question'],
            k=job.get('rag_k',8), budget_tokens=job.get('rag_budget',2000), model_name=job['model']
        )
        core_role = st.session_state.get('core_role_text', CORE_ROLE_DEFAULT)
        # core_role = """As the central coordinator in chemical process development, my primary role is strategic decomposition and sequential orchestration of complex tasks. I focus on understanding user requirements, breaking down problems into logical sub-tasks, and progressively executing them through specialized agents.
        #             My core responsibilities include: 1.Analyzing project background and clarifying objectives through dialog with users; 2. Decomposing complex process challenges into sequential, executable sub-tasks; 3.Recommending appropriate specialized agents from existing toolkit (Literature Searcher, ChemProcess Modeler, Experiment Designer, Fitting Wizard, Optimization Navigator, Process Analyzer) for each sub-task; 4.Suggesting new agent types when existing capabilities cannot address specific requirements;
        #             5.Implementing sequential workflow: plan → execute (by specialized agents) → evaluate (based on JSON conclusions) → proceed; 6.Making dynamic decisions based on intermediate results provided through JSON conclusion files from specialized agent interactions; 7.Explicitly prompting users to upload critical conclusion JSON files after completing dialogues with specialized agents; 8.Maintaining macro-level communication to discuss strategy, evaluate results, and determine next steps; 9.Responding in the same language as the user's prompt, using English as the default when the language is uncertain
        #             I systematically recommend agent deployment with clear rationale: "To address [current subtask], I recommend engaging the [Agent Name]. After completing this dialogue, please upload the conclusion JSON file for my evaluation before we proceed to the next phase." My communication style adapts to the user's language preference while maintaining professional clarity. My focus remains on strategic planning, result assessment, and coordinated progression. Note that if the called agents from existing toolkit, their names must be exactly the same as the corresponding agent names when written into the json file."""
        
        if not st.session_state.get('core_role_sent', False):
            # 首轮：带 Core 角色
            result = run_with_role(
                role_text=core_role, model=job['model'], temperature=job['temperature'],
                task_spec=parts['task'], examples=parts['examples'], cot=parts['cot'],
                question=parts['question'], output_format=parts['outfmt'],
                json_schema='{"stage":"string","decisions":[],"called_agents":[],"notes":""}',
                rag_snippets=rag_ctx,
                history_msgs=history_msgs
            )
            st.session_state['core_role_sent'] = True
        else:
             # 后续：不带 Core 角色
            result = run_no_role(
                model=job['model'], temperature=job['temperature'],
                task_spec=parts['task'], examples=parts['examples'], cot=parts['cot'],
                question=parts['question'], output_format=parts['outfmt'],
                json_schema='{"stage":"string","decisions":[],"called_agents":[],"notes":""}',
                rag_snippets=rag_ctx,
                history_msgs=history_msgs
            )

        # === 新增：保存历史 JSON（会话内），并把路径写回 saved_to ===
        try:
            cconc = result.get("conclusion", {})
            cfp = None
            if isinstance(cconc, dict) and cconc:
                cfp = save_history_json(st.session_state, 'Core agent', cconc)
                # 把 chat 里该条的 saved_to 更新为刚刚写入的历史路径
                # chat[-1]['saved_to'] = str(fp)
        except Exception:
            pass

        for i in range(len(cchat)-1, -1, -1):
            if cchat[i].get('pending'):
                cchat[i] = {'role':'assistant','content':result['reply'], 'conclusion':cconc,'saved_to':str(cfp) or '', 'rag_log': rag_log}
                break

        reply_text = result.get('reply','').strip()
        agents = []
        if isinstance(result.get('conclusion'), dict):
            agents = (result['conclusion'].get('called_agents') or [])[:3]

        if not agents:
            # ⬇️ 联合抽取：返回 dict，包含 agent_name 和 purpose
            js = extract_agent_and_purpose(reply_text, st.session_state.get('user_profile',{}),
                                        model=job['model'], temperature=0.2)
            inferred = js.get("agent_name", "")
            purpose = js.get("purpose", "")
            
            if inferred:
                agents = [inferred]
                # ✅ 保存 purpose 信息，供 generate_role 或 UI 使用
                st.session_state['core_suggest'] = {
                    'agents': agents,
                    'reason': 'core reply infer',
                    # 'hint': reply_text[-800:],   # 兼容性保留
                    'purpose': purpose
                }
            else:
                agents = _weak_heuristic_agents(reply_text)

        # —— 关键：把 Core 的回复当作 hint 传给后续 generate_role —— #
        if agents:
            # truncated_reply = reply_text[-800:] if len(reply_text) > 800 else reply_text
            purpose = _extract_purpose(reply_text, model=model, temperature=0.2)
            st.session_state['core_suggest'] = {
                'agents': agents,
                'reason': 'core reply',
                # 'hint': truncated_reply,      # 兼容保留，也可以不再显示
                'purpose': purpose             # ✅ 提前提取
            }

        # if agents:
        #     st.session_state['core_suggest'] = {
        #         'agents': agents,
        #         'reason': 'core_reply',
        #         'hint': reply_text        # 这就是联合抽取/生成 role 的 hint
        #     }
        # st.rerun()

        st.session_state['proc_log'].append({
            'type':'core',
            'ts': time.strftime("%H:%M:%S"),
            'saved_to': str(cfp) or '',
            'suggest': agents or None
        })

        st.session_state['pending_core'] = None


        st.rerun()
    
    # 新增：让用户选择“仅下一次调用时应用 Core 角色”
    # cols = st.columns([1,2,2])
    # with cols[0]:
    if st.button("Apply Core Role to next call", key="apply_core_role_once", type="secondary", use_container_width=True):
        st.session_state['core_role_sent'] = False  # 让下一次调用 run_with_role
        st.success("Core role will be applied on the next Core call only.")


    # 提交输入
    if user_text is not None and user_text.strip():
        if rag_enable:
            upload_paths = _persist_uploads(rag_uploads)
            rag_brief = {"json_files":[os.path.basename(p) for p in rag_sel],
                         "upload_files":[os.path.basename(p) for p in upload_paths],
                         "urls": rag_urls}
        else:
            upload_paths=[]; rag_brief=None
        chat.append({'role':'user','content':user_text, 'rag_brief': rag_brief})
        parts = parse_five_parts(user_text)
        chat.append({'role':'assistant','content':'(Answering…)', 'pending': True})
        req = detect_agent_request(user_text)
        if req['agents']:
            st.session_state['core_suggest'] = {
                'agents': req['agents'],
                'reason': 'user request',
                'hint': user_text,            # 作为 purpose 提炼提示
            }
        st.session_state['pending_core'] = {'parts': parts,'rag_enable': rag_enable,'rag_sel': rag_sel,'rag_urls': rag_urls,
                                            'rag_upload_paths': upload_paths,'rag_k': rag_k,'rag_budget': rag_budget,
                                            'model': model,'temperature': temperature}
        st.session_state['turn_core'] += 1
        st.rerun()

    ok, info = save_session(st.session_state)
    if not ok:
        # 可选：打印到日志或页面
        st.sidebar.warning(f"save_session 失败：{info}")


# ---------- Agent Page ----------
def agent_page():
    agent_name = st.session_state['current_agent']
    st.markdown(f'### <span class="gem-pill-sub">{AVMAP.get(agent_name,"🤖")} {agent_name}（Branch dialogue）</span>', unsafe_allow_html=True)
    chat_agents = st.session_state['chat']['agents']
    chat = chat_agents.setdefault(agent_name, [])
    history_msgs = build_history_msgs(chat, max_pairs=6, max_chars_each=1200)
    role_default = st.session_state['agent_roles'].get(agent_name, f'{agent_name}: specialist agent')
    schema_default = st.session_state['agent_schema'].get(agent_name, '{}')
    # 渲染彩色胶囊标题
    st.markdown('<span class="pill pill-role">Agent Role Setting (can be modified)</span>', unsafe_allow_html=True)

    # 渲染输入框，label设为空并禁止显示
    role_text = st.text_area(
        "Agent Role Setting",            # 非空 label（供无障碍/屏读用）
        value=role_default,
        height=80,
        label_visibility="collapsed",  # 折叠（隐藏视觉）
        key="agent_role_text"
    )

     # 渲染彩色胶囊标题
    st.markdown('<span class="pill pill-role">Agent Conclusion Schema Setting (can be modified)</span>', unsafe_allow_html=True)

    # === Schema 内联编辑（就放在 role_text 下方）===
    schema_text = st.text_area(
        "Conclusion JSON Schema (editable, used next call)",
        value=schema_default,
        height=80,
        key=f"{agent_name}_schema_text"
    )

    if st.button("💾 Save Schema", key=f"{agent_name}_save_schema"):
        try:
            _shape_obj = json.loads(schema_text)
            save_schema(agent_name, _shape_obj)  # 写入 schemas/custom/<Agent>.schema.json
            st.success("Schema saved. It will be used in next call.")
        except Exception as e:
            st.error(f"Schema not valid JSON: {e}")


    av = AVMAP.get(agent_name, "🤖")
    for m in chat:
        if m['role']=='user': _render_user_msg(m, "🙂")
        else: _render_assistant_msg(m, av)

    if st.session_state['pending_agent'] and st.session_state['pending_agent']['agent_name'] == agent_name:
        with stylable_container(
        key="danger_cancel_core",
        css_styles="""
        div[data-testid="stButton"] > button {
            background: #ef4444 !important;              /* 用 background 覆盖 background-image */
            background-image: none !important;           /* 防主题渐变/灰覆盖 */
            color: #fff !important;
            border: 1px solid #dc2626 !important;
            box-shadow: 0 2px 10px rgba(239,68,68,.25) !important;
            filter: none !important;                     /* 关掉可能的滤镜 */
            transition: background-color .15s ease, box-shadow .15s ease, transform .15s ease;
        }

        /* Hover / Focus / Active（红 600）——把所有可能态都重写，防发灰 */
        div[data-testid="stButton"] > button:not(:disabled):hover,
        div[data-testid="stButton"] > button:not(:disabled):focus,
        div[data-testid="stButton"] > button:not(:disabled):focus-visible,
        div[data-testid="stButton"] > button:not(:disabled):active {
            background: #dc2626 !important;              /* 更深一档而不是灰 */
            background-image: none !important;
            color: #fff !important;
            border-color: #b91c1c !important;            /* 边框也跟着加深 */
            box-shadow: 0 3px 12px rgba(220,38,38,.28) !important;
            filter: none !important;                     /* 禁止任何灰化滤镜 */
            transform: translateY(-1px);
            outline: none !important;
        }
        """
        ):
            if st.button("Cancel pending", key="cancel_pending"):
                st.session_state['pending_agent'] = None
                chat.append({'role':'assistant','content':'(Canceled this request)'})
                st.rerun()

    # 4) “如有必要显示并提供编辑”：两种情况展示变量表
    #    A) 本轮确实有需要（need_extract=True），不论有无提取结果，都展示表格让用户补充/编辑；
    #    B) 历史上已存在变量表（load_variables 非空），也展示以便随时修订。
    if st.session_state['variables']['needextracted'] and agent_name.lower().startswith("chemprocess"):
        # data = st.session_state['variables']['table'] or []  # 重新读取，确保上面 save 后的一致性
        data = load_variables(st.session_state) or []
        df = pd.DataFrame(data or [], columns=["name","symbol","unit","description","default_value"])
        st.markdown("#### Variable Table")
        # 显示提示
        st.caption("⚠️ Changes made in the table are not applied until you click **Save**.")
        edited = st.data_editor(
            df, use_container_width=True, num_rows="dynamic",
            key=f"{agent_name}_var_editor"
        )
        if st.button("💾 Save Variable Table", key=f"{agent_name}_save_vars"):
            # st.session_state['variables']['table'] = edited.to_dict(orient="records")
            save_variables(edited.to_dict(orient="records"), st.session_state)
            st.success(f"Variables saved to project_state/sessions/{sid}/variables.json. They will be used in the next call.")


    placeholder = "Natural language input; if structured output is needed, please use the five-part label:\nTask specification: ...\nExamples: ...\nChain of thought: ...\nQuestion: ...\nOutput format: ..."
    user_text = st.chat_input(placeholder)
    turn = st.session_state['turn_agents'].get(agent_name, 0)
    rag_enable, rag_sel, rag_urls, rag_uploads, rag_k, rag_budget = rag_for_this_turn(scope_key=agent_name.replace(' ','_'), turn=turn)

    # 处理 pending Agent
    if st.session_state['pending_agent'] and st.session_state['pending_agent']['agent_name'] == agent_name:
        job   = st.session_state['pending_agent']
        parts = job['parts']

        # --- Literature Searcher：自动检索在线文献库 ---
        sources_list = []
        online_snips  = ""
        query_debug   = {}
        dbg = {}
        ranking_debug = []
        search_debug = {}
        no_online_evidence = False
        final_booleanq = ""

        if "literature searcher" in agent_name.lower():
            # 1) 更稳健地拿 query：优先 Question，其次 Task，再退回用户这条原文
            q_raw = (parts.get('question') or parts.get('task') or "").strip()
            if not q_raw:
                # 退回到当前 user 文本（这轮 chat[-2] 通常是用户消息）
                try:
                    last_user = ""
                    for m_ in reversed(chat):
                        if m_['role'] == 'user':
                            last_user = m_.get('content',"")
                            break
                    q_raw = (last_user or "").strip()
                except Exception:
                    q_raw = ""
                st.caption('not literature searcher')

            if q_raw:
                # 2) LLM + 规则 聚合为布尔查询
                must, anyt, boolean_q, dbg = refine_query_with_llm(q_raw, model=job['model'], temperature=0.2)
                st.caption('query is refined')
                query_debug = dbg  # 一定带 reason/llm_raw 等

                if boolean_q:
                    st.caption(boolean_q)
                    # 3) 在线检索（取多一点便于排序）
                    # candidates = unified_search_with_boolean(boolean_q, year_from=2010, year_to=2025, limit=20)
                    # 0) 用你刚实现的 build_boolean_query（支持 mode 参数）
                    #   先严格：must AND (any)；若 0 命中且 anyt 非空 -> 仅 must
                    # strict_q = build_boolean_query(must, anyt, mode="strict")   # must AND (any)
                    must_q   = build_boolean_query(must, anyt, mode="must")     # 仅 must
                    
                    def _search(q, yf=None, yt=None, lim=20):
                        # 默认 yt = 当前年份
                        if yt is None:
                            yt = datetime.now().year
                        
                        # 默认 yf = yt 往前推 15 年
                        if yf is None:
                            yf = yt - 15
                        return unified_search_with_boolean(q, year_from=yf, year_to=yt, limit=lim) if q else []

                    # 1) 严格检索
                    candidates = _search(boolean_q)
                    search_debug["strict"] = {"query": boolean_q, "hits": len(candidates or [])}
                    final_booleanq = boolean_q
                    # 2) 若 0 命中且 anyt 非空：退回 must-only
                    if (not candidates) and anyt:
                        candidates = _search(must_q)
                        st.caption(must_q)
                        final_booleanq = must_q
                        search_debug["fallback_must_only"] = {"query": must_q, "hits": len(candidates or [])}

                    if not candidates:
                        st.caption('online search: no results after search')
                        no_online_evidence = True

                    else:
                        # 4) 摘要相似度排序 & 截断
                        top_k = 8
                        rank_q = (dbg.get("rank_query") or q_raw).strip()   # ← 用 Refiner 给出的英文 rank_query
                        sources_list, ranking_debug = rank_sources_by_relevance(rank_q, candidates, top_k=top_k)
                        if not sources_list:
                            st.caption('search ranked empty')
                            no_online_evidence = True
                        
                        else:
                            # 5) 生成在线片段注入 RAG
                            online_snips = make_online_snippets(sources_list)

                            # Literature Searcher 默认开启 RAG
                            job['rag_enable'] = True
            
            else:
                st.caption('boolean query empty, skip online search')
                no_online_evidence = True

        # ……构造 RAG……
        rag_ctx, rag_log = build_rag_ctx_and_log(
            job['rag_enable'], job['rag_sel'], job['rag_urls'],
            job['rag_upload_paths'], parts.get('question',""),
            k=job.get('rag_k', 8), budget_tokens=job.get('rag_budget', 2000),
            model_name=job['model']
        )
        if online_snips:
            rag_ctx = (rag_ctx + "\n\n" + online_snips).strip()

        # —— 无在线证据的“硬性约束”注入 —— #
        if no_online_evidence:
            # 方式A：在 RAG 前置一个“系统标签”，方便后续模型识别
            rag_ctx = (
                "## ONLINE_SEARCH_STATUS\n"
                "NO_ONLINE_EVIDENCE=TRUE\n"
                "GUIDELINE: Do NOT fabricate citations. If external evidence is required, state clearly that no online sources were found during this session.\n\n"
                + rag_ctx
            )
        
        # —— 合并“在线库/Refiner 调试/排序”到日志 —— #
        rag_log = rag_log or {}
        rag_log.setdefault('sources', {})
        rag_log['sources'].update({
            "online_libraries": [
                {"title": s.get("title",""), "year": s.get("year",""), "journal": s.get("journal",""),
                "doi": s.get("doi",""), "url": s.get("url","")}
                for s in (sources_list or [])
            ],
            "query_refiner_debug": dbg or {"reason":"no_query"},
            "rank_debug": ranking_debug or [],
            "search_debug": search_debug,
            "final_booleanq": final_booleanq,
        })

        # 7) 仅当有 sources_list 才启用括号式引用
        citations_parentheses = (("literature searcher" in agent_name.lower()) and bool(sources_list))

        # —— 合并“在线库/Refiner 调试/排序”到日志 —— #
        # rag_log = rag_log or {}
        # rag_log.setdefault('sources', {})
        # st.caption(agent_name)
        # if "literature searcher" in agent_name.lower() and sources_list:
        #     rag_log['sources']['online_libraries'] = [
        #         {"title":s.get("title",""), "year":s.get("year",""), "journal":s.get("journal",""),
        #         "doi":s.get("doi",""), "url":s.get("url","")} for s in sources_list
        #     ]
        #     rag_log['sources']['query_refiner_debug'] = query_debug or {"reason":"no_query"}
        #     rag_log['sources']['rank_debug'] = ranking_debug or []

        # else:
        #     # 保底结构，保证后续代码使用时不会出错
        #     rag_log['sources'] = {
        #         "online_libraries": [],
        #         "query_refiner_debug": {"reason": "not_literature_agent or no sources list found"},
        #         "rank_debug": []
        #     }
        #     if "literature searcher" in agent_name.lower() and no_online_evidence:
        #         rag_log['sources']['note'] = "No online hits after multi-stage backoff. Model must not fabricate; answer from prior knowledge/local context only."

        # --- 调 LLM ---
        # from agents.agent_specs import AGENTS as SPEC
        try:
            schema = schema_text
            # schema = next(a['json_schema'] for a in SPEC if a['name']==agent_name)
        except StopIteration:
            schema = '{}'

        # 这轮是否已发过角色？
        already_sent = st.session_state['agent_role_sent'].get(agent_name, False)

        # 额外的安全提示词
        no_evidence_clause = (
            "\n\nIMPORTANT:\n"
            "- No online sources were found in this session. Do not fabricate citations or specific data points.\n"
            "- If the answer requires external evidence, say clearly: 'No online sources found during the search' and proceed with general background knowledge only.\n"
        )

        role_text_final = job['role_text'] + "reply language based on the user's input language, default to English if uncertain."
        if no_online_evidence:
            role_text_final += no_evidence_clause

        citations_needed = (("literature searcher" in agent_name.lower()) and bool(sources_list))

        if not already_sent:
            # 首次发：带 role
            result = run_with_role(
                role_text=role_text_final,
                model=job['model'], temperature=job['temperature'],
                task_spec=parts['task'], examples=parts['examples'], cot=parts['cot'],
                question=parts['question'], output_format=parts['outfmt'],
                json_schema=schema, rag_snippets=rag_ctx,
                sources_list=sources_list,
                citations_parentheses=("literature searcher" in agent_name.lower()),  # 仅文献检索强制括号 DOI
                history_msgs=history_msgs
            )
            # 标记已发过
            st.session_state['agent_role_sent'][agent_name] = True
        else:
            # 后续轮次：不再带 role
            result = run_no_role(
                model=job['model'], temperature=job['temperature'],
                task_spec=parts['task'], examples=parts['examples'], cot=parts['cot'],
                question=parts['question'], output_format=parts['outfmt'],
                json_schema=schema,
                rag_snippets=rag_ctx,
                sources_list=sources_list,
                citations_parentheses=("literature searcher" in agent_name.lower()),
                history_msgs=history_msgs
            )


        # === 新增：保存历史 JSON（会话内），并把路径写回 saved_to ===
        try:
            conc = result.get("conclusion", {})
            fp = None
            if isinstance(conc, dict) and conc:
                fp = save_history_json(st.session_state, agent_name, conc)
                # 把 chat 里该条的 saved_to 更新为刚刚写入的历史路径
                # chat[-1]['saved_to'] = str(fp)
        except Exception:
            pass

         # …… 后续把 “正在回答…” 替换为 result['reply'] 并写入 rag_log（保持你现有代码）
        for i in range(len(chat)-1, -1, -1):
            if chat[i].get('pending'):
                chat[i] = {'role':'assistant','content':result['reply'],
                           'conclusion':conc,'saved_to':str(fp) or '',
                           'rag_log':rag_log}
                break

        #----- agent 为ChemProcess Modeler的情况 -----#
        if agent_name.lower().startswith("chemprocess"):
            # 1) 读取本轮结论中的布尔位（由 schema 新增的字段）
            conclusion = result.get("conclusion", {}) if isinstance(result, dict) else {}
            reply_text = result.get("reply", "").strip() if isinstance(result, dict) else ""
            raw_changed_flag = conclusion.get("variables_changed", None)  # True / False / None
            changed_flag = _to_bool(raw_changed_flag)
            raw_used_flag = conclusion.get("variables_used", None)
            used_flag = _to_bool(raw_used_flag)
            need_extract = False
            vars_list = load_variables(st.session_state) if load_variables else []
            if not vars_list and used_flag is True:
                need_extract = True
                st.info("No existing variables but model indicated variables were used (variables_used=true). Extracting…")
            elif changed_flag is True:
                need_extract = True
                st.info("Model indicated variables were changed (variables_changed=true). Extracting…")
            elif changed_flag is False:
                need_extract = False
                st.caption("Model indicated no variable changes (variables_changed=false).")
            else:
                # 布尔位缺失，做兜底判断
                if _maybe_changed_by_text(reply_text):
                    need_extract = True
                    st.info("No variables_changed flag. Heuristic indicates possible variable updates. Extracting…")
                else:
                    need_extract = False
                    st.caption("No variables_changed flag and no heuristic signal of variable updates.")

            # 3) 仅在需要时抽取/合并；否则不做无意义抽取
            extracted_any = False
            if need_extract is True:
                st.session_state['variables']['needextracted'] = need_extract
                try:
                    new_vars = extract_variables(reply_text, model=model, temperature=temperature)
                    if new_vars:
                        merged, changed = merge_variables(load_variables(st.session_state), new_vars)
                        if changed:
                            save_variables(merged, st.session_state)
                            st.success(f"Variables extracted/updated: {len(changed)}")
                            extracted_any = True
                            st.session_state['variables']['isextracted'] = extracted_any
                            # st.session_state['variables']['table'] = merged
                        else:
                            st.caption("No difference detected in variable table.")
                    else:
                        st.caption("No variables found in this reply.")
                except Exception as e:
                    st.warning(f"Variable extraction failed/skipped: {e}")

           
        st.session_state['proc_log'].append({'type':'agent','ts': time.strftime("%H:%M:%S"), 'agent': agent_name,'saved_to': str(fp) or ''})
        st.session_state['pending_agent'] = None
        st.rerun()

    # 提交输入
    if user_text is not None and user_text.strip():
        if rag_enable:
            upload_paths = _persist_uploads(rag_uploads)
            rag_brief = {"json_files":[os.path.basename(p) for p in rag_sel],
                         "upload_files":[os.path.basename(p) for p in upload_paths],
                         "urls": rag_urls}
        else:
            upload_paths=[]; rag_brief=None
        chat.append({'role':'user','content':user_text, 'rag_brief':rag_brief})
        parts = parse_five_parts(user_text)

        # --- 若是 ChemProcess Modeler，将最新变量表注入到 Task specification ---
        if agent_name.lower().startswith("chemprocess"):
            if load_variables:
                try:
                    vars_list = load_variables(st.session_state) or []
                except Exception:
                    vars_list = []

                if vars_list:
                    # 为了避免 prompt 过长，可选：做一个长度保护
                    vars_json = json.dumps(vars_list, ensure_ascii=False, indent=2)
                    # 可选长度限制（例如 8KB），太长就只给摘要
                    if len(vars_json) > 8_000:
                        # 取前 N 项并标注已截断（你也可以改成只取新近变更的变量）
                        head = vars_list[:40]
                        vars_json = json.dumps(head, ensure_ascii=False, indent=2)
                        vars_json += "\n/* NOTE: truncated for prompt length; see project_state/variables.json for full table. */"

                    # 将变量表附在 Task 段末尾（保留原 Task，不覆盖）
                    orig_task = parts.get('task') or ''
                    parts['task'] = (
                        f"{orig_task}\n\n"
                        f"[Variables table JSON] (auto-injected for latest symbols/units/meanings):\n"
                        f"{vars_json}\n"
                    )

        chat.append({'role':'assistant','content':'(Answering…)', 'pending': True})
        st.session_state['pending_agent'] = {'agent_name': agent_name,'role_text': role_text,'parts': parts,
                                             'rag_enable': rag_enable,'rag_sel': rag_sel,'rag_urls': rag_urls,
                                             'rag_upload_paths': upload_paths,'rag_k': rag_k,'rag_budget': rag_budget,
                                             'model': model,'temperature': temperature}
        st.session_state['turn_agents'][agent_name] = st.session_state['turn_agents'].get(agent_name, 0) + 1
        st.rerun()

    cols = st.columns([8,1])
    with cols[0]:
            # 新增：让用户选择“仅下一次调用时应用 Agent 角色”
        if st.button("Apply role to next call", key=f"{agent_name}_apply_role", type="primary"):
            st.session_state['agent_roles'][agent_name] = role_text
            st.session_state['agent_role_sent'][agent_name] = False  # 让下一次重新走 run_with_role
            st.success("This role will be applied on the next call only.")
    with cols[1]:    
        with stylable_container(
        key="danger_back_to_core",
        css_styles="""
        div[data-testid="stButton"] > button {
            background: #ef4444 !important;              /* 用 background 覆盖 background-image */
            background-image: none !important;           /* 防主题渐变/灰覆盖 */
            color: #fff !important;
            border: 1px solid #dc2626 !important;
            box-shadow: 0 2px 10px rgba(239,68,68,.25) !important;
            filter: none !important;                     /* 关掉可能的滤镜 */
            transition: background-color .15s ease, box-shadow .15s ease, transform .15s ease;
        }

        /* Hover / Focus / Active（红 600）——把所有可能态都重写，防发灰 */
        div[data-testid="stButton"] > button:not(:disabled):hover,
        div[data-testid="stButton"] > button:not(:disabled):focus,
        div[data-testid="stButton"] > button:not(:disabled):focus-visible,
        div[data-testid="stButton"] > button:not(:disabled):active {
            background: #dc2626 !important;              /* 更深一档而不是灰 */
            background-image: none !important;
            color: #fff !important;
            border-color: #b91c1c !important;            /* 边框也跟着加深 */
            box-shadow: 0 3px 12px rgba(220,38,38,.28) !important;
            filter: none !important;                     /* 禁止任何灰化滤镜 */
            transform: translateY(-1px);
            outline: none !important;
        }
        """
    ):
            if st.button('End branch', key="end_branch"):
                # 如果是 ChemProcess Modeler，删除变量表，避免影响其他 Agent 或新任务
                # if (st.session_state.get('current_agent') or "").lower().startswith("chemprocess"):
                #     try:
                #         ok, info = clear_variables(st.session_state)
                #         if ok:
                #             st.success(f"Variable table cleared ({info})")
                #         else:
                #             # 没有文件也不是错误，只提示
                #             st.caption(f"No variable table to clear ({info})")
                #     except Exception as e:
                #         st.warning(f"Failed to clear variables.json: {e}")
                
                # 重置该 agent 的已发送标记（下次再进入会重新发 role）
                try:
                    st.session_state['agent_role_sent'][agent_name] = False
                except Exception:
                    pass
                st.session_state['core_auto_rag'] = latest_entry_path(agent_name)
                st.session_state['page']='core'; st.session_state['current_agent']=None
                st.session_state['chat']['core'].append({'role':'assistant','content':f'You have completed the conversation with {agent_name}. If you want to use his conclusions, please upload the corresponding "conclusion json file" in the RAG of this round of dialogue with me, then continue to express your views and questions.'})
                st.rerun()
    ok, info = save_session(st.session_state)
    if not ok:
        # 可选：打印到日志或页面
        st.sidebar.warning(f"save_session failed: {info}")


# ---------- Main ----------
if st.session_state['page']=='core':
    core_page()
else:
    agent_page()
