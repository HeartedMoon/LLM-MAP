import os, json, re, tempfile, zipfile
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import tiktoken

# ---------- 文件加载 ----------
# def _load_file(p: str):
#     """
#     统一文件加载：支持 docx/pdf/txt/md
#     """
#     ext=os.path.splitext(p)[1].lower()
#     debug={'path':p,'size':os.path.getsize(p),'primary':None,'error':None,'chars':0}
#     try:
#         if ext=='.pdf':
#             loader=PyPDFLoader(p); debug['primary']='PyPDFLoader'
#         elif ext=='.docx':
#             loader=UnstructuredWordDocumentLoader(p); debug['primary']='UnstructuredWordDocumentLoader'
#         elif ext in ['.txt','.md']:
#             loader=TextLoader(p,encoding='utf-8'); debug['primary']='TextLoader'
#         else:
#             loader=UnstructuredFileLoader(p); debug['primary']='UnstructuredFileLoader'
#         docs=loader.load()
#         chars=sum(len(d.page_content or "") for d in docs)
#         debug['chars']=chars
#         return docs, debug
#     except Exception as e:
#         debug['error']=str(e)
#         return [], debug

def _load_json_file(p: str) -> List["Document"]:
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    text = json.dumps(data, ensure_ascii=False, indent=2)
    return [Document(page_content=text, metadata={"source": os.path.basename(p), "type": "json"})]

def _load_zip_file(p: str) -> List["Document"]:
    """
    解压并逐个文件调用相应 loader：
    支持：pdf/docx/txt/md/json；其他类型忽略
    """
    docs: List[Document] = []
    with zipfile.ZipFile(p, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):  # 目录
                continue
            ext = os.path.splitext(name)[1].lower()
            # 只处理我们支持的后缀
            if ext not in [".pdf", ".docx", ".txt", ".md", ".json"]:
                continue

            # 将压缩文件中的条目写到临时文件再喂 loader（有的 loader 不支持 file-like）
            data = zf.read(name)
            tmp_dir = os.path.join(os.path.dirname(p), "__unzipped_tmp__")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, os.path.basename(name))
            with open(tmp_path, "wb") as tmpf:
                tmpf.write(data)

            # 递用 _load_file 处理解压出的临时文件
            sub_docs, _ = _load_file(tmp_path)
            for d in sub_docs:
                # 把来源标注成 “zipname::innername”
                meta = dict(d.metadata or {})
                meta["source"] = f"{os.path.basename(p)}::{name}"
                d.metadata = meta
                docs.append(d)

    return docs

def _load_file(p: str) -> Tuple[List["Document"], dict]:
    """
    统一文件加载：支持 pdf/docx/txt/md/json/zip
    """
    ext = os.path.splitext(p)[1].lower()
    debug = {'path': p, 'size': os.path.getsize(p), 'primary': None, 'error': None, 'chars': 0}

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(p); debug['primary'] = 'PyPDFLoader'
            docs = loader.load()

        elif ext == ".docx":
            loader = UnstructuredWordDocumentLoader(p); debug['primary'] = 'UnstructuredWordDocumentLoader'
            docs = loader.load()

        elif ext in [".txt", ".md"]:
            loader = TextLoader(p, encoding="utf-8"); debug['primary'] = 'TextLoader'
            docs = loader.load()

        elif ext == ".json":
            debug['primary'] = 'json_loader'
            docs = _load_json_file(p)

        elif ext == ".zip":
            debug['primary'] = 'zip_loader'
            docs = _load_zip_file(p)
            if not docs:
                raise ValueError("No supported files (pdf/docx/txt/md/json) found inside the zip.")

        else:
            # 兜底（可能解析常见 office / html 等）
            loader = UnstructuredFileLoader(p); debug['primary'] = 'UnstructuredFileLoader'
            docs = loader.load()

        chars = sum(len(getattr(d, "page_content", "") or "") for d in docs)
        debug['chars'] = chars
        return docs, debug

    except Exception as e:
        debug['error'] = str(e)
        return [], debug

# ---------- RAG 入口 ----------
def build_snippets(selected_json_paths, upload_paths, urls):
    docs=[]
    debug_sources={"json_files":[],"upload_files":[],"urls":urls or [],"file_debug":[],"url_debug":[]}

    # 历史 JSON
    for p in selected_json_paths or []:
        try:
            data=json.load(open(p,'r',encoding='utf-8'))
            txt=json.dumps(data.get("conclusion") or data,ensure_ascii=False,indent=2)
            docs.append(Document(page_content=txt,metadata={"source_type":"json","source_name":os.path.basename(p)}))
            debug_sources["json_files"].append(os.path.basename(p))
            debug_sources["file_debug"].append({'path':p,'primary':'json','error':None,'chars':len(txt)})
        except Exception as e:
            debug_sources["file_debug"].append({'path':p,'primary':'json','error':str(e),'chars':0})

    # 上传文件
    for p in upload_paths or []:
        ld,info=_load_file(p)
        debug_sources["file_debug"].append(info)
        if ld:
            for d in ld:
                d.metadata.update({"source_type":"file","source_name":os.path.basename(p)})
            docs+=ld
            debug_sources["upload_files"].append(os.path.basename(p))

    big_text="\n\n---\n\n".join([d.page_content for d in docs])
    return big_text,len(docs),debug_sources,docs

def build_vectorstore(docs: List[Document]):
    if not docs: return None,[]
    splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=120)
    chunks=splitter.split_documents(docs)
    vs=FAISS.from_documents(chunks,OpenAIEmbeddings())
    return vs,chunks

def _get_encoder(model_name:str):
    try: return tiktoken.encoding_for_model(model_name)
    except Exception: return tiktoken.get_encoding("cl100k_base")

def shrink_to_budget_by_chunks(chunks, enc, budget, sep="\n\n----\n\n"):
    out=[]; total=0
    for t in chunks:
        toks=len(enc.encode(t))
        if total+toks>budget: break
        out.append(t); total+=toks
    return sep.join(out)

def topk_context(vs, query:str, k:int=8, budget_tokens:int=2000, model_name:str="gpt-4o"):
    hits_detail=[]
    if not vs: return '',{"k":k,"hits":0,"budget_tokens":budget_tokens,"hits_detail":[]}
    retriever=vs.as_retriever(search_type="mmr",search_kwargs={"k":k,"fetch_k":max(20,k*3),"lambda_mult":0.5})
    chunks=retriever.get_relevant_documents(query or 'LFP leaching and metal recovery')
    texts=[c.page_content for c in chunks]
    for i,d in enumerate(chunks):
        md=d.metadata or {}
        hits_detail.append({
            "mode":"MMR","rank":i+1,
            "chunk_id":md.get("chunk_id"),"chars":len(d.page_content or ""),
            "source_type":md.get("source_type"),"source_name":md.get("source_name"),
            "preview":(d.page_content[:60] if d.page_content else "")
        })
    enc=_get_encoder(model_name)
    ctx=shrink_to_budget_by_chunks(texts,enc,budget_tokens)
    return ctx,{"k":k,"hits":len(texts),"budget_tokens":budget_tokens,"hits_detail":hits_detail}

def build_rag_ctx_and_log(enable, selected, urls, upload_paths, query_text:str,
                          k:int=8, budget_tokens:int=2000, model_name:str="gpt-4o",
                          extra_texts:str=""):
    if not enable: return "",None
    texts,_count,sources,docs=build_snippets(selected,upload_paths,urls)
    if extra_texts.strip():
        docs.append(Document(page_content=extra_texts,metadata={"source_type":"tmp","source_name":"lit_search"}))
        sources.setdefault("tmp_search",True)
    vs,splits=build_vectorstore(docs)
    if vs: ctx,hitstats=topk_context(vs,query=query_text,k=k,budget_tokens=budget_tokens,model_name=model_name)
    else: ctx,hitstats=(" ",{"k":k,"hits":0,"budget_tokens":budget_tokens})
    log={"enabled":True,"sources":sources,"split_chunks":len(splits) if splits else 0,"retrieval":hitstats}
    return ctx,log
