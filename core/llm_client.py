import os, json, time
from openai import OpenAI
import tiktoken

_client = None

def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    return _client

def chat_complete(model:str, temperature:float, messages:list, max_tokens:int=2048, timeout:int=60):
    """
    统一调用接口
    """
    cli = get_client()
    try:
        resp = cli.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[错误详情] LLM请求失败或超时: {e}"

def count_tokens(text:str, model_name:str="gpt-4o-mini")->int:
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text or ""))
