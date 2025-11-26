from openai import OpenAI
import os
# 初始化客户端
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
messages = [
    {"role": "system", "content": "You are an expert chemist."}
]

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    messages.append({"role": "user", "content": user_input})

    MAX_HISTORY = 6  # 最近6条问答
    messages = messages[-(MAX_HISTORY * 2 + 1):]  # 包含 system + 6轮对话

    response = client.chat.completions.create(
        model="gpt-5-chat-latest",
        messages=messages
    )

    reply = response.choices[0].message.content
    print("GPT:", reply)

    messages.append({"role": "assistant", "content": reply})
