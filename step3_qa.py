# step3_qa.py
# RAG 第三步：问答生成
# 作用：把检索到的相关文档块喂给大模型，让它根据资料回答问题
# 流程：切分 -> 检索 -> 拼 prompt -> 调 DeepSeek API -> 输出答案

import re
import requests

# ============ 填你的 key（只放你自己电脑上，别发出来） ============
API_KEY = ""
# =================================================================


def split_text(text, chunk_size=300, overlap=50):
    """复用第一步的切分函数"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def search(query, chunks, top_k=3):
    """复用第二步的检索函数"""
    raw = re.findall(r'[a-zA-Z]{2,}|[\u4e00-\u9fff]+', query)
    keywords = []
    for w in raw:
        if w[0].isascii():
            keywords.append(w)
        else:
            keywords += [w[i:i+2] for i in range(len(w) - 1)]
    scores = []
    for chunk in chunks:
        score = sum(chunk.count(kw) for kw in keywords)
        scores.append(score)
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [(chunks[i], scores[i]) for i in ranked[:top_k]]


def ask_ai(query, context):
    """调 DeepSeek 生成回答"""
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + API_KEY,
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个技术支持知识库问答助手。只能根据下面提供的资料回答问题，资料里没有就说不知道，不要编造。"},
            {"role": "user", "content": "资料：\n" + context + "\n\n问题：" + query},
        ],
    }
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    if resp.status_code == 401:
        return "【key 错了】检查 API_KEY 是不是复制完整"
    return resp.json()["choices"][0]["message"]["content"]


with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()
chunks = split_text(text)

query = "蓝屏了怎么办"
results = search(query, chunks)
context = "\n\n".join([chunk for chunk, score in results])

print("问题：" + query)
print("检索到 " + str(len(results)) + " 块相关资料，正在问 AI……\n")
answer = ask_ai(query, context)
print("AI 回答：")
print(answer)
