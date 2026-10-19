# step3_vector_qa.py —— 2.0 完整问答：向量检索 + 大模型生成
# 英文关键词：context（上下文：拼给大模型看的资料）
#             system（系统提示：给大模型立规矩）

import requests
import numpy as np

# ============ 配置区：两个 key 都填上（只放自己电脑，别上传） ============
ZHIPU_KEY = ""     # ← 智谱 key（embedding 用）
DEEPSEEK_KEY = ""  # ← DeepSeek key（生成回答用）

EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"
CHAT_URL = "https://api.deepseek.com/v1/chat/completions"  # DeepSeek 对话接口地址

# ============ 读文档、切块 ============
with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks, start = [], 0
while start < len(text):
    end = start + 300
    chunks.append(text[start:end])
    start = end - 50
print(f"文档共切成 {len(chunks)} 块")

# ============ 向量化 ============
def get_embedding(text_piece):
    """文字变向量"""
    resp = requests.post(
        EMBED_URL,
        headers={"Authorization": f"Bearer {ZHIPU_KEY}", "Content-Type": "application/json"},
        json={"model": "embedding-3", "input": text_piece},
        timeout=60,
    )
    return resp.json()["data"][0]["embedding"]

print("向量化文档中...")
chunk_vectors = [get_embedding(c) for c in chunks]

def cosine_similarity(a, b):
    """余弦相似度"""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# ============ 检索 top3 ============
query = input("请输入你的问题：")
qv = get_embedding(query)
ranked = sorted(
    ((cosine_similarity(qv, cv), i) for i, cv in enumerate(chunk_vectors)),
    reverse=True,
)
top_chunks = [chunks[i] for _, i in ranked[:3]]
context = "\n\n".join(top_chunks)  # 把 3 块拼成一份资料

# ============ 调 DeepSeek 生成回答 ============
resp = requests.post(
    CHAT_URL,
    headers={"Authorization": f"Bearer {DEEPSEEK_KEY}", "Content-Type": "application/json"},
    json={
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是一个问答助手，只能根据下面提供的资料回答问题，不要编造。"},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{query}"},
        ],
    },
    timeout=60,
)
answer = resp.json()["choices"][0]["message"]["content"]
print("\n===== AI 回答 =====")
print(answer)
