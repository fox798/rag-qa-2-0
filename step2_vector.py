# step2_vector.py —— 2.0 向量检索版
# 英文关键词：embedding（嵌入/向量化：把文字变成一串数字）
#             cosine similarity（余弦相似度：算两个向量有多像，越接近 1 越像）

import requests  # 发网络请求（调 API 用）
import numpy as np  # 算向量、算相似度用

# ============ 配置区：key 填进引号里（只放自己电脑，别上传） ============
API_KEY = ""  # ← 把你的智谱 key 粘贴到这两个引号中间
EMBED_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"  # 智谱 embedding 接口地址
EMBED_MODEL = "embedding-3"  # 智谱的向量模型名字

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",  # 身份验证：带着 key 去，证明是你本人
    "Content-Type": "application/json",    # 告诉服务器：我发的是 JSON 格式数据
}

# ============ 第一步：读文档、切块（和 1.0 一样） ============
with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunk_size = 300  # 每块 300 字
overlap = 50      # 相邻两块重叠 50 字，别把句子切断

chunks = []
start = 0
while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start = end - overlap

print(f"文档共切成 {len(chunks)} 块")

# ============ 第二步：调 embedding 接口，把每块变成向量 ============
def get_embedding(text_piece):
    """把一段文字变成向量（一串小数，代表这段话的意思坐标）"""
    resp = requests.post(
        EMBED_URL,
        headers=HEADERS,
        json={"model": EMBED_MODEL, "input": text_piece},
        timeout=60,
    )
    data = resp.json()
    return data["data"][0]["embedding"]

print("正在把文档块变成向量，请稍等...")
chunk_vectors = [get_embedding(c) for c in chunks]
print("向量化完成！")

# ============ 第三步：问题也变成向量，算相似度排序 ============
def cosine_similarity(vec_a, vec_b):
    """余弦相似度：1 = 意思完全一样，0 = 完全不相关"""
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query = input("请输入你的问题：")
query_vector = get_embedding(query)

scores = []
for i, cv in enumerate(chunk_vectors):
    score = cosine_similarity(query_vector, cv)
    scores.append((score, i))

scores.sort(reverse=True)  # 分数从高到低排

top_k = 3
print(f"\n最相关的 {top_k} 块（向量检索结果）：")
for rank in range(top_k):
    score, idx = scores[rank]
    print(f"\n--- 第{rank+1}名：相似度 {score:.4f} ---")
    print(chunks[idx][:200])  # 只打印前 200 字预览
