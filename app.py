# app.py —— RAG 2.0 网页版（单文件）
# 英文关键词：Flask（轻量网页框架） route（路由：网址对应哪个函数处理）

from flask import Flask, request, render_template_string
import requests
import numpy as np

app = Flask(__name__)  # 创建网页应用

# ============ 配置区 ============
ZHIPU_KEY = ""     # ← 智谱 key
DEEPSEEK_KEY = ""  # ← DeepSeek key

# ============ 启动时一次性建索引（向量化文档，常驻内存） ============
with open("docs.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunks, start = [], 0
while start < len(text):
    end = start + 300
    chunks.append(text[start:end])
    start = end - 50

def get_embedding(t):
    """文字变向量"""
    r = requests.post(
        "https://open.bigmodel.cn/api/paas/v4/embeddings",
        headers={"Authorization": f"Bearer {ZHIPU_KEY}", "Content-Type": "application/json"},
        json={"model": "embedding-3", "input": t},
        timeout=60,
    )
    return r.json()["data"][0]["embedding"]

print("正在建索引（向量化文档），请稍等...")
chunk_vectors = [get_embedding(c) for c in chunks]
print(f"索引建好，共 {len(chunks)} 块，网页已就绪！")

def search(query):
    """向量检索 top3"""
    qv = np.array(get_embedding(query))
    scored = []
    for i, cv in enumerate(chunk_vectors):
        cv = np.array(cv)
        s = float(np.dot(qv, cv) / (np.linalg.norm(qv) * np.linalg.norm(cv)))
        scored.append((s, i))
    scored.sort(reverse=True)
    return [chunks[i] for _, i in scored[:3]]

def ask_ai(query, context):
    """调 DeepSeek 生成回答"""
    r = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
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
    return r.json()["choices"][0]["message"]["content"]

# ============ 网页模板 ============
PAGE = """
<!doctype html>
<html><head><meta charset="utf-8"><title>RAG 问答 2.0</title></head>
<body style="max-width:700px;margin:40px auto;font-family:sans-serif">
<h2>RAG 知识库问答 2.0</h2>
<form method="post">
<input name="q" style="width:400px;padding:8px" placeholder="输入你的问题，比如：蓝屏了怎么排查？">
<button style="padding:8px 16px">提问</button>
</form>
{% if answer %}
<h3>AI 回答</h3>
<p style="background:#f0f0f0;padding:12px">{{ answer }}</p>
{% endif %}
</body></html>
"""

@app.route("/", methods=["GET", "POST"])  # 网址 / 交给下面这个函数处理
def index():
    answer = ""
    if request.method == "POST":  # 用户点了"提问"按钮
        query = request.form["q"]       # 拿到输入框里的字
        top3 = search(query)            # 向量检索
        context = "\n\n".join(top3)     # 拼资料
        answer = ask_ai(query, context) # 生成回答
    return render_template_string(PAGE, answer=answer)

if __name__ == "__main__":
    app.run(debug=True)  # 启动网页服务
