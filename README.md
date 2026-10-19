# RAG 知识库问答 2.0

基于检索增强生成（RAG）的中文知识库问答：输入问题，自动从本地文档中检索相关内容，再调用大模型生成回答。

## 技术栈
- 检索：智谱 embedding-3（向量化）+ 余弦相似度排序
- 生成：DeepSeek deepseek-chat
- 网页：Flask
- 其他：Python 3.13 / requests / numpy

## 功能
- 本地文档切块、向量化，建成索引
- 提问 → 向量检索 Top3 相关块 → 拼成上下文 → 大模型生成回答
- 网页界面：启动时一次性建索引，提问时只向量化问题

## 1.0 → 2.0 升级
| | 1.0 | 2.0 |
|---|---|---|
| 检索方式 | 关键词匹配（数词频） | 向量检索（embedding + 余弦相似度） |
| 换说法提问 | 失效（字对不上就 0 分） | 按语义匹配，照样找到 |
| 索引 | 每次提问全量重跑 | 启动时一次性建索引，常驻内存 |
| 界面 | 命令行黑窗口 | Flask 网页，可演示 |

## 运行方法
1. 安装依赖：`pip install requests numpy flask`
2. 申请智谱 API Key（embedding 用）、DeepSeek API Key（生成用），填入 `app.py` 配置区
3. 知识文档保存为 `docs.txt`（utf-8 编码）
4. 运行 `python app.py`，浏览器打开 http://127.0.0.1:5000

## 文件说明
- `app.py`：2.0 网页版（推荐运行）
- `step3_vector_qa.py`：2.0 命令行版（向量检索 + 生成）
- `step2_vector.py`：向量检索演示
- `step1_split.py / step2_search.py / step3_qa.py`：1.0 学习过程（关键词检索版）
