
# 直接测试向量库检索
from pathlib import Path
import numpy as np

BASE_DIR = Path(__file__).parent
db_dir = BASE_DIR / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"

from langchain_ollama import OllamaEmbeddings

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"

print("加载向量库...")
import pickle
import json

with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    vectors = data['vectors']
    documents = data['documents']

print(f"向量数量: {len(vectors)}")
print(f"向量维度: {len(vectors[0]) if vectors else 0}")

print("\n加载Embedding模型...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

query = "工业管道的基本识别色有哪些"
print(f"\n查询: {query}")

query_vec = embeddings.embed_query(query)
query_vec = np.array(query_vec)

vectors_arr = np.array(vectors)
similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)

top_indices = np.argsort(similarities)[-5:][::-1]

print("\n检索结果：")
for i, idx in enumerate(top_indices):
    print(f"\n[结果 {i+1}] 相似度: {similarities[idx]:.4f}")
    print(f"内容: {documents[idx]['page_content'][:300]}...")

# 测试几个不同的查询
queries = [
    "工业管道的基本识别色",
    "管道识别色",
    "焊接方法",
    "机电安装"
]

print("\n" + "=" * 60)
print("测试多个查询：")
print("=" * 60)

for q in queries:
    query_vec = embeddings.embed_query(q)
    query_vec = np.array(query_vec)
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    top_idx = np.argmax(similarities)
    print(f"\n查询: {q}")
    print(f"最高相似度: {similarities[top_idx]:.4f}")
    print(f"内容: {documents[top_idx]['page_content'][:150]}...")
