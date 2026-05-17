
# 检查检索到的高相似度文档内容
from pathlib import Path
import numpy as np
from langchain_ollama import OllamaEmbeddings

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"

db_dir = Path(__file__).parent / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"

print("加载向量库...")
import pickle
with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    vectors = data['vectors']
    documents = data['documents']

vectors_arr = np.array(vectors)

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

query = "工业管道的基本识别色有哪些"
query_vec = np.array(embeddings.embed_query(query))

similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
top_indices = np.argsort(similarities)[-10:][::-1]

print(f"\n查询: {query}\n")
print("=" * 60)
print("Top 10 检索结果：")
print("=" * 60)

for i, idx in enumerate(top_indices):
    print(f"\n[结果 {i+1}] 文档{idx} 相似度: {similarities[idx]:.4f}")
    print(f"内容: {documents[idx]['page_content'][:200]}...")

# 特别检查文档558
print("\n" + "=" * 60)
print("文档558的真实内容：")
print("=" * 60)
print(documents[558]['page_content'][:500])
