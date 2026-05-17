
# 测试为什么检索不到正确内容
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

# 测试不同的查询
queries = [
    "工业管道的基本识别色有哪些",
    "管道识别色",
    "艳绿色",
    "水蒸气",
    "氧气",
    "基本识别色",
    "GB 7231",
    "识别符号和危险标识"
]

embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)

print("\n测试不同查询与文档557的相似度：")
print("文档557内容预览:", documents[557]['page_content'][:150], "...\n")

target_idx = 557

for q in queries:
    query_vec = embeddings.embed_query(q)
    query_vec = np.array(query_vec)
    
    # 与文档557的相似度
    doc_vec = vectors_arr[target_idx]
    sim_557 = np.dot(doc_vec, query_vec) / (np.linalg.norm(doc_vec) * np.linalg.norm(query_vec) + 1e-10)
    
    # 全局最高相似度
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    top_idx = np.argmax(similarities)
    
    print(f"查询: '{q}'")
    print(f"  与文档557相似度: {sim_557:.4f}")
    print(f"  最高相似度: {similarities[top_idx]:.4f} (文档{top_idx})")
    print()
