
# 增强版检索 - 关键词+向量混合
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

def hybrid_search(query, k=4, boost_keywords=True):
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    if boost_keywords:
        keywords = query.replace("？", "").replace("?", "").split()
        scores = similarities.copy()
        
        for i, doc in enumerate(documents):
            content = doc['page_content']
            keyword_matches = sum(1 for kw in keywords if kw in content)
            if keyword_matches > 0:
                scores[i] *= (1 + keyword_matches * 0.3)
        
        top_indices = np.argsort(scores)[-k*2:][::-1]
    else:
        top_indices = np.argsort(similarities)[-k*2:][::-1]
    
    return [(i, similarities[i], documents[i]) for i in top_indices[:k]]

# 测试查询
query = "工业管道的基本识别色有哪些"
print(f"\n查询: {query}\n")
print("=" * 60)

results = hybrid_search(query, k=5)

for i, (idx, score, doc) in enumerate(results):
    print(f"\n[结果 {i+1}] 文档{idx} (向量相似度: {score:.4f})")
    print(f"内容: {doc['page_content'][:250]}...")

print("\n\n" + "=" * 60)
print("✅ 混合检索成功！")
print("=" * 60)
