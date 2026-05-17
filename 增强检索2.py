
# 增强版检索 - 关键词优先匹配
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

def enhanced_search(query, k=4):
    query_words = set(query.replace("？", "").replace("?", "").replace("，", " ").replace(",", " ").split())
    
    keyword_matches = []
    for i, doc in enumerate(documents):
        content = doc['page_content']
        match_count = sum(1 for w in query_words if len(w) >= 2 and w in content)
        if match_count > 0:
            keyword_matches.append((i, match_count, match_count / len(query_words)))
    
    keyword_matches.sort(key=lambda x: (x[1], x[2]), reverse=True)
    
    if keyword_matches:
        top_k_from_keyword = [idx for idx, _, _ in keyword_matches[:k]]
    else:
        top_k_from_keyword = []
    
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    top_k_from_vector = np.argsort(similarities)[-k:][::-1]
    
    combined = list(set(top_k_from_keyword + list(top_k_from_vector)))
    combined_scores = [(i, similarities[i] * (1.5 if i in top_k_from_keyword else 1.0)) for i in combined]
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    
    return [(i, sim, documents[i]) for i, sim in combined_scores[:k]]

# 测试查询
query = "工业管道的基本识别色有哪些"
print(f"\n查询: {query}\n")
print("=" * 60)

results = enhanced_search(query, k=5)

for i, (idx, score, doc) in enumerate(results):
    print(f"\n[结果 {i+1}] 文档{idx} (得分: {score:.4f})")
    print(f"内容: {doc['page_content'][:250]}...")

# 验证文档558是否在结果中
result_indices = [idx for idx, _, _ in results]
if 558 in result_indices:
    print("\n\n✅ 文档558（正确答案）在结果中！")
else:
    print("\n\n❌ 文档558不在结果中")
