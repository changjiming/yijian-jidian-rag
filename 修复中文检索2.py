
# 修复中文分词 - 使用更好的正则
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

def simple_chinese_tokenize(text):
    import re
    tokens = []
    for match in re.finditer(r'[\u4e00-\u9fff]+', text):
        token = match.group()
        for i in range(len(token)):
            for j in range(i+2, min(i+5, len(token)+1)):
                tokens.append(token[i:j])
    return set(tokens)

def enhanced_search(query, k=4):
    query_words = simple_chinese_tokenize(query)
    print(f"提取的关键词数量: {len(query_words)}")
    print(f"部分关键词: {list(query_words)[:10]}")
    
    keyword_matches = []
    for i, doc in enumerate(documents):
        content = doc['page_content']
        matched_words = query_words.intersection(simple_chinese_tokenize(content))
        if matched_words:
            keyword_matches.append((i, len(matched_words)))
    
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
    print(f"关键词匹配到的文档数: {len(keyword_matches)}")
    
    top_keyword_indices = [idx for idx, _ in keyword_matches[:k*3]]
    
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    combined = {}
    for idx in top_keyword_indices:
        combined[idx] = similarities[idx] * 2.5
    
    for idx in np.argsort(similarities)[-k:][::-1]:
        if idx not in combined:
            combined[idx] = similarities[idx]
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [(i, sim, documents[i]) for i, sim in sorted_results[:k]]

query = "工业管道的基本识别色有哪些"
print(f"\n查询: {query}\n")
print("=" * 60)

results = enhanced_search(query, k=5)

for i, (idx, score, doc) in enumerate(results):
    print(f"\n[结果 {i+1}] 文档{idx} (得分: {score:.4f})")
    print(f"内容: {doc['page_content'][:200]}...")

result_indices = [idx for idx, _, _ in results]
if 557 in result_indices or 558 in result_indices:
    print("\n\n✅ 相关文档在结果中！")
