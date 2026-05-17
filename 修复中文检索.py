
# 修复中文分词
from pathlib import Path
import numpy as np
from langchain_ollama import OllamaEmbeddings
import re

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
    pattern = re.compile(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+')
    return set(pattern.findall(text))

def enhanced_search(query, k=4):
    query_words = simple_chinese_tokenize(query)
    print(f"提取的关键词: {query_words}")
    
    keyword_matches = []
    for i, doc in enumerate(documents):
        content = doc['page_content']
        matched_words = query_words.intersection(simple_chinese_tokenize(content))
        if matched_words:
            keyword_matches.append((i, len(matched_words), matched_words))
    
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
    print(f"关键词匹配结果数量: {len(keyword_matches)}")
    
    top_keyword_indices = [idx for idx, _, _ in keyword_matches[:k*2]]
    
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    combined = {}
    for idx in set(top_keyword_indices):
        combined[idx] = similarities[idx] * 2.0
    
    for idx in np.argsort(similarities)[-k:][::-1]:
        if idx not in combined:
            combined[idx] = similarities[idx]
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [(i, sim, documents[i]) for i, sim in sorted_results[:k]]

# 测试查询
query = "工业管道的基本识别色有哪些"
print(f"\n查询: {query}\n")
print("=" * 60)

results = enhanced_search(query, k=5)

for i, (idx, score, doc) in enumerate(results):
    print(f"\n[结果 {i+1}] 文档{idx} (得分: {score:.4f})")
    print(f"内容: {doc['page_content'][:250]}...")

result_indices = [idx for idx, _, _ in results]
if 557 in result_indices or 558 in result_indices:
    print("\n\n✅ 相关文档在结果中！")
else:
    print(f"\n\n❌ 相关文档不在结果中。结果索引: {result_indices}")
