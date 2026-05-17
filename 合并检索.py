
# 合并相邻文档的改进检索
from pathlib import Path
import numpy as np
from langchain_ollama import OllamaEmbeddings
import re
import pickle

db_dir = Path(__file__).parent / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"
with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    documents = data['documents']
    vectors = data['vectors']

vectors_arr = np.array(vectors)
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

def extract_chinese_tokens(text):
    tokens = set()
    for match in re.finditer(r'[\u4e00-\u9fff]+', text):
        token = match.group()
        for i in range(len(token)):
            for j in range(i+2, min(i+5, len(token)+1)):
                tokens.add(token[i:j])
    return tokens

def search_and_merge(query, k=4):
    query_tokens = extract_chinese_tokens(query)
    
    keyword_matches = []
    for i, doc in enumerate(documents):
        matched = query_tokens.intersection(extract_chinese_tokens(doc['page_content']))
        if matched:
            keyword_matches.append((i, len(matched)))
    
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
    
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    results = []
    for idx, match_count in keyword_matches[:k*10]:
        results.append((idx, similarities[idx] * (1 + match_count * 0.3)))
    
    for idx in np.argsort(similarities)[-k:][::-1]:
        if not any(idx == r[0] for r in results):
            results.append((idx, similarities[idx]))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    merged = []
    seen_content = set()
    for idx, score in results[:k*3]:
        content = documents[idx]['page_content']
        if content not in seen_content:
            merged.append((idx, content))
            seen_content.add(content)
            if len(merged) >= k:
                break
    
    return merged

question = "工业管道的基本识别色有哪些"
print(f"问题: {question}\n")

results = search_and_merge(question, k=5)

print("检索结果:")
for i, (idx, content) in enumerate(results):
    print(f"\n[文档{idx}] {content[:200]}...")

# 提取连续文档556,557,558合并
print("\n\n" + "="*60)
print("尝试合并连续文档556-558:")
print("="*60)

for start in range(len(documents)):
    if '基本识别色' in documents[start]['page_content']:
        print(f"\n从文档{start}开始:")
        merged = documents[start]['page_content']
        for i in range(start+1, min(start+3, len(documents))):
            merged += "\n" + documents[i]['page_content']
            if '识别符号' in documents[i]['page_content'] and '艳绿色' in documents[i]['page_content']:
                break
        print(merged[:800])
        break
