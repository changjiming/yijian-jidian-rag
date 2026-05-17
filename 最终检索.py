
# 最终版检索 - 关键词优先 + 内容合并
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

def final_search(query, k=4):
    query_tokens = extract_chinese_tokens(query)
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    keyword_docs = []
    for i, doc in enumerate(documents):
        matched = query_tokens.intersection(extract_chinese_tokens(doc['page_content']))
        if matched:
            keyword_docs.append((i, len(matched), matched))
    
    keyword_docs.sort(key=lambda x: x[1], reverse=True)
    
    priority_docs = []
    for idx, match_count, matched in keyword_docs[:k*10]:
        score = similarities[idx] * (1 + match_count * 0.2)
        priority_docs.append((idx, score, match_count))
    
    for idx in np.argsort(similarities)[-k:][::-1]:
        if not any(idx == p[0] for p in priority_docs):
            priority_docs.append((idx, similarities[idx], 0))
    
    priority_docs.sort(key=lambda x: x[1], reverse=True)
    
    results = []
    seen_indices = set()
    for idx, score, match_count in priority_docs[:k*3]:
        content = documents[idx]['page_content']
        
        merged_content = content
        if match_count > 5:
            for next_idx in [idx+1, idx-1]:
                if 0 <= next_idx < len(documents):
                    next_content = documents[next_idx]['page_content']
                    if len(next_content) + len(merged_content) < 2000:
                        if any(kw in next_content for kw in ['艳绿色', '大红色', '淡灰色', '中黄色', '淡蓝色', '基本识别色', '识别符号']):
                            merged_content += "\n" + next_content
                            if next_idx not in seen_indices:
                                seen_indices.add(next_idx)
        
        if idx not in seen_indices:
            seen_indices.add(idx)
            results.append((idx, merged_content))
        
        if len(results) >= k:
            break
    
    return results

question = "工业管道的基本识别色有哪些"
print(f"问题: {question}\n")

results = final_search(question, k=4)

print("检索结果:")
for i, (idx, content) in enumerate(results):
    print(f"\n[文档{idx}]")
    print(content[:600])
    print("...")
