
# 测试改进的检索 - 增加数量并合并相邻内容
from pathlib import Path
import numpy as np
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import pickle
import time

db_dir = Path(__file__).parent / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"
with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    vectors = data['vectors']
    documents = data['documents']

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

def improved_search(query, k=5):
    query_tokens = extract_chinese_tokens(query)
    
    keyword_matches = []
    for i, doc in enumerate(documents):
        matched = query_tokens.intersection(extract_chinese_tokens(doc['page_content']))
        if matched:
            keyword_matches.append((i, len(matched)))
    
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
    top_keyword_indices = [idx for idx, _ in keyword_matches[:k*5]]
    
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    combined = {}
    for idx in top_keyword_indices:
        combined[idx] = similarities[idx] * 3.0
    
    for idx in np.argsort(similarities)[-k:][::-1]:
        if idx not in combined:
            combined[idx] = similarities[idx]
    
    sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [i for i, _ in sorted_results[:k]]

question = "工业管道的基本识别色有哪些"
print(f"问题: {question}\n")

indices = improved_search(question, k=8)
print(f"检索到的文档: {indices}")

print("\n文档内容:")
for idx in indices:
    print(f"\n--- 文档{idx} ---")
    print(documents[idx]['page_content'][:300])
