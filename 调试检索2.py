
# 调试检索问题
from pathlib import Path
import numpy as np
from langchain_ollama import OllamaEmbeddings
import re

db_dir = Path(__file__).parent / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"
import pickle

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

question = "工业管道的基本识别色有哪些"
query_tokens = extract_chinese_tokens(question)

print(f"查询关键词数量: {len(query_tokens)}")

# 检查文档557, 558的关键词匹配
for doc_idx in [557, 558]:
    doc_tokens = extract_chinese_tokens(documents[doc_idx]['page_content'])
    matched = query_tokens.intersection(doc_tokens)
    print(f"\n文档{doc_idx}:")
    print(f"  匹配关键词数: {len(matched)}")
    print(f"  部分匹配: {list(matched)[:10]}")
    print(f"  内容开头: {documents[doc_idx]['page_content'][:100]}")

# 检查向量相似度
query_vec = np.array(embeddings.embed_query(question))
sim_557 = np.dot(vectors_arr[557], query_vec) / (np.linalg.norm(vectors_arr[557]) * np.linalg.norm(query_vec))
sim_558 = np.dot(vectors_arr[558], query_vec) / (np.linalg.norm(vectors_arr[558]) * np.linalg.norm(query_vec))
print(f"\n向量相似度:")
print(f"  文档557: {sim_557:.4f}")
print(f"  文档558: {sim_558:.4f}")

# 为什么558排不上？
print("\n558排在多少位？")
similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
rank_558 = (similarities > sim_558).sum() + 1
print(f"  文档558排名: {rank_558}")
print(f"  文档557排名: {(similarities > sim_557).sum() + 1}")
