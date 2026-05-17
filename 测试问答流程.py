
# 测试完整问答流程
from pathlib import Path
import sys
import time

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db_simple"

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "mxbai-embed-large"

print("=" * 60)
print("测试完整问答流程")
print("=" * 60)

# 1. 加载向量库
print("\n1. 加载向量库...")
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import pickle
import json
import numpy as np
import re

pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
if not pdf_files:
    print("❌ 没有找到PDF教材！")
    sys.exit(1)

pdf_path = pdf_files[0]
pdf_stem = pdf_path.stem
db_dir = VECTOR_DB_DIR / pdf_stem

with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    vectors = data['vectors']
    documents = data['documents']

vectors_arr = np.array(vectors)
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
print(f"✅ 加载了 {len(documents)} 个文档")

# 2. 增强检索
print("\n2. 增强检索...")
def extract_chinese_tokens(text):
    tokens = set()
    for match in re.finditer(r'[\u4e00-\u9fff]+', text):
        token = match.group()
        for i in range(len(token)):
            for j in range(i+2, min(i+5, len(token)+1)):
                tokens.add(token[i:j])
    return tokens

def enhanced_similarity_search(query, k=4):
    query_tokens = extract_chinese_tokens(query)
    
    keyword_matches = []
    for i, doc in enumerate(documents):
        content = doc['page_content']
        matched = query_tokens.intersection(extract_chinese_tokens(content))
        if matched:
            keyword_matches.append((i, len(matched)))
    
    keyword_matches.sort(key=lambda x: x[1], reverse=True)
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
    return [(i, documents[i]) for i, _ in sorted_results[:k]]

question = "工业管道的基本识别色有哪些"
print(f"\n问题: {question}")

docs = enhanced_similarity_search(question, k=3)
print(f"✅ 找到 {len(docs)} 个相关文档")

print("\n检索到的文档：")
for i, (idx, doc) in enumerate(docs):
    print(f"\n[文档 {idx}] {doc['page_content'][:200]}...")

# 3. 生成回答
print("\n3. 生成回答...")
context = "\n\n".join([f"--- 参考资料 {i+1} ---\n{doc['page_content'][:800]}" for i, (_, doc) in enumerate(docs)])

prompt_template = """你是一个专业的一建机电考试辅导老师。请根据以下教材内容回答问题。

参考内容:
{context}

问题:
{question}

请直接回答问题，答案要详细、专业、准确。
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=2048)

chain = (
    {"context": lambda x: context, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

print("正在生成回答...")
start = time.time()
answer = chain.invoke(question)
print(f"回答生成完成，耗时 {time.time()-start:.1f}秒")

print("\n" + "=" * 60)
print("AI回答：")
print("=" * 60)
print(answer)
print("\n" + "=" * 60)
print("✅ 测试完成！")
print("=" * 60)
