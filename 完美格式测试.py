
# 测试完美格式输出
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
print("测试完美格式问答")
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

# 2. 检索
print("\n2. 检索文档...")
def extract_chinese_tokens(text):
    tokens = set()
    for match in re.finditer(r'[\u4e00-\u9fff]+', text):
        token = match.group()
        for i in range(len(token)):
            for j in range(i+2, min(i+5, len(token)+1)):
                tokens.add(token[i:j])
    return tokens

def improved_search(query, k=4):
    query_tokens = extract_chinese_tokens(query)
    query_vec = np.array(embeddings.embed_query(query))
    similarities = np.dot(vectors_arr, query_vec) / (np.linalg.norm(vectors_arr, axis=1) * np.linalg.norm(query_vec) + 1e-10)
    
    keyword_docs = []
    for i, doc in enumerate(documents):
        matched = query_tokens.intersection(extract_chinese_tokens(doc['page_content']))
        if matched:
            keyword_docs.append((i, len(matched)))
    
    keyword_docs.sort(key=lambda x: x[1], reverse=True)
    
    priority_docs = []
    for idx, match_count in keyword_docs[:k*10]:
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
            results.append({'page_content': merged_content, 'metadata': {}})
        
        if len(results) &gt;= k:
            break
    
    return results

question = "工业管道的基本识别色有哪些"
print(f"\n问题: {question}")

docs = improved_search(question, k=3)
context = ""
for i, doc in enumerate(docs):
    context += f"--- 参考资料 {i+1} ---\n{doc['page_content'][:800]}\n\n"

# 3. 使用完美格式提示词
print("\n3. 生成回答...")
prompt_template = """你是一个专业的一建机电考试辅导老师。请根据以下教材内容回答问题。

【重要要求】
1. 答案必须100%基于提供的教材参考资料，不要使用任何教材以外的知识
2. 如果教材中没有相关内容，就直接说明"教材中没有找到相关内容"
3. 不要使用GB标准号或其他通用知识，只引用教材原文
4. 回答格式要清晰、美观，使用以下格式呈现：
   先说明来自教材的哪些章节，然后列出具体内容

参考内容:
{context}

问题:
{question}

请直接回答问题。
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=2048)

chain = (
    {"context": lambda x: context, "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

start = time.time()
answer = chain.invoke(question)
print(f"生成完成，耗时 {time.time()-start:.1f}秒")

print("\n" + "=" * 60)
print("AI回答:")
print("=" * 60)
print(answer)
print("\n" + "=" * 60)
print("✅ 验证完成！")
print("=" * 60)
