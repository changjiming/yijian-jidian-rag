
# 使用简单余弦相似度的向量库 - 避免ChromaDB HNSW问题
import os
import json
import pickle
import shutil
from pathlib import Path
from typing import List, Optional
import numpy as np

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db_simple"
OLLAMA_URL = "http://localhost:11434"

class SimpleVectorStore:
    def __init__(self, persist_dir: str):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.persist_dir / "vectors.pkl"
        self.metadata_file = self.persist_dir / "metadata.json"
        self.vectors = []
        self.documents = []
        self._load()
    
    def _load(self):
        if self.vectors_file.exists():
            with open(self.vectors_file, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.documents = data['documents']
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save(self):
        with open(self.vectors_file, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'documents': self.documents
            }, f)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)
    
    def add_texts(self, texts: List[str], embeddings: np.ndarray, metadatas: Optional[List] = None):
        for i, text in enumerate(texts):
            self.documents.append({'page_content': text, 'metadata': metadatas[i] if metadatas else {}})
            self.vectors.append(embeddings[i])
        self._save()
    
    def similarity_search(self, query: str, k: int = 4, embedding=None) -> List:
        if not self.vectors:
            return []
        query_vec = embedding.embed_query(query)
        query_vec = np.array(query_vec)
        vectors = np.array(self.vectors)
        similarities = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec) + 1e-10)
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k]
    
    def __len__(self):
        return len(self.vectors)

print("=" * 60)
print("简单向量库重建工具")
print("=" * 60)

pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
if not pdf_files:
    print("❌ 没有找到PDF教材！")
    exit(1)

pdf_path = pdf_files[0]
pdf_stem = pdf_path.stem
db_dir = VECTOR_DB_DIR / pdf_stem

print(f"教材: {pdf_path.name}")
print(f"向量库目录: {db_dir}")

if db_dir.exists():
    print("删除旧向量库...")
    shutil.rmtree(db_dir)

db_dir.mkdir(parents=True, exist_ok=True)

print("\n加载RAG组件...")
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

print("加载PDF教材...")
loader = PDFPlumberLoader(str(pdf_path))
documents = loader.load()
print(f"加载了 {len(documents)} 页")

print("分割文本...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"分割为 {len(chunks)} 个文本块")

print("创建向量库...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=OLLAMA_URL)
vectorstore = SimpleVectorStore(str(db_dir))

batch_size = 50
print(f"分批处理，每批 {batch_size} 个...")

all_texts = [chunk.page_content for chunk in chunks]
all_metadatas = [chunk.metadata for chunk in chunks]

for i in range(0, len(all_texts), batch_size):
    batch_texts = all_texts[i:i+batch_size]
    batch_metadatas = all_metadatas[i:i+batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    print(f"  批次 {batch_num}/{total_batches} ({i}-{min(i+batch_size, len(all_texts))})...")
    
    batch_embeddings = embeddings.embed_documents(batch_texts)
    vectorstore.add_texts(batch_texts, np.array(batch_embeddings), batch_metadatas)

print(f"\n✅ 向量库创建完成！共 {len(vectorstore)} 个向量")

print("\n测试检索...")
results = vectorstore.similarity_search("工业管道的基本识别色", k=2, embedding=embeddings)
if results:
    print(f"✅ 检索成功！找到 {len(results)} 个相关文档")
    for doc in results:
        content = doc['page_content'][:300].replace('\n', ' ')
        print(f"\n{content}...")
else:
    print("❌ 没有找到相关文档")

print("\n✅ 完成！向量库已保存到: " + str(db_dir))
