
# 使用新版 langchain_chroma 重建向量库
import os
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"
OLLAMA_URL = "http://localhost:11434"

pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
if not pdf_files:
    print("❌ 没有找到PDF教材！")
    exit(1)

pdf_path = pdf_files[0]
pdf_stem = pdf_path.stem
db_dir = VECTOR_DB_DIR / pdf_stem

print(f"教材: {pdf_path.name}")

if db_dir.exists():
    print("删除旧向量库...")
    shutil.rmtree(db_dir)

db_dir.mkdir(parents=True, exist_ok=True)

print("加载RAG组件...")
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

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

batch_size = 100
print(f"分批处理，每批 {batch_size} 个...")

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    print(f"  处理批次 {batch_num}/{total_batches} ({i}-{min(i+batch_size, len(chunks))})...")
    
    if i == 0:
        vectorstore = Chroma.from_documents(
            batch, 
            embeddings, 
            persist_directory=str(db_dir)
        )
    else:
        vectorstore.add_documents(batch)

count = vectorstore._collection.count()
print(f"✅ 向量库创建完成！共 {count} 个向量")

print("\n测试检索...")
docs = vectorstore.similarity_search("工业管道的基本识别色", k=2)
if docs:
    print(f"✅ 检索成功！找到 {len(docs)} 个相关文档")
    for doc in docs:
        print(f"\n{doc.page_content[:300]}...")
else:
    print("❌ 没有找到相关文档")

print("\n✅ 完成！")
