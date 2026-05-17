
# 快速重建向量库测试脚本
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "chroma_db"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"

print("=" * 60)
print("向量库重建测试")
print("=" * 60)

# 1. 找PDF
pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
if not pdf_files:
    print("❌ 没有找到PDF教材！")
    sys.exit(1)

pdf_path = pdf_files[0]
pdf_stem = pdf_path.stem
db_dir = VECTOR_DB_DIR / pdf_stem

print(f"\n📄 教材: {pdf_path.name}")
print(f"📦 向量库目录: {db_dir}")

# 2. 删除旧向量库
if db_dir.exists():
    import shutil
    shutil.rmtree(db_dir)
    print("🗑️ 已删除旧向量库")

db_dir.mkdir(parents=True, exist_ok=True)

# 3. 导入RAG组件
print("\n加载RAG组件...")
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

print("✅ RAG组件加载成功！")

# 4. 加载PDF
print("\n加载PDF教材...")
loader = PDFPlumberLoader(str(pdf_path))
documents = loader.load()
print(f"✅ 加载了 {len(documents)} 页")

# 5. 分割文本
print("\n分割文本...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"✅ 分割为 {len(chunks)} 个文本块")

# 6. 创建向量库
print("\n创建向量库（这可能需要几分钟）...")
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory=str(db_dir)
)
print("✅ 向量库创建完成！")

# 7. 测试检索
print("\n" + "=" * 60)
print("测试检索：工业管道的基本识别色有哪些")
print("=" * 60)

docs = vectorstore.similarity_search("工业管道的基本识别色有哪些", k=2)
if docs:
    print(f"\n✅ 找到 {len(docs)} 个相关文档：\n")
    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:300].replace('\n', ' ')
        print(f"文档 {i}: {content}...")
        print()
else:
    print("❌ 没有找到相关文档")

print("\n" + "=" * 60)
print("✅ 测试完成！向量库已重建成功！")
print("=" * 60)
