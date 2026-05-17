
# 验证向量库检索功能
from pathlib import Path
import sys

BASE_DIR = Path(__file__).parent
VECTOR_DB_DIR = BASE_DIR / "data" / "chroma_db"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"

print("=" * 60)
print("向量库检索验证")
print("=" * 60)

pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
if not pdf_files:
    print("❌ 没有找到PDF教材！")
    sys.exit(1)

pdf_stem = pdf_files[0].stem
db_dir = VECTOR_DB_DIR / pdf_stem

print(f"\n📄 教材: {pdf_files[0].name}")
print(f"📦 向量库目录: {db_dir}")

if not db_dir.exists():
    print("❌ 向量库目录不存在！需要重建。")
    sys.exit(1)

import os
files = list(db_dir.iterdir())
print(f"\n向量库文件数: {len(files)}")
for f in files:
    size = os.path.getsize(f) / 1024
    print(f"  - {f.name} ({size:.1f} KB)")

if len(files) == 0:
    print("❌ 向量库为空！需要重建。")
    sys.exit(1)

print("\n正在连接向量库...")
try:
    from langchain_ollama import OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vectorstore = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings
    )
    
    print("✅ 向量库连接成功！")
    
    # 获取向量数量
    count = vectorstore._collection.count()
    print(f"📊 向量数量: {count}")
    
    if count == 0:
        print("❌ 向量库为空！需要重建。")
        sys.exit(1)
    
    # 测试检索
    print("\n" + "=" * 60)
    print("测试检索：工业管道的基本识别色有哪些")
    print("=" * 60)
    
    docs = vectorstore.similarity_search("工业管道的基本识别色有哪些", k=2)
    
    if docs:
        print(f"\n✅ 找到 {len(docs)} 个相关文档：\n")
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:400].replace('\n', ' ')
            print(f"文档 {i}: {content}...")
            print()
    else:
        print("❌ 没有找到相关文档！向量库可能损坏。")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 验证通过！检索功能正常！")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ 验证失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n需要重建向量库。")
