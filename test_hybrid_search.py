
"""
测试混合检索功能
"""
import sys
from pathlib import Path
import pickle

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db_simple"

# 找到PDF文件
pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
if not pdf_files:
    print("❌ 未找到PDF文件")
    sys.exit(1)

pdf_path = pdf_files[0]
pdf_stem = pdf_path.stem
db_dir = VECTOR_DB_DIR / pdf_stem

print(f"📄 使用教材: {pdf_path.name}")
print(f"📦 向量库路径: {db_dir}")

# 检查是否有向量库
if not (db_dir.exists() and any(db_dir.iterdir())):
    print("❌ 向量库不存在，需要先运行主程序创建！")
    print("   请先运行: python desktop_app.py")
    sys.exit(1)

print("\n" + "="*60)
print("1. 测试加载向量库...")

# 先修改sys.path，让我们能导入desktop_app
sys.path.insert(0, str(BASE_DIR))
from desktop_app import SimpleVectorStore
from langchain_ollama import OllamaEmbeddings

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"

# 加载向量库
embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
vectorstore = SimpleVectorStore(str(db_dir), embeddings)
print(f"✅ 向量库加载成功，共 {len(vectorstore)} 个文档")

# 测试查询
queries = [
    "工业管道的基本识别色有哪些",
    "焊接方法有几种",
    "机电安装注意事项"
]

for i, query in enumerate(queries):
    print("\n" + "="*60)
    print(f"{i+1}. 测试查询: {query}")
    print("="*60)
    
    print("\n--- 混合检索结果 ---")
    hybrid_results = vectorstore.hybrid_search(query, k=5)
    for j, (idx, score) in enumerate(hybrid_results):
        doc = vectorstore.documents[idx]
        if isinstance(doc, dict):
            content = doc.get('page_content', '')
            metadata = doc.get('metadata', {})
        else:
            content = ''
            metadata = {}
        
        print(f"\n  结果{j+1}: 文档{idx} (得分: {score:.4f})")
        if metadata and 'page' in metadata:
            print(f"  页码: {metadata['page']}")
        print(f"  内容: {content[:150]}...")
    
    print("\n--- 最终检索结果 (合并后) ---")
    results = vectorstore.enhanced_similarity_search(query, k=3)
    for j, doc in enumerate(results):
        print(f"\n  结果{j+1}:")
        if 'page' in doc.get('metadata', {}):
            print(f"  页码: {doc['metadata']['page']}")
        print(f"  内容: {doc['page_content'][:200]}...")

print("\n" + "="*60)
print("✅ 混合检索功能测试完成！")
print("现在运行程序: python desktop_app.py")
