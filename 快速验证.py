"""
快速测试混合检索 - 等向量库创建完成后运行
"""
import sys
from pathlib import Path
import time

BASE_DIR = Path(__file__).parent
VECTOR_DB_DIR = BASE_DIR / "data" / "vector_db_simple"

# 检查向量库是否存在
pdf_files = list((BASE_DIR / "data" / "uploads").glob("*.pdf"))
if not pdf_files:
    print("❌ 未找到PDF文件")
    sys.exit(1)

pdf_stem = pdf_files[0].stem
db_dir = VECTOR_DB_DIR / pdf_stem

if not (db_dir.exists() and any(db_dir.iterdir())):
    print(f"❌ 向量库不存在: {db_dir}")
    print("请先运行 python desktop_app.py 创建向量库")
    sys.exit(1)

print("✅ 向量库已存在，开始测试...")

sys.path.insert(0, str(BASE_DIR))
from desktop_app import SimpleVectorStore
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")
vectorstore = SimpleVectorStore(str(db_dir), embeddings)

print(f"✅ 向量库加载完成，共 {len(vectorstore)} 个文档\n")

# 测试查询
test_query = "工业管道的基本识别色有哪些"
print(f"🔍 测试查询: {test_query}\n")
print("="*60)

# 使用混合检索
start = time.time()
results = vectorstore.enhanced_similarity_search(test_query, k=3)
hybrid_time = time.time() - start

print(f"\n⏱️ 混合检索耗时: {hybrid_time:.2f}秒\n")

for i, doc in enumerate(results):
    page_info = doc.get('metadata', {}).get('page', 'N/A')
    print(f"\n--- 结果 {i+1} (来自第{page_info}页) ---")
    print(doc['page_content'][:300] + "...")

print("\n" + "="*60)
print("✅ 测试完成！")
