
# 检索测试工具 - 测试向量库是否能正确检索
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from pathlib import Path
import sys

OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "mxbai-embed-large"
VECTOR_DB_DIR = Path(__file__).parent / "data" / "chroma_db"
UPLOAD_DIR = Path(__file__).parent / "data" / "uploads"

def test_retrieval():
    print("=" * 60)
    print("检索测试工具")
    print("=" * 60)
    
    # 1. 找到上传的教材
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        print("❌ 没有找到PDF教材！")
        return
    
    pdf_path = pdf_files[0]
    print(f"\n📄 测试教材: {pdf_path.name}")
    
    # 2. 加载向量库
    pdf_stem = pdf_path.stem
    db_dir = VECTOR_DB_DIR / pdf_stem
    
    if not db_dir.exists():
        print(f"❌ 向量库不存在: {db_dir}")
        print("请先在程序中处理这本教材")
        return
    
    print(f"✅ 找到向量库: {db_dir}")
    
    # 3. 加载向量库
    print("\n加载向量库...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    vectorstore = Chroma(
        persist_directory=str(db_dir),
        embedding_function=embeddings
    )
    
    # 4. 测试检索
    test_questions = [
        "工业管道的基本识别色有哪些",
        "管道工程",
        "焊接",
        "机电安装"
    ]
    
    print("\n" + "=" * 60)
    print("开始检索测试")
    print("=" * 60)
    
    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)
        
        # 检索相关文档
        docs = vectorstore.similarity_search(question, k=2)
        
        if docs:
            print(f"✅ 找到 {len(docs)} 个相关文档")
            for i, doc in enumerate(docs, 1):
                content = doc.page_content[:200]  # 只显示前200字
                print(f"\n  文档 {i}:")
                print(f"  {content}...")
                print(f"  (共 {len(doc.page_content)} 字符)")
        else:
            print("❌ 没有找到相关文档")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_retrieval()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
