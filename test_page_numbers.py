
"""
测试页码功能是否正常
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
print(f"📄 使用教材: {pdf_path.name}")

# 1. 测试PDF页码读取
print("\n" + "="*60)
print("1. 测试PDF页码读取...")
import pdfplumber
with pdfplumber.open(pdf_path) as pdf:
    print(f"   总页数: {len(pdf.pages)}")
    for page_num, page in enumerate(pdf.pages, 1):
        if page_num > 160 and page_num < 165:  # 看看我们知道有颜色信息的那几页
            text = page.extract_text()
            if '基本识别色' in text or '颜色' in text:
                print(f"\n✅ 第{page_num}页找到关键字:")
                print(text[:500])
                break

# 2. 加载并测试主程序
print("\n" + "="*60)
print("2. 测试加载教材并创建带页码的向量库...")

# 先修改sys.path，让我们能导入desktop_app
sys.path.insert(0, str(BASE_DIR))

try:
    from desktop_app import SimpleVectorStore, PDFPlumberLoader
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    OLLAMA_URL = "http://localhost:11434"
    EMBED_MODEL = "mxbai-embed-large"
    
    # 加载PDF，保存页码
    print("   加载PDF文件...")
    documents = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:
                doc = type('', (), {})()
                doc.page_content = text
                doc.metadata = {'page': page_num, 'source': str(pdf_path)}
                documents.append(doc)
    print(f"   ✅ 加载了 {len(documents)} 页")
    
    # 分割文本
    print("   分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"   ✅ 分割为 {len(chunks)} 个文本块")
    
    # 检查前几个文档是否有页码
    print("   检查前5个文档的元数据:")
    for i, chunk in enumerate(chunks[:5]):
        print(f"   文档{i+1}: page={chunk.metadata.get('page', 'N/A')}")
    
    # 创建向量库
    print("\n   创建向量库...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    file_stem = pdf_path.stem
    db_dir = VECTOR_DB_DIR / file_stem
    
    # 先确保是干净的
    import shutil
    if db_dir.exists():
        shutil.rmtree(db_dir)
    
    vectorstore = SimpleVectorStore(str(db_dir), embeddings)
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        batch_texts = [c.page_content for c in batch]
        batch_metadatas = [c.metadata for c in batch]
        vectorstore.add_texts(batch_texts, batch_metadatas)
        if i % 100 == 0:
            print(f"   已处理 {i}/{len(chunks)} 个文本块")
    
    print(f"\n✅ 向量库创建成功，共 {len(vectorstore)} 个向量")
    
    # 检查向量库内容
    print("\n" + "="*60)
    print("3. 检查向量库中的文档是否包含页码...")
    
    # 查找包含"基本识别色"的文档
    found = False
    for i, doc in enumerate(vectorstore.documents):
        if isinstance(doc, dict):
            content = doc.get('page_content', '')
            metadata = doc.get('metadata', {})
        else:
            content = ''
            metadata = {}
        
        if '基本识别色' in content or '艳绿色' in content:
            print(f"\n✅ 找到相关内容在文档 {i}:")
            print(f"   页码: {metadata.get('page', 'N/A')}")
            print(f"   内容: {content[:300]}")
            found = True
            if i > 0:
                break
    
    if not found:
        print("❌ 未在向量库中找到相关内容")
    
    print("\n" + "="*60)
    print("✅ 页码功能测试完成！")
    print("现在可以运行 python desktop_app.py 测试完整程序了！")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
