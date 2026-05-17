
# 检查所有RAG依赖的导入
import sys

print("=" * 60)
print("  检查RAG依赖导入")
print("=" * 60)

modules = [
    ("langchain_community.document_loaders", "PDFPlumberLoader"),
    ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
    ("langchain_ollama", "ChatOllama"),
    ("langchain_ollama", "OllamaEmbeddings"),
    ("langchain_community.vectorstores", "Chroma"),
    ("langchain.chains", "RetrievalQA"),
    ("langchain.prompts", "PromptTemplate"),
    ("langchain_community.document_loaders", "TextLoader"),
]

all_ok = True
for module_name, class_name in modules:
    try:
        module = __import__(module_name, fromlist=[class_name])
        obj = getattr(module, class_name)
        print(f"✅ {module_name}.{class_name} - OK")
    except Exception as e:
        print(f"❌ {module_name}.{class_name} - 失败: {e}")
        all_ok = False

print("=" * 60)
if all_ok:
    print("🎉 所有依赖导入成功！")
else:
    print("⚠️ 部分依赖导入失败，请检查错误信息")
