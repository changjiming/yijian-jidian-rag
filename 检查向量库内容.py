
# 检查向量库内容
from pathlib import Path
import pickle
import json

BASE_DIR = Path(__file__).parent
db_dir = BASE_DIR / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"

vectors_file = db_dir / "vectors.pkl"

if vectors_file.exists():
    with open(vectors_file, 'rb') as f:
        data = pickle.load(f)
        docs = data['documents']
    
    print(f"向量库共有 {len(docs)} 个文档\n")
    
    print("=" * 60)
    print("搜索包含'管道'关键词的文档：")
    print("=" * 60)
    
    for i, doc in enumerate(docs):
        content = doc['page_content']
        if '管道' in content:
            print(f"\n[文档 {i}] {content[:200]}...")
            if i > 20:
                print("\n... 还有更多包含'管道'的文档")
                break
    
    print("\n" + "=" * 60)
    print("搜索包含'识别'关键词的文档：")
    print("=" * 60)
    
    for i, doc in enumerate(docs):
        content = doc['page_content']
        if '识别' in content:
            print(f"\n[文档 {i}] {content[:200]}...")
            if i > 20:
                print("\n... 还有更多包含'识别'的文档")
                break
    
    print("\n" + "=" * 60)
    print("随机采样10个文档：")
    print("=" * 60)
    
    import random
    samples = random.sample(range(len(docs)), min(10, len(docs)))
    for i in samples:
        doc = docs[i]
        print(f"\n[文档 {i}] {doc['page_content'][:200]}...")
else:
    print("向量库文件不存在！")
