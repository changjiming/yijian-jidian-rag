
# 检查向量库中是否有管道识别色内容
from pathlib import Path
import pickle

db_dir = Path(__file__).parent / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"

with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    docs = data['documents']

# 搜索关键词
keywords = ["艳绿色", "大红色", "淡灰色", "中黄色", "淡蓝色", "基本识别色", "识别符号", "危险标识", "水蒸气是大红色"]

print("搜索向量库中的内容：\n")

for keyword in keywords:
    found = False
    for i, doc in enumerate(docs):
        content = doc['page_content']
        if keyword in content:
            if not found:
                print(f"'{keyword}' 找到在文档 {i}:")
                print(f"  {content[:300]}...\n")
                found = True
    if not found:
        print(f"'{keyword}': ❌ 未找到")
