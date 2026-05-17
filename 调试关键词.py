
# 调试关键词匹配
from pathlib import Path
import pickle

db_dir = Path(__file__).parent / "data" / "vector_db_simple" / "2026冬阳一建机电PDF教材(1)"

with open(db_dir / "vectors.pkl", 'rb') as f:
    data = pickle.load(f)
    documents = data['documents']

query = "工业管道的基本识别色有哪些"
query_words = set(query.replace("？", "").replace("?", "").replace("，", " ").replace(",", " ").split())

print(f"查询词: {query_words}\n")

# 检查文档558
doc = documents[558]
content = doc['page_content']
print(f"文档558内容片段:")
print(content[:500])
print("\n" + "="*60)

print("\n文档558关键词匹配情况:")
for word in query_words:
    if len(word) >= 2:
        if word in content:
            print(f"  '{word}' ✅ 匹配")
        else:
            print(f"  '{word}' ❌ 不匹配")

# 检查所有包含"基本识别色"的文档
print("\n\n所有包含'基本识别色'的文档:")
for i, doc in enumerate(documents):
    if '基本识别色' in doc['page_content']:
        print(f"  文档{i}")
