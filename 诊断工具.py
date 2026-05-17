
# 系统诊断工具
import os
import sys
from pathlib import Path
import requests
import json
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def check_system():
    log("=" * 60)
    log("系统诊断工具")
    log("=" * 60)
    
    # 1. 检查目录结构
    log("\n[1/5] 检查目录结构...")
    base_dir = Path(__file__).parent
    dirs = {
        "项目根目录": base_dir,
        "数据目录": base_dir / "data",
        "上传目录": base_dir / "data" / "uploads",
        "向量库目录": base_dir / "data" / "chroma_db"
    }
    
    for name, path in dirs.items():
        if path.exists():
            log(f"  ✅ {name}: {path}")
        else:
            log(f"  ❌ {name} 不存在: {path}")
    
    # 2. 检查上传的教材
    log("\n[2/5] 检查上传的教材...")
    upload_dir = dirs["上传目录"]
    if upload_dir.exists():
        files = list(upload_dir.glob("*"))
        if files:
            for f in files:
                size_mb = f.stat().st_size / (1024 * 1024)
                log(f"  📄 {f.name} ({size_mb:.2f} MB)")
        else:
            log("  ⚠️ 上传目录为空")
    else:
        log("  ❌ 上传目录不存在")
    
    # 3. 检查向量库
    log("\n[3/5] 检查向量库...")
    vector_dir = dirs["向量库目录"]
    if vector_dir.exists():
        subdirs = [d for d in vector_dir.iterdir() if d.is_dir()]
        if subdirs:
            log(f"  找到 {len(subdirs)} 个向量库:")
            for d in subdirs:
                files_count = len(list(d.rglob("*")))
                log(f"    📦 {d.name}/ ({files_count} 个文件)")
        else:
            log("  ⚠️ 向量库目录为空（还没有处理过教材）")
    else:
        log("  ❌ 向量库目录不存在")
    
    # 4. 检查Ollama服务
    log("\n[4/5] 检查Ollama服务...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            log(f"  ✅ Ollama服务正常")
            log(f"  已安装 {len(models)} 个模型:")
            for m in models:
                size_mb = m.get("size", 0) / (1024 * 1024 * 1024)
                log(f"    - {m['name']} ({size_mb:.2f} GB)")
        else:
            log(f"  ❌ Ollama服务异常: {response.status_code}")
    except Exception as e:
        log(f"  ❌ 无法连接Ollama: {e}")
    
    # 5. 检查依赖
    log("\n[5/5] 检查Python依赖...")
    dependencies = [
        "langchain_community",
        "langchain_text_splitters",
        "langchain_ollama",
        "chromadb",
        "pdfplumber"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep)
            log(f"  ✅ {dep}")
        except ImportError:
            log(f"  ❌ {dep} 未安装")
    
    log("\n" + "=" * 60)
    log("诊断完成！")
    log("=" * 60)

if __name__ == "__main__":
    check_system()
