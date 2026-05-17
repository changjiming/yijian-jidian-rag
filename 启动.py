
# 一建机电教材 RAG 问答系统 - 启动脚本
import os
import sys
import subprocess
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent

def check_ollama():
    print("=" * 50)
    print("   一建机电教材 RAG 问答系统")
    print("=" * 50)
    print()
    print("[1/3] 检查Ollama服务...")
    
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            print(f"[成功] Ollama服务运行中，已安装模型: {models}")
            return True
    except:
        pass
    
    print("[提示] 正在尝试启动Ollama服务...")
    try:
        subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        time.sleep(3)
        print("[成功] Ollama服务已启动")
        return True
    except Exception as e:
        print(f"[失败] 无法启动Ollama: {e}")
        return False

def check_dependencies():
    print()
    print("[2/3] 检查Python依赖...")
    try:
        import langchain
        import langchain_community
        import langchain_ollama
        import chromadb
        import pdfplumber
        print("[成功] 所有依赖已安装")
        return True
    except ImportError:
        print("[提示] 正在安装依赖...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", 
                                str(BASE_DIR / "requirements.txt")])
            print("[成功] 依赖安装完成")
            return True
        except Exception as e:
            print(f"[失败] 安装依赖时出错: {e}")
            return False

def start_app():
    print()
    print("[3/3] 启动桌面应用...")
    print()
    print("=" * 50)
    print("  应用已启动！请使用弹出的窗口")
    print("=" * 50)
    
    os.chdir(str(BASE_DIR))
    os.system(f'"{sys.executable}" "{str(BASE_DIR / "desktop_app.py")}"')

if __name__ == "__main__":
    if not check_ollama():
        print()
        print("请先安装并启动Ollama: https://ollama.com")
        input("按回车键退出...")
        sys.exit(1)
    
    if not check_dependencies():
        print()
        input("按回车键继续尝试启动...")
    
    try:
        start_app()
    except Exception as e:
        print(f"[错误] 启动应用失败: {e}")
        input("按回车键退出...")

