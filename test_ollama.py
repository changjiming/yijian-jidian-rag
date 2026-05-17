
import requests
import json

print("=" * 50)
print("  Ollama 连接测试")
print("=" * 50)
print()

OLLAMA_URL = "http://localhost:11434"

# 测试1: 检查API连接
print("[1/4] 检查Ollama服务...")
try:
    response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
    if response.status_code == 200:
        print("✅ API连接成功！")
        print()
    else:
        print(f"❌ API返回错误: {response.status_code}")
        exit(1)
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("请确保Ollama服务正在运行（运行 'ollama serve'）")
    exit(1)

# 测试2: 查看已安装模型
print("[2/4] 已安装的模型：")
models = response.json().get("models", [])
for model in models:
    print(f"  - {model['name']} ({model.get('size', 'N/A')})")
print()

# 测试3: 检查需要的模型
print("[3/4] 检查所需模型...")
required_models = ["qwen2.5:7b", "mxbai-embed-large"]
for model_name in required_models:
    found = any(m["name"] == model_name or m["name"].startswith(model_name) for m in models)
    if found:
        print(f"✅ {model_name} - 已安装")
    else:
        print(f"❌ {model_name} - 未安装")
print()

# 测试4: 简单测试LLM
print("[4/4] 测试语言模型...")
try:
    payload = {
        "model": "qwen2.5:7b",
        "prompt": "你好",
        "stream": False
    }
    response = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=30)
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 语言模型测试成功！")
        print(f"回答: {result.get('response', '无响应')[:100]}...")
    else:
        print(f"❌ 测试失败: {response.status_code}")
except Exception as e:
    print(f"❌ 测试出错: {e}")

print()
print("=" * 50)
print("测试完成！")
print("=" * 50)
