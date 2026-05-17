# 一建机电教材RAG系统 (MVP)

基于Ollama本地大模型的一级建造师机电工程教材智能学习助手。

## 🚀 快速开始

### 前置要求

1. **安装Ollama**
   - 访问 https://ollama.com/download 下载Windows版本
   - 安装完成后，确保Ollama服务正在运行

2. **下载模型**
   - 打开终端运行以下命令：
   ```bash
   ollama pull qwen2.5:7b
   ollama pull mxbai-embed-large
   ```

3. **安装Python依赖**
   ```bash
   cd 一建机电RAG
   pip install -r requirements.txt
   ```

## 📖 使用说明

### 启动系统

```bash
streamlit run app.py
```

浏览器会自动打开 http://localhost:8501

### 使用步骤

1. **启动Ollama服务**
   - 打开终端，运行：`ollama serve`

2. **打开应用**
   - 运行：`streamlit run app.py`

3. **上传教材**
   - 点击「上传教材」标签
   - 选择一建机电教材PDF
   - 点击「开始处理教材」按钮

4. **开始问答**
   - 切换到「教材问答」标签
   - 在输入框中提问
   - 等待AI回复（CPU模式可能需要15-45秒）

## 📦 项目结构

```
一建机电RAG/
├── app.py                # Streamlit主界面
├── core_rag.py           # 核心RAG逻辑
├── requirements.txt      # 依赖清单
├── .env                  # 配置文件
├── data/
│   ├── uploads/          # 上传的教材文件
│   └── chroma_db/        # ChromaDB向量数据
└── README.md             # 本文件
```

## ⚙️ 配置说明

编辑 `.env` 文件可以调整参数：

```env
# Ollama配置
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=qwen2.5:7b
OLLAMA_EMBED_MODEL=mxbai-embed-large

# RAG参数
CHUNK_SIZE=600          # 分块大小
CHUNK_OVERLAP=100       # 重叠大小
TOP_K=3                # 检索数量
```

## 💡 使用建议

### 针对12GB内存优化

- 首次启动会有些慢，后续会快很多
- 建议关闭其他占用内存的程序
- 可以考虑增加虚拟内存（Page File）

### 问答技巧

- 问题越具体，回答越准确
- 可以追问相关知识点
- 查看引用来源确保准确性

## 🎯 核心功能

- ✅ PDF教材解析
- ✅ 智能分块向量化
- ✅ 语义检索
- ✅ 专业问答
- ✅ 引用来源展示
- ✅ 聊天历史记录

## 📊 性能说明

### 硬件配置

- **内存**: 建议12GB+（您的配置刚好够用）
- **CPU**: 6核或以上
- **显卡**: 集成显卡（使用CPU推理）

### 预估时间

| 操作 | 预估时间 |
|------|---------|
| 模型加载 | 2-5分钟（首次） |
| 教材处理（300页） | 5-15分钟 |
| 单次问答 | 15-45秒（CPU） |

## ❓ 常见问题

### Q: 提示Ollama服务未启动
A: 打开新终端运行 `ollama serve`

### Q: 问答很慢
A: CPU推理模式确实较慢，请耐心等待，这是正常现象

### Q: 可以处理其他教材吗？
A: 可以，支持任意PDF教材

### Q: 数据安全吗？
A: 完全本地运行，所有数据都在您的电脑上

## 📝 更新日志

### v1.0 (MVP) - 2026-05-16
- 基础RAG功能
- PDF解析
- 简单界面
- 本地模型支持

---

祝您备考顺利！📚✨
