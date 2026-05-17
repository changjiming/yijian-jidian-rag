"""
一建机电教材RAG系统 - Streamlit界面
"""
import streamlit as st
import os
import sys
from pathlib import Path

# 导入核心模块
sys.path.insert(0, str(Path(__file__).parent))
from core_rag import TextbookRAG

# 页面配置
st.set_page_config(
    page_title="一建机电教材学习助手",
    page_icon="📚",
    layout="wide"
)

# 初始化状态
if "rag_system" not in st.session_state:
    st.session_state.rag_system = TextbookRAG()
    st.session_state.knowledge_loaded = False
    st.session_state.chat_history = []

# 页面标题
st.title("📚 一建机电教材智能学习助手")
st.divider()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 系统配置")
    
    # 检查Ollama服务
    st.subheader("服务状态")
    try:
        # 简单测试Ollama连接
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            st.success("✅ Ollama服务正常")
        else:
            st.error("❌ Ollama服务异常")
    except Exception as e:
        st.error("❌ Ollama服务未启动")
        st.warning("请先启动Ollama服务: `ollama serve`")
    
    st.divider()
    st.subheader("📝 操作指南")
    st.markdown("""
    1. **启动Ollama**: 打开终端运行 `ollama serve`
    2. **上传教材**: 点击左侧上传按钮
    3. **提问**: 在右侧输入问题开始学习
    """)
    
    st.divider()
    st.subheader("📦 模型信息")
    st.info("""
    **语言模型**: qwen2.5:7b  
    **向量化**: mxbai-embed-large
    """)
    
    st.divider()
    st.caption("一建机电RAG系统 v1.0 (MVP)")

# 主界面
tab1, tab2 = st.tabs(["📖 教材问答", "📤 上传教材"])

# 教材问答页面
with tab1:
    st.header("🤖 与教材对话")
    
    if not st.session_state.knowledge_loaded:
        st.warning("⚠️ 请先上传教材！请切换到「上传教材」标签页。")
    else:
        st.success("✅ 知识库已加载，可以开始提问！")
        
        # 聊天界面
        if st.session_state.chat_history:
            for role, message in st.session_state.chat_history:
                if role == "user":
                    with st.chat_message("user"):
                        st.markdown(message)
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message)
        
        # 输入框
        if question := st.chat_input("请输入您的问题，例如：什么是工业管道安装的基本要求？"):
            # 用户消息
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.markdown(question)
            
            # AI回答
            with st.chat_message("assistant"):
                with st.spinner("正在分析教材，请稍候...（CPU模式可能需要15-45秒）"):
                    try:
                        result = st.session_state.rag_system.ask(question)
                        
                        # 显示回答
                        answer = result["answer"]
                        st.markdown(answer)
                        
                        # 显示来源
                        with st.expander("📖 查看引用来源"):
                            for source in result["sources"]:
                                st.markdown(f"**来源{source['rank']}**:")
                                st.info(source["content"])
                        
                        # 保存到历史
                        st.session_state.chat_history.append(("assistant", answer))
                        
                    except Exception as e:
                        st.error(f"问答失败: {e}")
                        st.info("请确保已上传教材且Ollama服务正常运行")

# 上传教材页面
with tab2:
    st.header("📤 上传一建机电教材")
    
    uploaded_file = st.file_uploader(
        "选择教材PDF文件",
        type=["pdf"],
        help="支持PDF格式的一建机电教材"
    )
    
    if uploaded_file:
        st.success(f"✅ 已选择文件: {uploaded_file.name}")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🚀 开始处理教材", type="primary", use_container_width=True):
                # 保存文件
                save_dir = "./data/uploads"
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # 处理教材
                with st.spinner("正在处理教材，请耐心等待...（根据文件大小可能需要5-15分钟）"):
                    try:
                        st.info("步骤1/4: 解析PDF...")
                        success = st.session_state.rag_system.process_textbook(file_path)
                        
                        if success:
                            st.session_state.knowledge_loaded = True
                            st.success("🎉 教材处理完成！现在可以切换到「教材问答」标签页开始学习！")
                        
                    except Exception as e:
                        st.error(f"处理失败: {e}")
                        st.info("请确保Ollama服务正常运行，且已下载所需模型")
        
        with col2:
            st.info("💡 提示：首次上传需要较长时间，请耐心等待")
    
    # 快速测试（示例问题）
    st.divider()
    st.subheader("💬 示例问题")
    example_questions = [
        "什么是工业管道安装的基本要求？",
        "电气设备安装有哪些注意事项？",
        "机械设备安装的一般程序是什么？",
        "压力容器的分类标准是什么？"
    ]
    
    for q in example_questions:
        if st.button(q, key=f"example_{q}"):
            if st.session_state.knowledge_loaded:
                st.session_state.chat_history.append(("user", q))
                st.rerun()
            else:
                st.warning("请先上传教材！")

# 页面底部提示
st.divider()
st.caption("📝 提示：这是MVP版本，后续会增加更多功能！")
