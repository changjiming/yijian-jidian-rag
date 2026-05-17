"""
一建机电教材RAG系统 - 核心模块
"""
import os
from dotenv import load_dotenv
from typing import List, Dict
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 加载环境变量
load_dotenv()

# 配置参数
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "qwen2.5:7b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
TOP_K = int(os.getenv("TOP_K", 3))

# 一建机电教材专用Prompt
TEXTBOOK_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""你是一位一级建造师机电工程专业的辅导老师。请根据教材内容回答学员的问题。

【重要原则】
1. 只基于提供的教材内容回答，不要编造信息
2. 如果教材中没有相关信息，明确说明
3. 回答要专业、准确，符合一建考试要求
4. 可以引用具体的教材章节和页码
5. 对于重要的考点，适当提醒学员

【教材内容】
{context}

【学员问题】
{question}

【回答格式】
1. 直接回答问题
2. 引用相关教材内容
3. 标注相关考点（如果有）
"""
)


class TextbookRAG:
    """一建机电教材RAG系统"""
    
    def __init__(self):
        self.llm = ChatOllama(
            model=OLLAMA_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            temperature=0.3,
            num_ctx=4096
        )
        self.embeddings = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self.vectorstore = None
        self.qa_chain = None
    
    def parse_pdf(self, file_path: str) -> List[str]:
        """解析PDF教材"""
        text_chunks = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_chunks.append(text)
        except Exception as e:
            print(f"PDF解析错误: {e}")
        
        return text_chunks
    
    def split_text(self, texts: List[str]) -> List[str]:
        """文本分块"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )
        
        chunks = []
        for text in texts:
            if text:
                split_chunks = text_splitter.split_text(text)
                chunks.extend(split_chunks)
        
        return chunks
    
    def create_vectorstore(self, chunks: List[str], collection_name: str = "textbook"):
        """创建向量库"""
        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_DB_PATH,
            collection_name=collection_name
        )
        return self.vectorstore
    
    def load_vectorstore(self, collection_name: str = "textbook"):
        """加载向量库"""
        try:
            self.vectorstore = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )
            return True
        except Exception as e:
            print(f"加载向量库失败: {e}")
            return False
    
    def build_qa_chain(self):
        """构建问答链"""
        if not self.vectorstore:
            raise ValueError("请先加载或创建向量库")
        
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": TOP_K}
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": TEXTBOOK_QA_PROMPT}
        )
        
        return self.qa_chain
    
    def ask(self, question: str) -> Dict:
        """问答"""
        if not self.qa_chain:
            self.build_qa_chain()
        
        result = self.qa_chain({"query": question})
        
        answer = result["result"]
        sources = result["source_documents"]
        
        # 格式化来源
        formatted_sources = []
        for i, doc in enumerate(sources, 1):
            source_text = doc.page_content
            if len(source_text) > 200:
                source_text = source_text[:200] + "..."
            
            formatted_sources.append({
                "rank": i,
                "content": source_text,
                "metadata": doc.metadata
            })
        
        return {
            "answer": answer,
            "sources": formatted_sources
        }
    
    def process_textbook(self, pdf_path: str, collection_name: str = "textbook") -> bool:
        """完整处理流程"""
        print(f"正在解析教材: {pdf_path}")
        texts = self.parse_pdf(pdf_path)
        
        if not texts:
            print("解析失败，未获取到文本内容")
            return False
        
        print(f"解析完成，共 {len(texts)} 页")
        
        print("正在分块...")
        chunks = self.split_text(texts)
        print(f"分块完成，共 {len(chunks)} 个块")
        
        print("正在向量化...")
        self.create_vectorstore(chunks, collection_name)
        
        print("正在构建问答链...")
        self.build_qa_chain()
        
        print("✅ 教材处理完成！")
        return True
