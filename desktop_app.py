
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import requests
import time
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

def log(msg, type="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{type}] {msg}")
    sys.stdout.flush()

log("=" * 60)
log("一建机电教材 RAG 问答系统 启动中...", "START")
log("=" * 60)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTOR_DB_DIR = DATA_DIR / "vector_db_simple"
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "mxbai-embed-large"

# 主题配色
COLORS = {
    "primary": "#1E88E5",
    "primary_light": "#42A5F5",
    "primary_dark": "#1565C0",
    "success": "#43A047",
    "warning": "#FB8C00",
    "error": "#E53935",
    "info": "#00ACC1",
    "bg": "#F5F7FA",
    "bg_card": "#FFFFFF",
    "text_primary": "#212121",
    "text_secondary": "#757575",
    "text_hint": "#BDBDBD",
    "border": "#E0E0E0"
}

log("正在加载RAG依赖...", "INFO")
try:
    from langchain_community.document_loaders import PDFPlumberLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_community.document_loaders import TextLoader
    from agent import Agent
    HAS_RAG = True
    log("✅ 所有RAG依赖加载成功！", "SUCCESS")
except ImportError as e:
    HAS_RAG = False
    log(f"❌ RAG依赖加载失败: {e}", "ERROR")

class SimpleVectorStore:
    def __init__(self, persist_dir: str, embeddings):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_file = self.persist_dir / "vectors.pkl"
        self.metadata_file = self.persist_dir / "metadata.json"
        self.embeddings = embeddings
        self.vectors = []
        self.documents = []
        self._load()
    
    def _load(self):
        if self.vectors_file.exists():
            with open(self.vectors_file, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                docs = data.get('documents', [])
                self.documents = []
                for doc in docs:
                    if isinstance(doc, dict):
                        self.documents.append(doc)
                    elif hasattr(doc, 'page_content'):
                        self.documents.append({
                            'page_content': doc.page_content,
                            'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                        })
                    else:
                        self.documents.append({'page_content': str(doc), 'metadata': {}})
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def _save(self):
        docs_to_save = []
        for doc in self.documents:
            if isinstance(doc, dict):
                docs_to_save.append(doc)
            elif hasattr(doc, 'page_content'):
                docs_to_save.append({
                    'page_content': doc.page_content,
                    'metadata': doc.metadata if hasattr(doc, 'metadata') else {}
                })
            else:
                docs_to_save.append({'page_content': str(doc), 'metadata': {}})
        
        with open(self.vectors_file, 'wb') as f:
            pickle.dump({
                'vectors': self.vectors,
                'documents': docs_to_save
            }, f)
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False)
    
    def add_texts(self, texts: List[str], metadatas: Optional[List] = None):
        batch_embeddings = self.embeddings.embed_documents(texts)
        for i, text in enumerate(texts):
            self.documents.append({'page_content': text, 'metadata': metadatas[i] if metadatas else {}})
            self.vectors.append(batch_embeddings[i])
        self._save()
    
    def similarity_search(self, query: str, k: int = 4) -> List:
        if not self.vectors:
            return []
        query_vec = self.embeddings.embed_query(query)
        query_vec = np.array(query_vec)
        vectors = np.array(self.vectors)
        similarities = np.dot(vectors, query_vec) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vec) + 1e-10)
        top_k = np.argsort(similarities)[-k:][::-1]
        return [self.documents[i] for i in top_k]
    
    def enhanced_similarity_search(self, query: str, k: int = 4) -> List:
        hybrid_results = self.hybrid_search(query, k)
        results = []
        seen_indices = set()
        for idx, score in hybrid_results:
            doc = self.documents[idx]
            if isinstance(doc, dict):
                content = doc.get('page_content', '')
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
            
            merged_content = content
            for next_idx in [idx+1, idx-1]:
                if 0 <= next_idx < len(self.documents):
                    next_doc = self.documents[next_idx]
                    if isinstance(next_doc, dict):
                        next_content = next_doc.get('page_content', '')
                    elif hasattr(next_doc, 'page_content'):
                        next_content = next_doc.page_content
                    else:
                        next_content = str(next_doc)
                        
                    if len(next_content) + len(merged_content) < 2000:
                        if any(kw in next_content for kw in ['艳绿色', '大红色', '淡灰色', '中黄色', '淡蓝色', '基本识别色', '识别符号']):
                            merged_content += "\n" + next_content
                            if next_idx not in seen_indices:
                                seen_indices.add(next_idx)
            
            if idx not in seen_indices:
                seen_indices.add(idx)
                if isinstance(doc, dict):
                    metadata = doc.get('metadata', {})
                elif hasattr(doc, 'metadata'):
                    metadata = doc.metadata
                else:
                    metadata = {}
                    
                results.append({
                    'page_content': merged_content, 
                    'metadata': metadata
                })
            
            if len(results) >= k:
                break
        
        return results
    
    def __len__(self):
        return len(self.vectors)
    
    def _extract_keywords(self, text):
        keywords = set()
        n = 2
        while n <= 4:
            for i in range(len(text) - n + 1):
                word = text[i:i+n]
                if len(word) == n:
                    keywords.add(word)
            n += 1
        return keywords
    
    def _compute_bm25_score(self, query_keywords):
        scores = {}
        avg_doc_len = 1
        if len(self.documents) > 0:
            total_len = sum(len(d.get('page_content', '') if isinstance(d, dict) 
                                 else (d.page_content if hasattr(d, 'page_content') else str(d))) 
                           for d in self.documents)
            if total_len > 0:
                avg_doc_len = total_len / len(self.documents)
        
        k1 = 1.5
        b = 0.75
        
        for doc_idx, doc in enumerate(self.documents):
            if isinstance(doc, dict):
                content = doc.get('page_content', '')
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
            
            doc_len = len(content) if content else 1
            
            score = 0
            for keyword in query_keywords:
                if keyword in content:
                    tf = content.count(keyword)
                    df = sum(1 for d in self.documents if keyword in 
                            (d.get('page_content', '') if isinstance(d, dict) 
                             else (d.page_content if hasattr(d, 'page_content') else str(d))))
                    
                    idf = (len(self.documents) - df + 0.5) / (df + 0.5)
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                    score += idf * (numerator / denominator)
            
            scores[doc_idx] = score
        
        return scores
    
    def hybrid_search(self, query, k=4):
        if not self.documents or not self.vectors:
            return []
        
        query_keywords = self._extract_keywords(query)
        
        keyword_scores = {}
        for i, doc in enumerate(self.documents):
            if isinstance(doc, dict):
                content = doc.get('page_content', '')
            elif hasattr(doc, 'page_content'):
                content = doc.page_content
            else:
                content = str(doc)
            
            matched = query_keywords.intersection(self._extract_keywords(content))
            keyword_scores[i] = len(matched)
        
        bm25_scores = self._compute_bm25_score(query_keywords)
        
        query_vec = self.embeddings.embed_query(query)
        if not query_vec:
            return [(i, 0.0) for i in range(min(k*3, len(self.documents)))]
        
        query_vec = np.array(query_vec)
        vectors = np.array(self.vectors)
        
        vec_norms = np.linalg.norm(vectors, axis=1)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm < 1e-10:
            return [(i, 0.0) for i in range(min(k*3, len(self.documents)))]
        
        similarities = np.dot(vectors, query_vec) / (vec_norms * query_norm + 1e-10)
        
        final_scores = {}
        max_keyword = max(keyword_scores.values()) if keyword_scores else 1
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 1
        
        for i in range(len(self.documents)):
            keyword_normalized = keyword_scores.get(i, 0) / max_keyword if max_keyword > 0 else 0
            bm25_normalized = bm25_scores.get(i, 0) / max_bm25 if max_bm25 > 0 else 0
            vector_score = similarities[i] if i < len(similarities) else 0
            
            final_scores[i] = (
                0.3 * keyword_normalized + 
                0.4 * bm25_normalized + 
                0.3 * vector_score
            )
        
        sorted_indices = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:k*3]
        return sorted_indices

class ChatHistory:
    def __init__(self, save_dir: str = None):
        if save_dir is None:
            save_dir = DATA_DIR / "chat_history"
        else:
            save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.save_path = save_dir / "history.json"
        self.current_conversation: List[Dict] = []
        self.all_conversations: List[Dict] = []
        self._load()
    
    def _load(self):
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    self.all_conversations = json.load(f)
            except Exception:
                self.all_conversations = []
    
    def _save(self):
        with open(self.save_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_conversations, f, ensure_ascii=False, indent=2)
    
    def start_new_conversation(self):
        if self.current_conversation:
            self.all_conversations.insert(0, {
                'timestamp': datetime.now().isoformat(),
                'messages': self.current_conversation
            })
            self._save()
        self.current_conversation = []
    
    def add_message(self, question: str, answer: str, sources: List[str] = None):
        self.current_conversation.append({
            'question': question,
            'answer': answer,
            'sources': sources or [],
            'time': datetime.now().strftime("%H:%M:%S")
        })
        self._save()
    
    def get_recent_conversations(self, count: int = 20) -> List[Dict]:
        return self.all_conversations[:count]
    
    def get_current_conversation(self) -> List[Dict]:
        return self.current_conversation
    
    def load_conversation_to_current(self, conversation: Dict):
        self.current_conversation = conversation.get('messages', [])
    
    def clear_history(self):
        self.all_conversations = []
        self.current_conversation = []
        self._save()
    
    def search_history(self, keyword: str) -> List[Dict]:
        results = []
        for conv in self.all_conversations:
            for msg in conv.get('messages', []):
                if keyword in msg.get('question', '') or keyword in msg.get('answer', ''):
                    results.append({
                        'conversation': conv,
                        'message': msg
                    })
                    break
        return results

class ModernButton(ttk.Button):
    def __init__(self, parent, text="", command=None, style="Modern.TButton", **kwargs):
        super().__init__(parent, text=text, command=command, style=style, **kwargs)

class ConversationCard:
    def __init__(self, parent, is_user=False):
        self.parent = parent
        self.is_user = is_user
        self.frame = ttk.Frame(parent, style="Card.TFrame")
        
        if is_user:
            self.frame.pack(anchor='e', padx=10, pady=5, fill='x')
            self.bubble = ttk.Label(self.frame, style="UserBubble.TLabel")
        else:
            self.frame.pack(anchor='w', padx=10, pady=5, fill='x')
            self.bubble = ttk.Label(self.frame, style="AIBubble.TLabel")
        
        self.bubble.pack(padx=15, pady=8, fill='x')
    
    def set_text(self, text):
        self.bubble.config(text=text)
    
    def set_sources(self, sources):
        if sources:
            source_text = f"\n\n📚 参考来源: {', '.join(sources)}"
            current = self.bubble.cget('text')
            self.bubble.config(text=current + source_text)

class ContractRAGApp:
    def __init__(self, root):
        log("初始化GUI界面...", "INFO")
        self.root = root
        self.root.title("📚 一建机电教材 RAG 问答系统")
        self.root.geometry("1100x750")
        self.root.minsize(900, 600)
        self.root.configure(bg=COLORS["bg"])
        
        self.current_file = None
        self.chat_history = ChatHistory()
        self.agent = None
        self.vector_store = None
        self.use_agent_mode = False
        
        self.setup_styles()
        log("创建界面组件...", "INFO")
        self.create_widgets()
        log("检查Ollama服务...", "INFO")
        self.check_ollama()
        log("✅ 程序启动完成！", "SUCCESS")
    
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Main.TFrame', background=COLORS["bg"])
        style.configure('Card.TFrame', background=COLORS["bg_card"], relief='flat')
        style.configure('Header.TFrame', background=COLORS["primary"], relief='flat')
        
        style.configure('Modern.TButton',
                       background=COLORS["primary"],
                       foreground='white',
                       padding=(12, 6),
                       font=('Microsoft YaHei', 9))
        style.map('Modern.TButton',
                 background=[('active', COLORS["primary_light"]), ('pressed', COLORS["primary_dark"])],
                 relief=[('pressed', 'sunken'), ('!pressed', 'raised')])
        
        style.configure('Success.TButton',
                       background=COLORS["success"],
                       foreground='white',
                       padding=(12, 6),
                       font=('Microsoft YaHei', 9))
        style.map('Success.TButton',
                 background=[('active', '#66BB6A')])
        
        style.configure('Warning.TButton',
                       background=COLORS["warning"],
                       foreground='white',
                       padding=(12, 6),
                       font=('Microsoft YaHei', 9))
        
        style.configure('UserBubble.TLabel',
                       background=COLORS["primary"],
                       foreground='white',
                       padding=(15, 12),
                       font=('Microsoft YaHei', 10),
                       wraplength=500,
                       justify='left')
        
        style.configure('AIBubble.TLabel',
                       background=COLORS["bg_card"],
                       foreground=COLORS["text_primary"],
                       padding=(15, 12),
                       font=('Microsoft YaHei', 10),
                       wraplength=500,
                       justify='left',
                       relief='solid',
                       borderwidth=1)
        
        style.configure('Status.TLabel',
                       font=('Microsoft YaHei', 10),
                       padding=(5, 2),
                       background=COLORS["primary"],
                       foreground=COLORS["bg_card"])
        
        style.configure('Title.TLabel',
                       font=('Microsoft YaHei', 16, 'bold'),
                       foreground=COLORS["bg_card"],
                       background=COLORS["primary"])
        
        style.configure('Section.TLabel',
                       background=COLORS["bg_card"],
                       font=('Microsoft YaHei', 11, 'bold'),
                       foreground=COLORS["text_primary"])
        
        style.configure('Progress.TProgressbar',
                       troughcolor=COLORS["bg"],
                       background=COLORS["primary"])
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, style="Main.TFrame", padding="15")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=3)
        main_frame.rowconfigure(1, weight=1)
        
        header_frame = ttk.Frame(main_frame, style="Header.TFrame", padding="15")
        header_frame.grid(row=0, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(0, 15))
        header_frame.columnconfigure(1, weight=1)
        
        title_label = ttk.Label(header_frame, text="📚 一建机电教材 RAG 问答系统", style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        self.status_label = ttk.Label(header_frame, text="🔍 检查Ollama服务中...", 
                                     style="Status.TLabel", foreground="orange")
        self.status_label.grid(row=1, column=0, sticky=tk.W, pady=(5, 0))
        
        btn_frame = ttk.Frame(header_frame, style="Main.TFrame")
        btn_frame.grid(row=0, column=1, sticky=tk.E)
        
        ttk.Button(btn_frame, text="📖 上传教材", command=self.upload_file, 
                   style="Modern.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="🔄 清空对话", command=self.clear_chat, 
                   style="Modern.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="🆕 新对话", command=self.start_new_chat, 
                   style="Modern.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="📜 历史记录", command=self.toggle_history_panel, 
                   style="Modern.TButton").pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="🤖 Agent模式", command=self.toggle_agent_mode, 
                   style="Success.TButton").pack(side=tk.LEFT, padx=8)
        
        left_panel = ttk.Frame(main_frame, style="Card.TFrame", padding="10")
        left_panel.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)
        
        book_label = ttk.Label(left_panel, text="📂 教材文件", style="Section.TLabel")
        book_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.file_list = tk.Listbox(left_panel, height=25, bg=COLORS["bg_card"], 
                                   fg=COLORS["text_primary"], relief='solid', 
                                   borderwidth=1, highlightthickness=0,
                                   selectbackground=COLORS["primary_light"],
                                   selectforeground='white',
                                   font=('Microsoft YaHei', 10))
        self.file_list.pack(fill=tk.BOTH, expand=True)
        self.refresh_file_list()
        
        self.history_frame = ttk.Frame(main_frame, style="Card.TFrame", padding="10")
        self.history_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        history_label = ttk.Label(self.history_frame, text="📋 对话历史", style="Section.TLabel")
        history_label.pack(anchor=tk.W, pady=(0, 10))
        
        history_btn_frame = ttk.Frame(self.history_frame)
        history_btn_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(history_btn_frame, text="🔄 刷新", command=self.refresh_history_list, 
                   style="Modern.TButton").pack(side=tk.LEFT)
        ttk.Button(history_btn_frame, text="🗑️ 清空", command=self.clear_history, 
                   style="Warning.TButton").pack(side=tk.RIGHT)
        
        self.history_list = tk.Listbox(self.history_frame, height=18, bg=COLORS["bg_card"], 
                                      fg=COLORS["text_primary"], relief='solid', 
                                      borderwidth=1, highlightthickness=0,
                                      selectbackground=COLORS["primary_light"],
                                      selectforeground='white',
                                      font=('Microsoft YaHei', 9))
        self.history_list.pack(fill=tk.BOTH, expand=True)
        self.history_list.bind('<Double-Button-1>', lambda e: self.load_selected_history())
        self.refresh_history_list()
        
        self.hide_history_panel()
        
        self.thought_chain_frame = ttk.Frame(main_frame, style="Card.TFrame", padding="10")
        self.thought_chain_frame.grid(row=1, column=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        thought_label = ttk.Label(self.thought_chain_frame, text="🧠 Agent 思考链", style="Section.TLabel")
        thought_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.thought_chain_display = scrolledtext.ScrolledText(self.thought_chain_frame, 
                                                              wrap=tk.WORD, height=12, 
                                                              bg=COLORS["bg_card"],
                                                              fg=COLORS["text_primary"],
                                                              relief='solid',
                                                              borderwidth=1,
                                                              font=('Microsoft YaHei', 9))
        self.thought_chain_display.pack(fill=tk.BOTH, expand=True)
        self.thought_chain_display.tag_config('thought', foreground=COLORS["warning"], font=('Microsoft YaHei', 9, 'italic'))
        self.thought_chain_display.tag_config('tool', foreground=COLORS["primary"])
        self.thought_chain_display.tag_config('answer', foreground=COLORS["success"], font=('Microsoft YaHei', 10, 'bold'))
        
        chat_frame = ttk.Frame(main_frame, style="Card.TFrame", padding="15")
        chat_frame.grid(row=1, column=3, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)
        
        chat_label = ttk.Label(chat_frame, text="💬 问答对话", style="Section.TLabel")
        chat_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.chat_display = scrolledtext.ScrolledText(chat_frame, wrap=tk.WORD, height=15,
                                                     bg=COLORS["bg"],
                                                     fg=COLORS["text_primary"],
                                                     relief='flat',
                                                     borderwidth=0,
                                                     font=('Microsoft YaHei', 10))
        self.chat_display.pack(fill=tk.BOTH, expand=True)
        self.chat_display.tag_config('user', foreground='black', justify='right')
        self.chat_display.tag_config('ai', foreground=COLORS["text_primary"], justify='left')
        self.chat_display.tag_config('system', foreground=COLORS["text_secondary"])
        
        input_frame = ttk.Frame(chat_frame, style="Main.TFrame")
        input_frame.pack(fill=tk.X, pady=(10, 0))
        input_frame.columnconfigure(0, weight=1)
        
        self.question_entry = ttk.Entry(input_frame, font=("Microsoft YaHei", 11), 
                                        style="Modern.TEntry")
        self.question_entry.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        self.question_entry.bind('<Return>', lambda e: self.send_question())
        
        self.send_button = ttk.Button(input_frame, text="➤ 发送", command=self.send_question, 
                                     style="Success.TButton")
        self.send_button.grid(row=0, column=1)
        
        progress_frame = ttk.Frame(main_frame, style="Card.TFrame", padding="10")
        progress_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=10)
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.progress.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 10))
        
        self.progress_label = ttk.Label(progress_frame, text="", style="Status.TLabel")
        self.progress_label.grid(row=0, column=1)
        
        self.hide_thought_chain()
    
    def check_ollama(self):
        def _check():
            try:
                response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m["name"] for m in models]
                    
                    def model_exists(name):
                        return any(name in m or m.startswith(name) for m in model_names)
                    
                    llm_ok = model_exists(LLM_MODEL)
                    embed_ok = model_exists(EMBED_MODEL)
                    
                    if llm_ok and embed_ok:
                        self.status_label.config(text=f"✅ Ollama服务正常！模型已就绪 ({LLM_MODEL})", 
                                                foreground=COLORS["success"])
                        self.send_button.config(state="normal")
                    else:
                        missing = []
                        if not llm_ok:
                            missing.append(LLM_MODEL)
                        if not embed_ok:
                            missing.append(EMBED_MODEL)
                        self.status_label.config(text=f"⚠️ 缺少模型: {', '.join(missing)}", 
                                                foreground=COLORS["warning"])
                        self.send_button.config(state="disabled")
                else:
                    self.status_label.config(text="❌ Ollama服务异常", foreground=COLORS["error"])
                    self.send_button.config(state="disabled")
            except Exception as e:
                self.status_label.config(text=f"❌ Ollama未启动: {str(e)}", foreground=COLORS["error"])
                self.send_button.config(state="disabled")
        threading.Thread(target=_check, daemon=True).start()
    
    def refresh_file_list(self):
        self.file_list.delete(0, tk.END)
        for f in UPLOAD_DIR.iterdir():
            if f.is_file():
                size = f.stat().st_size / 1024 / 1024
                mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%m-%d %H:%M")
                display = f"{f.name} ({size:.2f}MB) - {mtime}"
                self.file_list.insert(tk.END, display)
    
    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="选择教材文件",
            filetypes=[("PDF文件", "*.pdf"), ("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        if file_path:
            src = Path(file_path)
            dest = UPLOAD_DIR / src.name
            import shutil
            shutil.copy(src, dest)
            self.refresh_file_list()
            messagebox.showinfo("成功", f"文件 {src.name} 已上传！")
            self.add_message(f"📂 已上传教材: {src.name}", 'system')
    
    def clear_chat(self):
        self.chat_display.config(state='normal')
        self.chat_display.delete(1.0, tk.END)
        self.chat_display.config(state='disabled')
    
    def start_new_chat(self):
        if self.chat_history.current_conversation:
            self.chat_history.start_new_conversation()
        self.clear_chat()
    
    def toggle_history_panel(self):
        if self.history_frame.winfo_viewable():
            self.hide_history_panel()
        else:
            self.show_history_panel()
    
    def show_history_panel(self):
        self.history_frame.grid()
    
    def hide_history_panel(self):
        self.history_frame.grid_remove()
    
    def refresh_history_list(self):
        self.history_list.delete(0, tk.END)
        conversations = self.chat_history.get_recent_conversations(50)
        for i, conv in enumerate(conversations):
            timestamp = conv.get('timestamp', '')
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%m-%d %H:%M")
            except:
                time_str = timestamp[:10] if timestamp else "未知时间"
            
            first_q = ""
            for msg in conv.get('messages', []):
                first_q = msg.get('question', '')[:30]
                if first_q:
                    break
            
            if first_q:
                display = f"[{time_str}] {first_q}..."
            else:
                display = f"[{time_str}] 空对话"
            
            self.history_list.insert(tk.END, display)
    
    def load_selected_history(self):
        selection = self.history_list.curselection()
        if not selection:
            return
        
        idx = selection[0]
        conversations = self.chat_history.get_recent_conversations(50)
        if idx >= len(conversations):
            return
        
        conv = conversations[idx]
        self.chat_history.load_conversation_to_current(conv)
        
        self.clear_chat()
        for msg in conv.get('messages', []):
            self.add_message(f"👤 {msg.get('question', '')}", 'user')
            self.add_message(f"🤖 {msg.get('answer', '')}", 'ai')
        
        self.hide_history_panel()
    
    def clear_history(self):
        if messagebox.askyesno("确认", "确定清空所有对话历史吗？"):
            self.chat_history.clear_history()
            self.refresh_history_list()
    
    def toggle_agent_mode(self):
        self.use_agent_mode = not self.use_agent_mode
        if self.use_agent_mode:
            self.show_thought_chain()
            self.status_label.config(text="✅ Agent 模式已开启", foreground=COLORS["primary"])
        else:
            self.hide_thought_chain()
            self.status_label.config(text="✅ 普通 RAG 模式", foreground=COLORS["success"])
    
    def show_thought_chain(self):
        self.thought_chain_frame.grid()
    
    def hide_thought_chain(self):
        self.thought_chain_frame.grid_remove()
    
    def add_thought(self, message, tag):
        self.thought_chain_display.config(state='normal')
        self.thought_chain_display.insert(tk.END, message + "\n\n", tag)
        self.thought_chain_display.see(tk.END)
        self.thought_chain_display.config(state='disabled')
    
    def clear_thought_chain(self):
        self.thought_chain_display.config(state='normal')
        self.thought_chain_display.delete(1.0, tk.END)
        self.thought_chain_display.config(state='disabled')
    
    def add_message(self, message, tag):
        self.chat_display.config(state='normal')
        
        if tag == 'user':
            self.chat_display.insert(tk.END, message + "\n\n", tag)
        elif tag == 'ai':
            self.chat_display.insert(tk.END, message + "\n\n", tag)
        else:
            self.chat_display.insert(tk.END, message + "\n\n", tag)
        
        self.chat_display.see(tk.END)
        self.chat_display.config(state='disabled')
    
    def send_question(self):
        question = self.question_entry.get().strip()
        if not question:
            messagebox.showwarning("提示", "请输入问题！")
            return
        if not self.file_list.curselection():
            messagebox.showwarning("提示", "请先选择教材文件！")
            return
        
        self.question_entry.delete(0, tk.END)
        self.add_message(f"👤 {question}", 'user')
        
        if self.use_agent_mode:
            self.clear_thought_chain()
        
        self.send_button.config(state="disabled")
        self.progress.start(10)
        self.progress_label.config(text="正在思考...")
        
        selected_item = self.file_list.get(self.file_list.curselection())
        import re
        match = re.match(r'^(.+?)\s+\(', selected_item)
        if match:
            filename = match.group(1).strip()
        else:
            filename = selected_item.split(" ")[0]
        
        selected_file = UPLOAD_DIR / filename
        
        def _ask():
            try:
                if self.use_agent_mode:
                    result = self._agent_ask(question, str(selected_file))
                    answer = result.get("answer", "")
                    
                    thought_chain = result.get("thought_chain", [])
                    for step in thought_chain:
                        step_num = step.get("step", 1)
                        state = step.get("state", "")
                        thought = step.get("thought", "")
                        tool_name = step.get("tool", "")
                        tool_result = step.get("tool_result", "")
                        
                        self.root.after(0, lambda: self.add_thought(f"📌 步骤 {step_num} - {state}", 'system'))
                        if thought:
                            self.root.after(0, lambda t=thought: self.add_thought(f"💭 思考: {t}", 'thought'))
                        if tool_name:
                            self.root.after(0, lambda tn=tool_name, ti=step.get("tool_input", ""): 
                                             self.add_thought(f"🔧 使用工具: {tn}({ti})", 'tool'))
                        if tool_result:
                            self.root.after(0, lambda tr=tool_result: self.add_thought(f"📦 工具结果: {tr[:300]}...", 'tool'))
                    
                    if answer:
                        self.root.after(0, lambda: self.add_thought(f"✅ 最终回答:\n{answer}", 'answer'))
                        self.root.after(0, lambda: self.add_message(f"🤖 {answer}", 'ai'))
                        self.root.after(0, lambda: self.chat_history.add_message(question, answer))
                else:
                    answer = self._real_ask(question, str(selected_file))
                    self.root.after(0, lambda: self.add_message(f"🤖 {answer}", 'ai'))
                    self.root.after(0, lambda: self.chat_history.add_message(question, answer))
            except Exception as e:
                self.root.after(0, lambda: self.add_message(f"❌ 错误: {str(e)}", 'system'))
            finally:
                self.root.after(0, self._finish)
        
        threading.Thread(target=_ask, daemon=True).start()
    
    def _analyze_and_optimize_question(self, question):
        """
        分析问题意图并优化搜索关键词
        """
        question_lower = question.lower()
        
        # 问题类型识别
        is_smoke_detector = any(keyword in question_lower for keyword in ['烟感', '烟探测器', '探测器型号', '探测器'])
        is_module = any(keyword in question_lower for keyword in ['模块', '模块型号'])
        is_capacity = any(keyword in question_lower for keyword in ['容量', '系统容量', '最大容量'])
        is_system_capacity = is_capacity and any(keyword in question_lower for keyword in ['系统', 'est3', '整体'])
        is_classification = any(keyword in question_lower for keyword in ['分类', '种类', '类型', '分为'])
        
        # 优化搜索关键词
        optimized_question = question
        
        if is_smoke_detector and not is_module:
            # 烟感问题：强调SIGA系列探测器，排除SIGI模块
            optimized_question = f"{question} signature探测器 SIGA系列 不含模块"
        elif is_module and not is_smoke_detector:
            # 模块问题：强调模块SIGI系列
            optimized_question = f"{question} 模块型号 SIGI系列 CT2 CC1"
        elif is_system_capacity:
            # 系统容量问题：强调系统整体容量
            optimized_question = f"{question} 系统整体容量 EST3 最多64节点 160000设备"
        elif is_classification:
            # 分类问题：添加更多相关关键词
            optimized_question = f"{question} 分类 类型 种类 分类方法"
        
        return optimized_question
    
    def _filter_and_verify_docs(self, docs, question):
        """
        过滤和验证检索结果，提高相关性
        """
        if not docs:
            return docs
        
        question_lower = question.lower()
        
        # 判断问题类型
        is_smoke_detector = any(keyword in question_lower for keyword in ['烟感', '烟探测器', '探测器型号', '探测器'])
        is_module = any(keyword in question_lower for keyword in ['模块', '模块型号'])
        is_capacity = any(keyword in question_lower for keyword in ['容量', '系统容量', '最大容量'])
        is_system_capacity = is_capacity and any(keyword in question_lower for keyword in ['系统', 'est3', '整体'])
        
        filtered_docs = []
        
        for doc in docs:
            content = doc.get('page_content', '')
            content_lower = content.lower()
            keep = True
            
            # 根据问题类型过滤
            if is_smoke_detector and not is_module:
                # 烟感问题：检查是否包含模块相关型号（SIGI开头、CT2、CC1等）
                module_signatures = ['sigi-', 'sigi.', 'sigi ', '-ct2', '-cc1', '模块型号', '模块型号：']
                has_module_signature = any(sig in content_lower for sig in module_signatures)
                
                # 检查是否明确标注为探测器/烟感
                is_detector_section = any(keyword in content_lower for keyword in ['探测器型号', '烟感型号', '探测器型号：', '烟感型号：', 'signature系列探测器'])
                
                # 如果包含模块签名且不是探测器章节，则排除
                if has_module_signature and not is_detector_section:
                    keep = False
                
                # 如果内容主要在"模块型号"部分，也排除
                if '模块型号' in content_lower and not is_detector_section:
                    detector_pos = content_lower.find('探测器型号')
                    module_pos = content_lower.find('模块型号')
                    
                    if detector_pos == -1 and module_pos >= 0:
                        keep = False
            
            elif is_module and not is_smoke_detector:
                is_module_section = any(keyword in content_lower for keyword in ['模块型号', 'sigi-', 'sigi.', '-ct2', '-cc1'])
                
                if not is_module_section:
                    keep = False
            
            elif is_system_capacity:
                has_system_keywords = any(keyword in content_lower for keyword in ['系统容量', '系统整体容量', '最多64个', '160,000', '160000', '网络节点', '网络容量'])
                
                if not has_system_keywords:
                    keep = False
            
            if keep:
                filtered_docs.append(doc)
        
        # 如果过滤后没有结果，尝试扩大范围重新过滤
        if not filtered_docs:
            # 对于烟感问题，排除明确包含模块的内容
            if is_smoke_detector:
                for doc in docs:
                    content_lower = doc.get('page_content', '').lower()
                    # 如果文档中没有"模块型号"，就保留
                    if '模块型号' not in content_lower:
                        filtered_docs.append(doc)
        
        # 如果还是没有结果，保留原始结果
        if not filtered_docs:
            return docs
        
        return filtered_docs[:3]  # 保留前3个最相关的
    
    def _real_ask(self, question, file_path):
        if not HAS_RAG:
            return "⚠️ RAG依赖未安装！请先运行: pip install -r requirements.txt"
        try:
            file_stem = Path(file_path).stem
            db_dir = VECTOR_DB_DIR / file_stem
            embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
            
            if db_dir.exists() and any(db_dir.iterdir()):
                vectorstore = SimpleVectorStore(str(db_dir), embeddings)
            else:
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext == '.pdf':
                    import pdfplumber
                    documents = []
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, 1):
                            text = page.extract_text()
                            if text:
                                doc = type('', (), {})()
                                doc.page_content = text
                                doc.metadata = {'page': page_num, 'source': file_path}
                                documents.append(doc)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                chunks = text_splitter.split_documents(documents)
                
                vectorstore = SimpleVectorStore(str(db_dir), embeddings)
                
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    batch_texts = [c.page_content for c in batch]
                    batch_metadatas = [c.metadata for c in batch]
                    vectorstore.add_texts(batch_texts, batch_metadatas)
            
            # 问题意图分析 - 优化检索
            optimized_question = self._analyze_and_optimize_question(question)
            
            docs = vectorstore.enhanced_similarity_search(optimized_question, k=4)
            
            # 检索结果验证和过滤
            docs = self._filter_and_verify_docs(docs, question)
            
            context_parts = []
            for i, doc in enumerate(docs):
                page_info = ""
                metadata = doc.get('metadata', {})
                if 'page' in metadata:
                    page_info = f" (来自第{metadata['page']}页)"
                context_parts.append(f"--- 参考资料 {i+1}{page_info} ---\n{doc['page_content'][:800]}")
            
            context = "\n\n".join(context_parts)
            
            prompt_template = """你是一个专业的一建机电考试辅导老师。请根据以下教材内容回答问题。

【重要要求】
1. 答案必须100%基于提供的教材参考资料，不要使用任何教材以外的知识
2. 仔细阅读所有参考资料，即使内容不完全匹配也要尽力从中提取相关信息
3. 如果教材中有部分相关内容，请基于这些内容进行回答，不要轻易说"没有找到"
4. 只有在完全没有任何相关信息时，才说明"教材中没有找到相关内容"
5. 必须明确注明答案来自教材的第几页
6. 回答格式要清晰美观

【分类验证要求】
- 如果问题涉及设备型号：仔细区分探测器/烟感型号 vs 模块/其他设备型号，只回答用户明确询问的那种类型
- 如果问题涉及容量：明确区分系统整体容量 vs 单个设备容量 vs 回路容量
- 如果搜索结果中包含多种类型的信息，只保留与问题直接相关的部分

参考内容:
{context}

问题:
{question}

请直接回答问题。"""
            
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=2048)
            
            chain = (
                {"context": lambda x: context, "question": lambda x: x}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            result = chain.invoke(question)
            return result
        except Exception as e:
            return f"❌ 错误: {str(e)}"
    
    def _agent_ask(self, question, file_path):
        identity_questions = ["你是谁", "你是什么", "你的身份", "自我介绍", "你叫什么", "你是做什么的"]
        for identity_q in identity_questions:
            if identity_q in question:
                answer = "我是知识达人，专门服务于知识问题方面的回答与讲解，你有什么问题都可以与我沟通咨询！"
                return {
                    "answer": answer,
                    "thought_chain": [{
                        "step": 1,
                        "state": "done",
                        "thought": "这是一个自我介绍问题，直接回复",
                        "answer": answer
                    }],
                    "success": True
                }
        
        if not HAS_RAG:
            return {"answer": "⚠️ RAG依赖未安装！", "thought_chain": []}
        try:
            file_stem = Path(file_path).stem
            db_dir = VECTOR_DB_DIR / file_stem
            embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
            llm = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0, num_ctx=4096)
            
            if db_dir.exists() and any(db_dir.iterdir()):
                self.vector_store = SimpleVectorStore(str(db_dir), embeddings)
            else:
                file_ext = Path(file_path).suffix.lower()
                
                if file_ext == '.pdf':
                    import pdfplumber
                    documents = []
                    with pdfplumber.open(file_path) as pdf:
                        for page_num, page in enumerate(pdf.pages, 1):
                            text = page.extract_text()
                            if text:
                                doc = type('', (), {})()
                                doc.page_content = text
                                doc.metadata = {'page': page_num, 'source': file_path}
                                documents.append(doc)
                else:
                    loader = TextLoader(file_path, encoding='utf-8')
                    documents = loader.load()
                
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                chunks = text_splitter.split_documents(documents)
                
                self.vector_store = SimpleVectorStore(str(db_dir), embeddings)
                
                batch_size = 50
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i:i+batch_size]
                    batch_texts = [c.page_content for c in batch]
                    batch_metadatas = [c.metadata for c in batch]
                    self.vector_store.add_texts(batch_texts, batch_metadatas)
            
            self.agent = Agent(llm, self.vector_store)
            result = self.agent.run(question, max_steps=3)
            return result
        except Exception as e:
            return {"answer": f"❌ 错误: {str(e)}", "thought_chain": []}
    
    def _finish(self):
        self.progress.stop()
        self.progress_label.config(text="")
        self.send_button.config(state="normal")

def on_closing():
    if hasattr(app, 'chat_history'):
        if app.chat_history.current_conversation:
            app.chat_history.start_new_conversation()
    root.destroy()

def main():
    root = tk.Tk()
    global app
    app = ContractRAGApp(root)
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
