
import json
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum

class AgentState(Enum):
    THINKING = "thinking"
    USING_TOOL = "using_tool"
    ANSWERING = "answering"
    DONE = "done"
    ERROR = "error"

class AgentTool:
    """工具基类"""
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func
    
    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class AgentMemory:
    """Agent 记忆系统"""
    def __init__(self, max_short_term: int = 10):
        self.short_term: List[Dict] = []
        self.long_term: List[Dict] = []
        self.max_short_term = max_short_term
    
    def add(self, role: str, content: str):
        self.short_term.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        if len(self.short_term) > self.max_short_term:
            moved = self.short_term.pop(0)
            self.long_term.append(moved)
    
    def get_recent(self, n: int = 5) -> List[Dict]:
        return self.short_term[-n:]
    
    def clear(self):
        self.short_term = []
        self.long_term = []

class Agent:
    """RAG Agent 核心类"""
    
    def __init__(self, llm, vector_store=None):
        self.llm = llm
        self.vector_store = vector_store
        self.memory = AgentMemory()
        self.tools: Dict[str, AgentTool] = {}
        self.state = AgentState.THINKING
        self.thought_chain: List[Dict] = []
        
        if vector_store:
            self._register_rag_tools()
    
    def _register_rag_tools(self):
        """注册 RAG 相关工具"""
        self.register_tool(
            name="search_textbook",
            description="在教材中搜索相关内容，必须先使用这个工具才能回答问题",
            func=self._search_textbook
        )
    
    def register_tool(self, name: str, description: str, func: Callable):
        """注册新工具"""
        self.tools[name] = AgentTool(name, description, func)
    
    def _search_textbook(self, query: str, k: int = 4) -> str:
        """在教材中搜索相关内容"""
        if not self.vector_store:
            return "【错误】向量库未初始化，无法搜索教材内容。"
        
        docs = self.vector_store.enhanced_similarity_search(query, k=k)
        
        if not docs:
            return "【提示】在教材中没有找到相关内容。"
        
        results = []
        for i, doc in enumerate(docs, 1):
            metadata = doc.get('metadata', {})
            page = metadata.get('page', '未知')
            content = doc.get('page_content', '')
            results.append(f"【第{i}条 - 第{page}页】\n{content}")
        
        return "\n\n".join(results)
    
    def _format_thought_prompt(self, question: str) -> str:
        """生成思考提示词"""
        tools_str = "\n".join([
            f"- {t.name}: {t.description}" 
            for t in self.tools.values()
        ])
        
        return f"""你是一个专业的一建机电考试辅导 Agent。

【重要规则】
1. 回答必须100%基于教材内容，禁止使用任何教材以外的知识
2. 在回答任何问题之前，必须先使用 search_textbook 工具搜索教材
3. 如果搜索结果中没有相关内容，直接说明"教材中没有找到相关内容"
4. 回答时必须注明答案来自教材的第几页

【当前问题】
{question}

【可用工具】
{tools_str if self.tools else "无可用工具"}

【输出格式】
请以下面的JSON格式输出：
{{
    "thought": "你的思考过程",
    "need_tool": true,
    "tool_name": "search_textbook",
    "tool_input": "要搜索的关键词（通常是问题本身）",
    "answer": ""
}}

注意：必须设置 need_tool: true，必须使用 search_textbook 工具！"""
    
    def _parse_llm_response(self, response: str) -> Dict:
        """解析 LLM 响应"""
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[0]
            
            parsed = json.loads(response.strip())
            
            if not parsed.get("need_tool", False) and len(self.tools) > 0:
                parsed["need_tool"] = True
                parsed["tool_name"] = "search_textbook"
                parsed["tool_input"] = parsed.get("tool_input", "") or response
            
            return parsed
        except Exception as e:
            return {
                "thought": f"解析响应出错，直接使用搜索工具",
                "need_tool": True,
                "tool_name": "search_textbook",
                "tool_input": response
            }
    
    def run(self, question: str, max_steps: int = 2) -> Dict:
        """Agent 主运行循环"""
        try:
            self.state = AgentState.THINKING
            self.thought_chain = []
            self.memory.clear()
            self.memory.add("user", question)
            
            search_result = None
            
            identity_questions = ["你是谁", "你是什么", "你的身份", "自我介绍", "你叫什么", "你是做什么的"]
            for identity_q in identity_questions:
                if identity_q in question:
                    answer = "我是知识达人，专门服务于知识问题方面的回答与讲解，你有什么问题都可以与我沟通咨询！"
                    self.thought_chain = [{
                        "step": 1,
                        "state": "done",
                        "thought": "这是一个自我介绍问题",
                        "answer": answer
                    }]
                    return {
                        "question": question,
                        "answer": answer,
                        "thought_chain": self.thought_chain,
                        "success": True
                    }
            
            for step in range(max_steps):
                step_info = {"step": step + 1, "state": self.state.value}
                self.thought_chain.append(step_info)
                
                if self.state == AgentState.THINKING:
                    prompt = self._format_thought_prompt(question)
                    response = self.llm.invoke(prompt).content
                    
                    parsed = self._parse_llm_response(response)
                    step_info["thought"] = parsed.get("thought", "准备搜索教材内容")
                    
                    self.memory.add("assistant", f"思考: {step_info['thought']}")
                    
                    if step == 0 or not search_result:
                        self.state = AgentState.USING_TOOL
                        step_info["next_action"] = "使用搜索工具"
                    else:
                        self.state = AgentState.ANSWERING
                        step_info["next_action"] = "生成答案"
                
                elif self.state == AgentState.USING_TOOL:
                    tool_name = "search_textbook"
                    tool_input = question
                    
                    step_info["tool"] = tool_name
                    step_info["tool_input"] = tool_input
                    
                    if tool_name in self.tools:
                        try:
                            search_result = self.tools[tool_name].run(tool_input)
                            step_info["tool_result"] = search_result
                            self.memory.add("tool", f"搜索结果:\n{search_result}")
                        except Exception as tool_error:
                            search_result = f"搜索出错: {str(tool_error)}"
                            step_info["tool_result"] = search_result
                    else:
                        search_result = "工具不存在"
                        step_info["tool_result"] = search_result
                    
                    self.state = AgentState.ANSWERING
                
                elif self.state == AgentState.ANSWERING:
                    try:
                        final_answer = self._generate_final_answer(question, search_result)
                        step_info["answer"] = final_answer
                        self.memory.add("assistant", final_answer)
                    except Exception as answer_error:
                        final_answer = f"生成答案时出错: {str(answer_error)}"
                        step_info["answer"] = final_answer
                    
                    self.state = AgentState.DONE
                    break
            
            return {
                "question": question,
                "answer": self.thought_chain[-1].get("answer", "抱歉，无法生成答案。") if self.thought_chain else "抱歉，无法生成答案。",
                "thought_chain": self.thought_chain,
                "success": self.state == AgentState.DONE
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Agent运行出错: {str(e)}",
                "thought_chain": self.thought_chain if hasattr(self, 'thought_chain') else [],
                "success": False
            }
    
    def _generate_final_answer(self, question: str, context: str) -> str:
        """生成最终答案"""
        prompt = f"""你是一个专业的一建机电考试辅导老师。请根据以下教材内容回答问题。

【重要要求】
1. 答案必须100%基于提供的教材参考资料，不要使用任何教材以外的知识
2. 如果教材中没有相关内容，就直接说明"教材中没有找到相关内容"
3. 必须明确注明答案来自教材的第几页
4. 回答格式要清晰美观

【参考内容】
{context}

【问题】
{question}

请直接回答问题。"""
        
        try:
            result = self.llm.invoke(prompt).content
            return result
        except Exception as e:
            return f"生成答案时出错: {str(e)}\n\n搜索到的内容:\n{context[:500]}"
    
    def get_thought_chain(self) -> List[Dict]:
        """获取思考链"""
        return self.thought_chain
    
    def reset(self):
        """重置 Agent 状态"""
        self.memory.clear()
        self.state = AgentState.THINKING
        self.thought_chain = []
