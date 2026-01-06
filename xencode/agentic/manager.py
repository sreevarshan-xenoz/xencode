from typing import List, Optional

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.tools import BaseTool

from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool


class LangChainManager:
    """Manages the LangChain agent and tools."""

    def __init__(self, model_name: str = "qwen3:4b", base_url: str = "http://localhost:11434", 
                 use_memory: bool = True, db_path: str = "agentic_memory.db", 
                 smart_model_selection: bool = False, use_rag: bool = False):
        self.model_name = model_name
        self.base_url = base_url
        self.smart_model_selection = smart_model_selection
        self.use_rag = use_rag
        
        # Initialize RAG if enabled
        self.vector_store = None
        if use_rag:
            try:
                from ..rag.vector_store import VectorStore
                self.vector_store = VectorStore(embedding_model=model_name)
            except ImportError:
                print("Warning: RAG dependencies not found.")
            except Exception as e:
                print(f"Warning: RAG initialization failed: {e}")
        
        # Initialize model selector if enabled
        if smart_model_selection:
            from ..multi_model_system import MultiModelManager
            self.model_selector = MultiModelManager()
        
        self.llm = ChatOllama(model=model_name, base_url=base_url, temperature=0)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()
        
        # Memory system
        self.use_memory = use_memory
        if use_memory:
            from .memory import ConversationMemory, ContextManager
            self.memory = ConversationMemory(db_path)
            self.context_manager = ContextManager()
            self.memory.start_session(model_name=model_name)

    def _setup_tools(self) -> List[BaseTool]:
        """Initialize the tools available to the agent."""
        from .advanced_tools import ToolRegistry
        
        registry = ToolRegistry()
        return registry.get_all_tools()

    def _setup_agent(self) -> AgentExecutor:
        """Set up the ReAct agent."""
        # Define a simple ReAct prompt
        # Note: This is a basic prompt, might need tuning for specific models
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

{context_block}

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)

        agent = create_react_agent(self.llm, self.tools, prompt)
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10
        )

    def run_agent(self, user_input: str) -> str:
        """Run the agent with the given input."""
        try:
            # RAG Context Retrieval
            context_block = ""
            if self.use_rag and self.vector_store:
                try:
                    docs = self.vector_store.similarity_search(user_input, k=3)
                    if docs:
                        context_strings = []
                        for doc in docs:
                            source = doc.metadata.get('filename', 'unknown')
                            context_strings.append(f"--- snippet from {source} ---\n{doc.page_content}\n")
                        context_block = "Context from Vector Store:\n" + "\n".join(context_strings) + "\nEnd of Context.\n"
                except Exception as e:
                    print(f"RAG retrieval failed: {e}")

            # Store user message
            if self.use_memory:
                self.memory.add_message(role="user", content=user_input)
            
            # Run agent
            # We pass context_block as a partial variable if needed, or inject into input.
            # But the prompt template expects {context_block}
            # Or if it doesn't, we can prepend it to input.
            # Since I modified the template, I must pass `context_block` to invoke.
            
            # The agent executor invoke takes input.
            # To pass extra variables to prompt, we might need to modify how the agent is structured.
            # But create_react_agent binds prompt.
            # Actually, `AgentExecutor.invoke` accepts input dict that fills prompt variables.
            
            result = self.agent_executor.invoke({
                "input": user_input,
                "context_block": context_block
            })
            output = result.get("output", "No output returned")
            
            # Store assistant response
            if self.use_memory:
                self.memory.add_message(role="assistant", content=output)
            
            return output
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            if self.use_memory:
                self.memory.add_message(role="assistant", content=error_msg)
            return error_msg
    
    def suggest_model_for_task(self, task: str) -> str:
        """Suggest the best model for a given task using MultiModelManager."""
        if not self.smart_model_selection:
            return self.model_name
        
        suggested_model, reason = self.model_selector.suggest_best_model(task)
        return suggested_model
    
    def switch_model(self, new_model: str) -> bool:
        """Switch to a different model."""
        try:
            self.model_name = new_model
            self.llm = ChatOllama(model=new_model, base_url=self.base_url, temperature=0)
            # Rebuild agent with new model
            self.agent_executor = self._setup_agent()
            return True
        except Exception:
            return False
    
    def run_agent_with_smart_model(self, user_input: str) -> str:
        """Run agent with automatic model selection based on task."""
        if self.smart_model_selection:
            suggested_model = self.suggest_model_for_task(user_input)
            if suggested_model != self.model_name:
                self.switch_model(suggested_model)
        
        return self.run_agent(user_input)
