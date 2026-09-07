from typing import List, Optional

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None
try:
    from langchain_anthropic import ChatAnthropic
except ImportError:
    ChatAnthropic = None
try:
    from langchain.tools import BaseTool
except ImportError:
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        BaseTool = object
try:
    from langchain_core.tools import BaseTool as BaseToolNew
except ImportError:
    BaseToolNew = BaseTool
try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    ChatPromptTemplate = None

from .tools import ReadFileTool, WriteFileTool, ExecuteCommandTool

# Import AgentExecutor with fallback
try:
    from langchain.agents import AgentExecutor
except ImportError:
    # If AgentExecutor is not available, define a minimal placeholder
    class AgentExecutor:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("AgentExecutor not available in this LangChain version")


class LangChainManager:
    """Manages the LangChain agent and tools."""

    def __init__(self, model_name: str = "qwen3:4b", base_url: str = "http://localhost:11434",
                 use_memory: bool = True, db_path: str = "agentic_memory.db",
                 smart_model_selection: bool = False, use_rag: bool = False,
                 provider: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url
        self.smart_model_selection = smart_model_selection
        self.use_rag = use_rag
        self.provider = provider or self._detect_provider_from_model(model_name)

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
            try:
                from ..multi_model_system import MultiModelManager
                self.model_selector = MultiModelManager()
            except Exception:
                self.model_selector = None

        # Initialize LLM based on provider
        self.llm = self._create_llm(model_name, base_url)
        self.tools = self._setup_tools()
        self.agent_executor = self._setup_agent()

    @staticmethod
    def _detect_provider_from_model(model_name: str) -> str:
        """Detect provider from model name prefix."""
        known = {'openai', 'anthropic', 'google_gemini', 'openrouter', 'qwen', 'huggingface'}
        if ':' in model_name:
            prefix = model_name.split(':', 1)[0].lower()
            if prefix in known:
                return prefix
        return 'ollama'

    def _create_llm(self, model_name: str, base_url: str):
        """Create the appropriate LLM based on provider."""
        provider = self.provider

        if provider == 'openai' and ChatOpenAI is not None:
            clean = model_name.split(':', 1)[1] if ':' in model_name else model_name
            return ChatOpenAI(model=clean, temperature=0)
        elif provider == 'anthropic' and ChatAnthropic is not None:
            clean = model_name.split(':', 1)[1] if ':' in model_name else model_name
            return ChatAnthropic(model=clean, temperature=0)
        elif provider == 'ollama':
            if ChatOllama is None:
                raise ImportError(
                    "langchain-ollama is required for Ollama models. "
                    "Install with: pip install langchain-ollama"
                )
            return ChatOllama(model=model_name, base_url=base_url, temperature=0)
        else:
            # Fallback: try ollama, then raise
            if ChatOllama is not None:
                return ChatOllama(model=model_name, base_url=base_url, temperature=0)
            raise ImportError(
                f"No langchain provider available for '{provider}'. "
                "Install langchain-ollama, langchain-openai, or langchain-anthropic."
            )

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
        """Set up the tool-calling agent."""
        # Create a basic agent that works with the current LangChain version
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

        # Create a simple prompt that works with most LangChain versions
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant. Use the provided tools to answer questions.

{context_block}

You have access to the following tools:
{tools}

When you need to use a tool, respond with a JSON object like this:
```json
{{"name": "tool_name", "arguments": {{"arg1": "value1", "arg2": "value2"}}}}
```

After receiving the tool result, respond with your final answer."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        # Try to create an agent using the most compatible approach
        try:
            # Try the create_tool_calling_agent approach
            from langchain.agents import create_tool_calling_agent
            agent = create_tool_calling_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=prompt
            )

            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=10,
                max_execution_time=30.0
            )
        except ImportError:
            # If create_tool_calling_agent is not available, try create_json_chat_agent
            try:
                from langchain.agents import create_json_chat_agent
                agent = create_json_chat_agent(
                    llm=self.llm,
                    tools=self.tools,
                    prompt=prompt
                )

                return AgentExecutor(
                    agent=agent,
                    tools=self.tools,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,
                    max_execution_time=30.0
                )
            except ImportError:
                # If neither is available, try create_structured_chat_agent as another option
                try:
                    from langchain.agents import create_structured_chat_agent
                    agent = create_structured_chat_agent(
                        llm=self.llm,
                        tools=self.tools,
                        prompt=prompt
                    )

                    return AgentExecutor(
                        agent=agent,
                        tools=self.tools,
                        verbose=True,
                        handle_parsing_errors=True,
                        max_iterations=10,
                        max_execution_time=30.0
                    )
                except ImportError:
                    # If none of the above are available, create a basic executor that just uses the LLM directly
                    # This is a fallback that provides basic functionality
                    class BasicAgentExecutor:
                        def __init__(self, llm, tools):
                            self.llm = llm
                            self.tools = tools

                        def invoke(self, inputs):
                            # Basic implementation that just sends the input to the LLM
                            input_text = inputs.get("input", "")
                            response = self.llm.invoke(input_text)
                            return {"output": str(response)}

                    return BasicAgentExecutor(self.llm, self.tools)

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
                    # Continue without RAG context if it fails

            # Store user message
            if self.use_memory:
                try:
                    self.memory.add_message(role="user", content=user_input)
                except Exception as e:
                    print(f"Failed to store user message in memory: {e}")

            # Prepare inputs for agent
            inputs = {
                "input": user_input,
                "context_block": context_block
            }

            # Add agent_scratchpad if needed by the agent
            if hasattr(self.agent_executor, 'agent') and hasattr(self.agent_executor.agent, 'input_keys'):
                if 'agent_scratchpad' in self.agent_executor.agent.input_keys:
                    from langchain_core.messages import AIMessage, HumanMessage
                    inputs["agent_scratchpad"] = []

            # Run agent with context
            result = self.agent_executor.invoke(inputs)

            # The new agent returns the result directly
            if isinstance(result, dict):
                output = result.get("output", str(result))
            else:
                output = str(result)

            # Store assistant response
            if self.use_memory:
                try:
                    self.memory.add_message(role="assistant", content=output)
                except Exception as e:
                    print(f"Failed to store assistant response in memory: {e}")

            return output
        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            if self.use_memory:
                try:
                    self.memory.add_message(role="assistant", content=error_msg)
                except Exception as mem_e:
                    print(f"Failed to store error message in memory: {mem_e}")
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
