import traceback
try:
    from langchain_ollama import OllamaEmbeddings
    print("OllamaEmbeddings imported")
    from xencode.rag.vector_store import VectorStore
    print("VectorStore imported")
    vs = VectorStore(persist_directory="d:\\xencode\\.xencode\\test_store")
    print("VectorStore instantiated")
except:
    traceback.print_exc()
