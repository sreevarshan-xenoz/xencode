import traceback

try:
    print("OllamaEmbeddings imported")
    from xencode.rag.vector_store import VectorStore
    print("VectorStore imported")
    vs = VectorStore(persist_directory="d:\\xencode\\.xencode\\test_store")
    print("VectorStore instantiated")
except Exception:
    traceback.print_exc()
