import traceback

try:
    import chromadb
    client = chromadb.PersistentClient(path=".")
    print("Success")
except Exception:
    traceback.print_exc()
