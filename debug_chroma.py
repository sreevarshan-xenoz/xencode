import traceback
try:
    import chromadb
    client = chromadb.PersistentClient(path=".")
    print("Success")
except:
    traceback.print_exc()
