# backend/test_faiss.py
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
idx_path = os.path.join(HERE, "ml_book.index")
print("Python:", sys.version)
print("FAISS test index path:", idx_path)

import faiss
print("faiss imported:", faiss.__file__)
try:
    idx = faiss.read_index(idx_path)
    print("Index loaded. ntotal =", idx.ntotal)
except Exception as e:
    print("Error loading index:", repr(e))
