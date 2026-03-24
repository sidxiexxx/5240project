import streamlit as st
from transformers import pipeline


# Start measuring runtime
start_time = time.perf_counter()

#===========
testing_pipe1 = pipeline(    "zero-shot-classification",
    model="facebook/bart-large-mnli")
review = """
This store is great; they delivered what I needed very quickly, and the products were just as good as advertised.
"""

labels = ["product quality", "delivery", "customer service", "price"]

result = testing_pipe1(review, candidate_labels=labels)
print(result['labels'],result['scores'])
#===========
# Stop measuring runtime
run_time = time.perf_counter() - start_time
print(f"Runtime: {run_time:.4f} seconds")
