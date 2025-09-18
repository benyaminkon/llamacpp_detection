# test_singel_token_memory.py
import sys
from llama_cpp import Llama

token = int(sys.argv[1]) if len(sys.argv) > 1 else 10

print("Loading model and calling detokenize")
llm = Llama(
    model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
    vocab_only=True,
    verbose=False
)
print("Model loaded, calling detokenize")
result = llm.detokenize([token])
print(f"Detokenize result: {result}")
