from llama_cpp import Llama
import sys

if len(sys.argv) != 2:
    print("Usage: python3 test_different_tokens.py <token_id>")
    sys.exit(1)

token_id = int(sys.argv[1])
model = Llama(model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf", vocab_only=True, verbose=False)
result = model.detokenize([token_id])
print(f"Token {token_id}: {result}")