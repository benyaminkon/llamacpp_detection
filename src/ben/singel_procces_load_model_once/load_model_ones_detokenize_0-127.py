# load_model_ones_detokenize_0-127.py
from llama_cpp import Llama

print("Loading model once…")
llm = Llama(
    model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf",
    vocab_only=True,
    verbose=False
)

print("Model loaded. Detokenizing tokens 0..127")
for tok in range(128):
    # this triggers token_to_piece (our pintool will log memcpy_src for these tokens)
    _ = llm.detokenize([tok])

print("Done.")
