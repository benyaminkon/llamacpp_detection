from llama_cpp import Llama
model = Llama(model_path="/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf", vocab_only=True, verbose=False)
result = model.detokenize([10000])