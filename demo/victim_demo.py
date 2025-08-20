# victim_llama_loop.py

import time
import random
import os
from llama_cpp import Llama

def main():
    # Adjust this path to wherever you downloaded your GGUF vocab-only file
    model_path = "/home/user/Projects/llamacpp_detection/models/phi3-mini/Phi-3-mini-4k-instruct-q4.gguf"

    print(f"[{os.getpid()}] Loading vocabulary-only from {model_path}…")
    llm = Llama(
        model_path=model_path,
        vocab_only=True,  # skip all weights
        use_mmap=True     # mmap the vocab for minimal overhead
    )
    print(f"[{os.getpid()}] Vocabulary loaded. Starting continuous decode of token 2.")

    token_id = 2
    try:
        while True:
            # Optional random sleep; remove or adjust as needed
            time.sleep(random.uniform(0.3, 0.5))
            llm.detokenize([token_id])
            # print(f"[{os.getpid()}] Decoded token {token_id} at {time.time()}")
    except KeyboardInterrupt:
        print("\nInterrupted; exiting.")

if __name__ == "__main__":
    main()
