# File: src/victim.py

import sys
import os
import time
import random
import signal
from pathlib import Path
from llama_cpp import Llama

# Globals
llm: Llama
tokens = [i for i in range(32064)]
current_index = 0
parent_pid = None

def decode_token_x_times(token_id: int):
    """Call llm.detokenize() eight times with random sleeps."""
    for _ in range(8):
        time.sleep(random.uniform(0.1, 0.3))
        llm.detokenize([token_id])

def handle_sigusr1(signum, frame):
    """SIGUSR1 handler: decode next token, then ACK back via SIGUSR2."""
    global current_index
    token_id = tokens[current_index]
    # print(f"[+] Decoding token #{current_index} (ID {token_id})")
    decode_token_x_times(token_id)
    current_index += 1

    # send back ACK to orchestrator
    os.kill(parent_pid, signal.SIGUSR2)

def main():
    global llm, parent_pid

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    parent_pid = os.getppid()

    llm = Llama(
        model_path=model_path,
        vocab_only=True,
        use_mmap=True
    )
    print("Vocabulary loaded. Awaiting SIGUSR1…")

    # Register handlers
    signal.signal(signal.SIGUSR1, handle_sigusr1)
    # Ignore SIGUSR2 so that if it ever arrives unexpectedly it won't kill us
    signal.signal(signal.SIGUSR2, lambda s, f: None)

    # Loop forever, waking only on signals
    try:
        while True:
            signal.pause()
    except KeyboardInterrupt:
        print("Interrupted; exiting.")

if __name__ == "__main__":
    main()
