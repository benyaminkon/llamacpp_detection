# File: victim_64_test.py

import sys
import os
import time
import random
import signal
from pathlib import Path
from llama_cpp import Llama

# === CONFIGURATION FOR 64-TOKEN TEST ===
NUM_TOKENS = 64  # Only test first 64 tokens
DECODE_REPEATS = 8  # How many times to decode each token
MIN_DELAY = 0.05  # Minimum delay between decodes (50ms)
MAX_DELAY = 0.15  # Maximum delay between decodes (150ms)

# Globals
llm: Llama
tokens = list(range(NUM_TOKENS))  # [0, 1, 2, ..., 63]
current_index = 0
parent_pid = None
decode_count = 0  # Track total decodings for debugging

def decode_token_x_times(token_id: int):
    """
    Decode a token multiple times with controlled delays
    Reduced delays for faster testing
    """
    global decode_count
    for i in range(DECODE_REPEATS):
        # Shorter, more consistent delays for testing
        delay = random.uniform(MIN_DELAY, MAX_DELAY)
        time.sleep(delay)
        
        # The actual decode that creates cache activity
        result = llm.detokenize([token_id])
        decode_count += 1
        
        # Optional: Print progress every 10 decodings
        if decode_count % 10 == 0:
            print(f"  [Debug] Completed {decode_count} decodings...", flush=True)
    
    return result

def handle_sigusr1(signum, frame):
    """
    SIGUSR1 handler: decode next token, then ACK back via SIGUSR2
    """
    global current_index
    
    if current_index >= NUM_TOKENS:
        print(f"[!] Warning: Received signal but all {NUM_TOKENS} tokens already processed!")
        return
    
    token_id = tokens[current_index]
    print(f"[+] Decoding token #{current_index} (ID {token_id})")
    
    # Perform the decodings (this is what the attacker monitors)
    start_time = time.time()
    decode_token_x_times(token_id)
    elapsed = time.time() - start_time
    
    print(f"    Completed in {elapsed:.2f} seconds")
    
    current_index += 1
    
    # Send ACK to orchestrator
    os.kill(parent_pid, signal.SIGUSR2)
    
    # Progress report
    print(f"[*] Progress: {current_index}/{NUM_TOKENS} tokens completed\n", flush=True)

def handle_sigterm(signum, frame):
    """Clean shutdown handler"""
    print(f"\n[*] Victim shutting down. Processed {current_index}/{NUM_TOKENS} tokens.")
    print(f"[*] Total decodings performed: {decode_count}")
    sys.exit(0)

def main():
    global llm, parent_pid
    
    print("=" * 60)
    print("VICTIM PROCESS - 64 TOKEN TEST")
    print("=" * 60)
    
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <model_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    parent_pid = os.getppid()
    
    print(f"[*] Victim PID: {os.getpid()}")
    print(f"[*] Parent PID: {parent_pid}")
    print(f"[*] Model path: {model_path}")
    print(f"[*] Testing {NUM_TOKENS} tokens with {DECODE_REPEATS} decodes each")
    print(f"[*] Delay range: {MIN_DELAY*1000:.0f}-{MAX_DELAY*1000:.0f}ms")
    
    # Load model (vocabulary only for speed)
    print("\n[*] Loading vocabulary...", flush=True)
    start_load = time.time()
    
    llm = Llama(
        model_path=model_path,
        vocab_only=True,  # Only load tokenizer, not model weights
        use_mmap=True,
        verbose=False  # Reduce output noise
    )
    
    load_time = time.time() - start_load
    print(f"[✓] Vocabulary loaded in {load_time:.2f} seconds")
    
    # Verify vocabulary size
    vocab_size = llm.n_vocab()
    print(f"[*] Vocabulary size: {vocab_size} tokens")
    
    if NUM_TOKENS > vocab_size:
        print(f"[!] Warning: Testing {NUM_TOKENS} tokens but vocab only has {vocab_size}")
    
    # Test decode function with token 0
    print("\n[*] Testing decode function...", flush=True)
    test_result = llm.detokenize([0])
    print(f"[✓] Token 0 decodes to: {repr(test_result)}")
    
    # Register signal handlers
    signal.signal(signal.SIGUSR1, handle_sigusr1)
    signal.signal(signal.SIGUSR2, lambda s, f: None)  # Ignore
    signal.signal(signal.SIGTERM, handle_sigterm)  # Clean shutdown
    signal.signal(signal.SIGINT, handle_sigterm)   # Ctrl+C handler
    
    print("\n[✓] Signal handlers registered")
    print("[*] Victim ready. Awaiting SIGUSR1 signals...")
    print("=" * 60)
    print(flush=True)
    
    # Main loop - wait for signals
    try:
        while current_index < NUM_TOKENS:
            signal.pause()  # Sleep until signal arrives
        
        print(f"\n[✓] All {NUM_TOKENS} tokens processed successfully!")
        print(f"[*] Total decodings: {decode_count}")
        
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print(f"\n[*] Final stats: {current_index}/{NUM_TOKENS} tokens, {decode_count} total decodings")

if __name__ == "__main__":
    main()