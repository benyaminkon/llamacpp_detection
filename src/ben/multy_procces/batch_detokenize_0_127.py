#!/usr/bin/env python3
import subprocess
import re
import csv
import os
from pathlib import Path

# --- paths (edit if yours differ) ---
PIN = "/home/user/pin-3.31-intel/pin"
PINTOOL = "/home/user/pin-3.31-intel/source/tools/detokenize_tracer/obj-intel64/detokenize_tracer.so"
TEST_SCRIPT = "test_singel_token_memory.py"  # expects a single argv: token id

# Output from the pintool lands here every run
PINTOOL_OUT = "detokenize_trace.out"

# Final CSV (keeping your requested extension)
CSV_OUT = "0-127_tokens_addresses.cvs"

# --- regex to parse one line like:
# memcpy_src 0x64c9f683d8b0 len 4 (token=10000)
LINE_RE = re.compile(
    r"^memcpy_src\s+0x([0-9a-fA-F]+)\s+len\s+(\d+)\s+\(token=(\d+)\)\s*$"
)

def run_for_token(tok: int):
    """
    Run PIN + pintool for one token and return (addr_hex, length_int) or (None, None).
    """
    # Remove old trace to avoid stale parses if something fails mid-run
    try:
        if os.path.exists(PINTOOL_OUT):
            os.remove(PINTOOL_OUT)
    except OSError:
        pass

    cmd = [
        PIN,
        "-t", PINTOOL,
        "-gate_detok", "0",
        "-only_tokens", str(tok),
        "-dump_bytes", "0",
        "--",
        "python3",
        TEST_SCRIPT,
        str(tok),
    ]

    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Optional: show progress line from the tool / python
    print(f"[{tok:3d}] return={proc.returncode}")

    # Parse the pintool output file
    if not os.path.exists(PINTOOL_OUT):
        return (None, None)

    addr = length = token_seen = None
    with open(PINTOOL_OUT, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            a, l, t = m.groups()
            if int(t) != tok:
                # if gate is off and other work happened, ignore mismatched tokens
                continue
            addr = "0x" + a.lower()
            length = int(l)
            token_seen = tok
            # we expect exactly one relevant memcpy per token_to_piece call
            break

    if token_seen is None:
        return (None, None)
    return (addr, length)

def main():
    rows = []
    for tok in range(128):
        addr, length = run_for_token(tok)
        rows.append({"token": tok, "address": addr or "", "len": length if length is not None else ""})

    # Write CSV
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["token", "address", "len"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved {CSV_OUT} with {len(rows)} rows.")
    # Quick summary: how many tokens resolved
    resolved = sum(1 for r in rows if r["address"])
    print(f"Resolved addresses for {resolved}/128 tokens.")

if __name__ == "__main__":
    main()
