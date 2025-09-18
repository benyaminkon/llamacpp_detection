# collect_first_128.py
import subprocess, re, sys, os, csv

PIN = "/home/user/pin-3.31-intel/pin"
PINTOOL = "/home/user/pin-3.31-intel/source/tools/detokenize_tracer/obj-intel64/detokenize_tracer.so"

TOKENS = list(range(128))
ONLY_TOKENS_ARG = ",".join(str(t) for t in TOKENS)

OUT_LOG = "detokenize_trace.out"              # produced by the pintool
OUT_CSV = "0-127_tokens_addresses.cvs"        # (name as you requested)

def run_under_pin():
    cmd = [
        PIN, "-t", PINTOOL,
        "-gate_detok", "0",                 # always off per your request
        "-only_tokens", ONLY_TOKENS_ARG,    # filter to 0..127 so the log is tiny
        "-dump_bytes", "0",                 # no byte preview
        "--", "python3", "load_model_ones_detokenize_0-127.py",
    ]
    print("Running:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_and_write_csv():
    if not os.path.exists(OUT_LOG):
        print(f"ERROR: {OUT_LOG} not found. Did the PIN run succeed?", file=sys.stderr)
        sys.exit(1)

    rx = re.compile(r"memcpy_src 0x([0-9a-fA-F]+) len (\d+) \(token=(\d+)\)")
    rows = {}
    with open(OUT_LOG, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = rx.search(line)
            if not m:
                continue
            addr_hex, length, token = m.group(1), int(m.group(2)), int(m.group(3))
            if token in TOKENS:
                # keep the last occurrence if multiple (should normally be one per token)
                rows[token] = (f"0x{addr_hex}", length)

    # write CSV in token order
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["token", "address_hex", "len"])
        for t in TOKENS:
            addr, ln = rows.get(t, ("", ""))  # blank if missing
            w.writerow([t, addr, ln])

    missing = [t for t in TOKENS if t not in rows]
    if missing:
        print(f"Warning: no address found for tokens: {missing}")
    print(f"Wrote {OUT_CSV} with {len(rows)} rows.")

if __name__ == "__main__":
    run_under_pin()
    parse_and_write_csv()
