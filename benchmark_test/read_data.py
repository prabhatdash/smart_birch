import time
import os
import json

FILE_PATH = "dataset/data.txt"        # input TXT (space-separated)
JSON_LOG = "dataset/stream.jsonl"     # output NDJSON file (appends)
DELAY_SEC = 1.0                       # pause between prints
START_AT_END = False                  # True -> start tailing new lines only

def parse_line(line):
    """
    Parse a space-separated line:
    date time epoch moteid temperature humidity light voltage
    Example:
    2004-03-31 03:38:15.757551 2 1 122.153 -3.91901 11.04 2.03397
    """
    parts = line.strip().split()
    if len(parts) != 8:
        return None
    date_str, time_str = parts[0], parts[1]
    try:
        return {
            "datetime": f"{date_str} {time_str}",
            "epoch": int(parts[2]),
            "moteid": int(parts[3]),
            "temperature": float(parts[4]),
            "humidity": float(parts[5]),
            "light": float(parts[6]),
            "voltage": float(parts[7]),
        }
    except ValueError:
        return None

def append_json(record, path):
    """Append one JSON object per line (NDJSON)."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as jf:
        jf.write(json.dumps(record, ensure_ascii=False))
        jf.write("\n")
        jf.flush()
        os.fsync(jf.fileno())  # make it durable

def stream_file(path):
    while not os.path.exists(path):
        print(f"Waiting for file: {path}", flush=True)
        time.sleep(0.5)

    print("Starting sensor data simulation...\n", flush=True)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        if START_AT_END:
            f.seek(0, os.SEEK_END)

        try:
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.2)  # no new data yet
                    continue

                rec = parse_line(line)
                if rec is None:
                    continue

                # Print to console
                print(rec, flush=True)

                # Append to JSON log
                append_json(rec, JSON_LOG)

                # pacing
                time.sleep(DELAY_SEC)
        except KeyboardInterrupt:
            print("\nStopped by user.", flush=True)

if __name__ == "__main__":
    stream_file(FILE_PATH)
