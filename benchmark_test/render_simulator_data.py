# rta_birch_plotly.py
import os
import time
import json
import psutil
import pandas as pd
import numpy as np

from sklearn.cluster import Birch, KMeans
from kneed import KneeLocator

import plotly.express as px

from dash import Dash, dcc, html, Output, Input, State, no_update

# ------------------ Config ------------------
JSONL_PATH = "dataset/stream.jsonl"       # streaming NDJSON file
LOG_PATH   = "logs/birch_exec_log.csv"    # metrics log CSV
REFRESH_MS = 2000                         # auto-refresh interval (2s)
MAX_ROWS   = 100000                       # cap rows to avoid memory bloat
SAMPLE_FOR_K = 20000                      # sample size to estimate k (speed)
RANDOM_STATE = 42

# ------------------ Globals -----------------
ITERATION = 0
TOTAL_TIME = 0.0
PREV_ROWS = 0

# Ensure directories exist
os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)

# Create CSV with header if not exists
if not os.path.exists(LOG_PATH):
    pd.DataFrame(columns=[
        "Iteration No", "Data Size", "Batch Size",
        "Execution Time", "Total Time", "Memory Utilization (MB)"
    ]).to_csv(LOG_PATH, index=False)

def safe_read_jsonl(path: str) -> pd.DataFrame:
    """
    Safely read NDJSON (one JSON object per line).
    Returns empty DataFrame if file missing/empty.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return pd.DataFrame(columns=["datetime","epoch","moteid","temperature","humidity","light","voltage"])
    try:
        df = pd.read_json(path, lines=True)
        # Normalize canonical column names to match your clustering code (`temp`, `hum`, `light`)
        rename_map = {
            "temperature": "temp",
            "humidity": "hum"
        }
        df = df.rename(columns=rename_map)
        # Keep only needed columns (plus cluster meta)
        keep = ["datetime", "epoch", "moteid", "temp", "hum", "light", "voltage"]
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        df = df[keep].dropna(subset=["temp", "hum", "light"])
        # Cap size (keep latest)
        if len(df) > MAX_ROWS:
            df = df.tail(MAX_ROWS)
        return df
    except Exception:
        # If a line is being written while we read, try a more robust manual parse
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    rows.append(obj)
                except json.JSONDecodeError:
                    continue
        if not rows:
            return pd.DataFrame(columns=["datetime","epoch","moteid","temp","hum","light","voltage"])
        df = pd.DataFrame(rows)
        df = df.rename(columns={"temperature":"temp","humidity":"hum"})
        keep = ["datetime", "epoch", "moteid", "temp", "hum", "light", "voltage"]
        for col in keep:
            if col not in df.columns:
                df[col] = np.nan
        df = df[keep].dropna(subset=["temp", "hum", "light"])
        if len(df) > MAX_ROWS:
            df = df.tail(MAX_ROWS)
        return df

def estimate_optimal_k(X: np.ndarray, k_min=2, k_max=10) -> int:
    """
    Estimate optimal k using KMeans inertia elbow.
    Samples X for speed. Falls back to 3 if no elbow detected.
    """
    if X.shape[0] < k_min:
        return 3
    # Sample for knee detection
    if X.shape[0] > SAMPLE_FOR_K:
        idx = np.random.RandomState(RANDOM_STATE).choice(X.shape[0], SAMPLE_FOR_K, replace=False)
        Xs = X[idx]
    else:
        Xs = X

    inertias = []
    ks = list(range(k_min, k_max + 1))
    for k in ks:
        km = KMeans(n_clusters=k, n_init="auto", random_state=RANDOM_STATE)
        km.fit(Xs)
        inertias.append(km.inertia_)
    kl = KneeLocator(ks, inertias, curve="convex", direction="decreasing")
    k_star = kl.elbow if kl.elbow is not None else 3
    return max(2, int(k_star))

def run_birch(df: pd.DataFrame):
    """
    Fit Birch and return clustered df + chosen k.
    Uses 3D features: temp, hum, light
    """
    X = df[["temp", "hum", "light"]].to_numpy(dtype=float)
    if X.shape[0] < 2:
        df["cluster"] = -1
        return df, 1

    k = estimate_optimal_k(X)
    birch = Birch(n_clusters=k)
    birch.fit(X)
    labels = birch.predict(X)
    out = df.copy()
    out["cluster"] = labels
    return out, k

def memory_mb() -> float:
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024**2)

# ------------------ Dash App ------------------
app = Dash(__name__)
app.title = "Real-Time Birch 3D Clustering"

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, sans-serif", "padding": "12px"},
    children=[
        html.H2("Real-Time Birch 3D Clustering (Intel Lab Stream)"),
        html.Div(id="stats", style={"marginBottom": "10px", "whiteSpace": "pre"}),
        dcc.Graph(id="cluster-graph", style={"height": "78vh"}),
        dcc.Interval(id="tick", interval=REFRESH_MS, n_intervals=0),
        dcc.Store(id="store-prev-rows", data=0),
        dcc.Store(id="store-iter", data=0),
        dcc.Store(id="store-total-time", data=0.0)
    ]
)

@app.callback(
    Output("cluster-graph", "figure"),
    Output("stats", "children"),
    Output("store-prev-rows", "data"),
    Output("store-iter", "data"),
    Output("store-total-time", "data"),
    Input("tick", "n_intervals"),
    State("store-prev-rows", "data"),
    State("store-iter", "data"),
    State("store-total-time", "data"),
    prevent_initial_call=False
)
def update_fig(_n, prev_rows, iterno, total_time):
    t0 = time.time()

    df = safe_read_jsonl(JSONL_PATH)
    data_size = len(df)
    batch_size = max(0, data_size - int(prev_rows or 0))

    if data_size < 2:
        fig = px.scatter_3d(x=[], y=[], z=[], title="Waiting for data...")
        # Log minimal iteration (no clustering)
        exec_time = time.time() - t0
        total_time += exec_time
        mem = memory_mb()
        _log_metrics(iterno+1, data_size, batch_size, exec_time, total_time, mem)
        stats = _format_stats(iterno+1, data_size, batch_size, exec_time, total_time, mem, k="—")
        return fig, stats, data_size, iterno+1, total_time

    clustered, k = run_birch(df)

    # 3D scatter: temp-hum-light
    fig = px.scatter_3d(
        clustered,
        x="temp", y="hum", z="light",
        color="cluster",
        hover_data=["datetime", "moteid", "epoch", "voltage"],
        title=f"Birch 3D Clusters (k={k}) • Rows={data_size} (+{batch_size})"
    )
    fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))

    exec_time = time.time() - t0
    total_time += exec_time
    mem = memory_mb()

    _log_metrics(iterno+1, data_size, batch_size, exec_time, total_time, mem)
    stats = _format_stats(iterno+1, data_size, batch_size, exec_time, total_time, mem, k=k)

    return fig, stats, data_size, iterno+1, total_time

def _log_metrics(iter_no, data_size, batch_size, exec_time, total_time, mem_mb):
    row = {
        "Iteration No": iter_no,
        "Data Size": data_size,
        "Batch Size": batch_size,
        "Execution Time": round(exec_time, 4),
        "Total Time": round(total_time, 4),
        "Memory Utilization (MB)": round(mem_mb, 2)
    }
    # Append safely
    df = pd.DataFrame([row])
    # Ensure directory exists
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    df.to_csv(LOG_PATH, mode="a", index=False, header=not os.path.exists(LOG_PATH) or os.path.getsize(LOG_PATH) == 0)

def _format_stats(iter_no, data_size, batch_size, exec_time, total_time, mem_mb, k):
    return (
        f"Iteration: {iter_no}\n"
        f"Data Size: {data_size}  |  Batch Size: {batch_size}\n"
        f"k (Birch): {k}\n"
        f"Execution Time: {exec_time:.4f}s  |  Total Time: {total_time:.4f}s\n"
        f"Memory Utilization: {mem_mb:.2f} MB\n"
        f"Log CSV: {LOG_PATH}"
    )

if __name__ == "__main__":
    # Friendly tip if file not present yet
    if not os.path.exists(JSONL_PATH):
        print(f"[Info] Waiting for data. Start your streamer to append NDJSON to: {JSONL_PATH}")
    app.run(debug=True)
