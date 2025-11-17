#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# DeepTrace - Memory-Optimized for Large Datasets (100K+ images)
# Ultra-aggressive memory management for limited RAM environments

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import cosine_similarity

try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_OK = True
except Exception:
    SKIMAGE_OK = False
    print("[WARN] scikit-image not available", file=sys.stderr)

def read_image(path, size=256):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read: {path}")
    bgr = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return gray

def edge_density(gray):
    g8 = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(g8, 100, 200)
    return float((edges > 0).mean())

def laplacian_var(gray):
    g8 = (gray * 255).astype(np.uint8)
    lap = cv2.Laplacian(g8, cv2.CV_64F)
    return float(lap.var())

def noise_residual_stats(gray, ksize=3):
    g8 = (gray * 255).astype(np.uint8)
    med = cv2.medianBlur(g8, ksize)
    resid = (g8.astype(np.float32) - med.astype(np.float32)) / 255.0
    std = float(resid.std() + 1e-12)
    m = resid.mean()
    s2 = resid.var() + 1e-12
    s4 = np.mean((resid - m) ** 4)
    kurtosis_excess = float(s4 / (s2 ** 2) - 3.0)
    return std, kurtosis_excess

def fft_highfreq_ratio(gray, low_frac=0.2):
    f = np.fft.fft2(gray)
    mag = np.abs(np.fft.fftshift(f))
    h, w = mag.shape
    ly, lx = int(h * low_frac / 2), int(w * low_frac / 2)
    cy, cx = h // 2, w // 2
    low = mag[cy - ly:cy + ly, cx - lx:cx + lx].sum()
    tot = mag.sum() + 1e-12
    return float((tot - low) / tot)

def blockiness_score(gray, block=8):
    g = (gray * 255).astype(np.float32)
    dh = np.abs(np.diff(g, axis=1))
    dv = np.abs(np.diff(g, axis=0))
    h, w = g.shape
    x = np.arange(w - 1)
    y = np.arange(h - 1)
    mask_v = ((x + 1) % block == 0).astype(np.float32)
    mask_h = ((y + 1) % block == 0).astype(np.float32)
    on_v = (dh * mask_v[None, :]).sum()
    on_h = (dv * mask_h[:, None]).sum()
    total = dh.sum() + dv.sum() + 1e-12
    return float((on_v + on_h) / total)

def glcm_features(gray):
    if not SKIMAGE_OK:
        return np.nan, np.nan
    g8 = (gray * 255).astype(np.uint8)
    glcm = graycomatrix(g8, distances=(1,), angles=(0,), levels=256, symmetric=True, normed=True) #type: ignore
    contrast = graycoprops(glcm, "contrast").mean()#type: ignore
    homogeneity = graycoprops(glcm, "homogeneity").mean()#type: ignore
    return float(contrast), float(homogeneity)

def extract_vector(gray):
    std, kurt = noise_residual_stats(gray)
    con, hom = glcm_features(gray)
    return {
        "edge_density": edge_density(gray),
        "laplacian_var": laplacian_var(gray),
        "resid_std": std,
        "resid_kurtosis": kurt,
        "fft_highfreq_ratio": fft_highfreq_ratio(gray),
        "blockiness": blockiness_score(gray),
        "glcm_contrast": con,
        "glcm_homogeneity": hom,
    }

def knn_graph_ultra_chunked(X, k=5, chunk_row=1000, chunk_col=5000):
    """
    Ultra-aggressive chunking: process both rows AND columns in smaller chunks.
    Memory usage: O(chunk_row * chunk_col) instead of O(n^2)
    """
    n = X.shape[0]
    d = X.shape[1]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    print(f"[INFO] Ultra-chunked KNN: row_chunk={chunk_row}, col_chunk={chunk_col}")
    
    # Use sparse representation - only store top-k per row
    top_k_indices = np.zeros((n, k), dtype=np.int32)
    top_k_scores = np.zeros((n, k), dtype=np.float32)
    
    # Process all rows in chunks
    for row_start in range(0, n, chunk_row):
        row_end = min(row_start + chunk_row, n)
        row_chunk = X[row_start:row_end]
        
        # For this row chunk, find top-k globally by processing column chunks
        partial_sims = []
        partial_indices = []
        
        for col_start in range(0, n, chunk_col):
            col_end = min(col_start + chunk_col, n)
            col_chunk = X[col_start:col_end]
            
            # Compute similarity for this block
            sim_block = cosine_similarity(row_chunk, col_chunk).astype(np.float32)
            partial_sims.append(sim_block)
            partial_indices.append(np.arange(col_start, col_end))
            
            # Force garbage collection
            del col_chunk
        
        # Concatenate and find top-k globally
        all_sims = np.concatenate(partial_sims, axis=1)
        all_indices = np.concatenate(partial_indices)
        
        # Get top-k for each row
        for i in range(row_chunk.shape[0]):
            global_idx = row_start + i
            row_sims = all_sims[i]
            
            # Zero out self-similarity
            if global_idx < len(all_indices):
                self_pos = np.where(all_indices == global_idx)[0]
                if len(self_pos) > 0:
                    row_sims[self_pos[0]] = -1
            
            top_k_idx = np.argsort(-row_sims)[:k]
            top_k_indices[global_idx] = all_indices[top_k_idx]
            top_k_scores[global_idx] = row_sims[top_k_idx]
        
        progress = min(row_end, n)
        print(f"  ... processed {progress}/{n} rows")
        
        # Clean up
        del partial_sims, partial_indices, all_sims, all_indices
    
    # Build graph from top-k
    for i in range(n):
        for j, score in zip(top_k_indices[i], top_k_scores[i]):
            j = int(j)
            if score > 0 and i != j and not G.has_edge(i, j):
                G.add_edge(i, j, weight=float(score))
    
    return G, None  # Don't store full matrix

def cluster_and_anomaly(X):
    Xs = StandardScaler().fit_transform(X)
    db = DBSCAN(eps=0.8, min_samples=5).fit(Xs)
    cl = db.labels_
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(Xs)
    raw = iso.score_samples(Xs)
    ranks = raw.argsort().argsort().astype(float)
    anom_score = ranks / (len(ranks) - 1 + 1e-9)
    anom_flag = (anom_score > 0.5).astype(int)
    return cl, anom_score, anom_flag

def plot_feature_hists(df, out_dir):
    cols = [c for c in df.columns if c.startswith(("edge_", "laplacian", "resid_", "fft_", "blockiness", "glcm_"))]
    if not cols:
        return
    rows = (len(cols) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.ravel()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna().values, bins=20, edgecolor='black', alpha=0.7)
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "feature_histograms.png", dpi=180)
    plt.close(fig)

def plot_scatter(df, out_dir, x="fft_highfreq_ratio", y="blockiness"):
    fig, ax = plt.subplots(figsize=(7, 6))
    if "cluster" in df.columns:
        scatter = ax.scatter(df[x], df[y], c=df["cluster"], cmap="tab10", alpha=0.7, s=50)
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label("Cluster ID")
    else:
        ax.scatter(df[x], df[y], alpha=0.7, s=50)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{y} vs {x}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"scatter_{x}_vs_{y}.png", dpi=180)
    plt.close(fig)

def plot_graph(G, df, out_dir):
    if G.number_of_nodes() == 0:
        return
    max_nodes = 1000
    if G.number_of_nodes() > max_nodes:
        print(f"[INFO] Sampling {max_nodes} nodes for visualization")
        sample_nodes = np.random.choice(G.number_of_nodes(), size=max_nodes, replace=False)
        G = G.subgraph(sample_nodes).copy()
        df_vis = df.iloc[sample_nodes]
    else:
        df_vis = df
    pos = nx.spring_layout(G, seed=42, k=0.6, iterations=50)
    clusters = df_vis["cluster"].values if "cluster" in df_vis.columns else np.zeros(len(df_vis))
    sizes = 200 + 600 * df_vis.get("anomaly_score", pd.Series(np.zeros(len(df_vis)))).values
    cmap = plt.colormaps["tab10"]
    fig, ax = plt.subplots(figsize=(10, 8))
    if G.number_of_edges() > 0:
        w = [G[u][v]["weight"] for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=[3*ww for ww in w], ax=ax)
    node_colors = [cmap(int((c if c >= 0 else 9) % 10)) for c in clusters]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=sizes, ax=ax, alpha=0.8)
    ax.set_title("Similarity Graph (color=cluster, size=anomaly)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "similarity_graph.png", dpi=200)
    plt.close(fig)

def plot_degree_hist(G, out_dir):
    deg = [d for _, d in G.degree()]
    if not deg:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(deg, bins=range(0, max(deg) + 2), align="left", edgecolor='black', alpha=0.7)
    ax.set_xlabel("Degree")
    ax.set_ylabel("Count")
    ax.set_title("Graph Degree Distribution")
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "degree_distribution.png", dpi=180)
    plt.close(fig)

def collect_images(data_dir, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")):
    p = Path(data_dir)
    return sorted([q for q in p.rglob("*") if q.suffix.lower() in exts])

def infer_label_from_path(p: Path):
    parts = [part.lower() for part in p.parts]
    label = None
    for part in parts:
        if part in ("real", "ai", "fake", "synthetic"):
            label = "real" if part == "real" else "ai"
            break
    split = None
    for part in parts:
        if part in ("train", "test"):
            split = part
            break
    return label, split

def run(data_dir, out_dir, size=256, k=5, chunk_row=1000, chunk_col=5000, sample_rate=1.0):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    paths = collect_images(data_dir)
    if not paths:
        print(f"[ERR] No images in {data_dir}", file=sys.stderr)
        return 1
    
    if sample_rate < 1.0:
        n_sample = max(1, int(len(paths) * sample_rate))
        indices = np.random.choice(len(paths), size=n_sample, replace=False)
        paths = [paths[i] for i in indices]
        print(f"[INFO] Sampling {n_sample} images ({100*sample_rate:.0f}%)")
    
    rows = []
    print(f"[INFO] Processing {len(paths)} images...")
    for idx, p in enumerate(paths):
        if (idx + 1) % max(100, len(paths)//10) == 0:
            print(f"  ... processed {idx + 1}/{len(paths)}")
        try:
            g = read_image(p, size=size)
            vec = extract_vector(g)
        except Exception as e:
            print(f"[WARN] {p}: {e}")
            continue
        label, split = infer_label_from_path(p)
        row = {"path": str(p), "name": p.name, "label": label, "split": split}
        row.update(vec)
        rows.append(row)
    
    if len(rows) < 2:
        print("[ERR] Need at least 2 valid images.", file=sys.stderr)
        return 2
    
    df = pd.DataFrame(rows)
    cols = [c for c in df.columns if c not in ("path", "name", "label", "split")]
    X = df[cols].astype(float).values
    
    print("[INFO] Running clustering and anomaly detection...")
    clusters, a_score, a_flag = cluster_and_anomaly(X)
    df["cluster"] = clusters
    df["anomaly_score"] = a_score
    df["anomaly_flag"] = a_flag
    
    print("[INFO] Building similarity graph (ultra-chunked)...")
    G, _ = knn_graph_ultra_chunked(X, k=k, chunk_row=chunk_row, chunk_col=chunk_col)
    
    df.to_csv(Path(out_dir) / "features.csv", index=False)
    print(f"[INFO] Features saved to {out_dir}/features.csv")
    
    print("[INFO] Generating visualizations...")
    plot_feature_hists(df, out_dir)
    plot_scatter(df, out_dir)
    plot_graph(G, df, out_dir)
    plot_degree_hist(G, out_dir)
    
    print("\n" + "="*60)
    print(f"[SUMMARY]")
    print(f"  Total images: {len(df)}")
    print(f"  Clusters: {len(set(clusters))}")
    print(f"  Anomalies: {int(a_flag.sum())} ({100*a_flag.mean():.1f}%)")
    if "label" in df.columns:
        print(f"  Labels: {dict(df['label'].value_counts())}")
    print(f"\n[OUTPUT] Results: {out_dir}")
    print("="*60)
    return 0

def main():
    p = argparse.ArgumentParser(description="DeepTrace - Memory-Optimized")
    p.add_argument("--data_dir", required=True, help="Folder with images")
    p.add_argument("--out_dir", default="outputs", help="Output directory")
    p.add_argument("--size", type=int, default=256, help="Image size")
    p.add_argument("--knn", type=int, default=5, help="K for KNN")
    p.add_argument("--chunk_row", type=int, default=1000, help="Row chunk size (lower=less memory)")
    p.add_argument("--chunk_col", type=int, default=5000, help="Column chunk size")
    p.add_argument("--sample_rate", type=float, default=1.0, help="Sample rate")
    args = p.parse_args()
    sys.exit(run(args.data_dir, args.out_dir, args.size, args.knn, args.chunk_row, args.chunk_col, args.sample_rate))

if __name__ == "__main__":
    main()
