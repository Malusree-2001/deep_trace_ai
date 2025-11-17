#!/usr/bin/env python3
"""DeepTrace Results Analysis - 100K Image Processing"""

import pandas as pd
import numpy as np

df = pd.read_csv('outputs/full_train/features.csv')

print("=" * 70)
print("DeepTrace Analysis: 100K Images")
print("=" * 70)

# Summary
print("\n[1] DATASET")
total = len(df)
real = len(df[df['label']=='real'])
ai = len(df[df['label']=='ai'])
print(f"Total: {total:,} | Real: {real:,} | AI: {ai:,}")

# Features
print("\n[2] FEATURE STATS")
features = [c for c in df.columns if c.startswith(('edge_','laplacian','resid_','fft_','blockiness','glcm_'))]
print(df[features].describe())

# Clustering
print("\n[3] CLUSTERING")
n_clusters = df['cluster'].nunique()
noise = len(df[df['cluster']==-1])
print(f"Clusters: {n_clusters} | Noise: {noise:,} ({100*noise/total:.1f}%)")

# Anomalies
print("\n[4] ANOMALIES")
anom = len(df[df['anomaly_flag']==1])
print(f"Detected: {anom:,} ({100*anom/total:.1f}%)")
print(f"Score - Mean: {df['anomaly_score'].mean():.4f} | Median: {df['anomaly_score'].median():.4f}")

# By label
print("\n[5] ANOMALY BY LABEL")
real_anom = len(df[(df['label']=='real') & (df['anomaly_flag']==1)])
ai_anom = len(df[(df['label']=='ai') & (df['anomaly_flag']==1)])
print(f"Real flagged: {real_anom:,} ({100*real_anom/real:.1f}%)")
print(f"AI flagged: {ai_anom:,} ({100*ai_anom/ai:.1f}%)")

# Features
print("\n[6] TOP FEATURES")
real_m = df[df['label']=='real'][features].mean()
ai_m = df[df['label']=='ai'][features].mean()
diffs = (ai_m - real_m).abs().sort_values(ascending=False)
for i, f in enumerate(diffs.head(5).index, 1):
    print(f"  {i}. {f}: Real={real_m[f]:.4f} | AI={ai_m[f]:.4f}")

# Metrics
print("\n[7] CLASSIFICATION METRICS")
y_t = np.array((df['label']=='ai').astype(int))
y_p = np.array((df['anomaly_flag']==1).astype(int))
tp = np.sum((y_t == 1) & (y_p == 1))
tn = np.sum((y_t == 0) & (y_p == 0))
fp = np.sum((y_t == 0) & (y_p == 1))
fn = np.sum((y_t == 1) & (y_p == 0))
acc = (tp + tn) / (tp + tn + fp + fn)
prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
print(f"Accuracy:  {acc:.4f} ({100*acc:.2f}%)")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1-Score:  {f1:.4f}")

print("\n" + "=" * 70)
print("✓ Analysis Complete!")
print("=" * 70)
