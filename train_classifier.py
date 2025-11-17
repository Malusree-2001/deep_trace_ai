#!/usr/bin/env python3
"""Train Random Forest Classifier - Ultra Clean"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle

print("Loading data...")
df = pd.read_csv('outputs/full_train/features.csv')

top_features = ['laplacian_var', 'resid_kurtosis', 'glcm_contrast', 'fft_highfreq_ratio', 'glcm_homogeneity']
X = np.asarray(df[top_features], dtype=float)
y = np.asarray((df['label'] == 'ai'), dtype=int)

print(f"\n[1] DATASET LOADED")
print(f"Total: {len(X):,}")
ai_count = int(np.sum(y))
real_count = len(y) - ai_count
print(f"AI: {ai_count:,} | Real: {real_count:,}")

print(f"\n[2] SPLITTING DATA")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

print(f"\n[3] TRAINING")
clf = RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=10, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
print("✓ Complete!")

print(f"\n[4] EVALUATION")
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {100*acc:.2f}%")
print(f"Precision: {100*prec:.2f}%")
print(f"Recall:    {100*rec:.2f}%")
print(f"F1-Score:  {f1:.4f}")

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(f"\nTP: {int(tp):,} | TN: {int(tn):,} | FP: {int(fp):,} | FN: {int(fn):,}")

print(f"\n[5] FEATURES")
for i, (f, imp) in enumerate(zip(top_features, clf.feature_importances_), 1):
    print(f"{i}. {f}: {imp*100:5.1f}%")

with open('deeptrace_classifier.pkl', 'wb') as file:
    pickle.dump(clf, file)

print(f"\n{'='*60}")
print(f"✓ Model trained: {100*acc:.2f}% accuracy")
print(f"✓ Saved: deeptrace_classifier.pkl")
print(f"{'='*60}")
