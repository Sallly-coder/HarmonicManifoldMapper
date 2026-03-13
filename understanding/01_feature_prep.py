# This file's entire job is to convert messy audio into a clean numerical matrix 
# that becomes the input layer of my network.
# Cell 1 - Imports : I'm importing the tool that will measure where each audio file sits in the raw 52-dimensional space
import numpy as np
import os
import sys
sys.path.append('../src')
from feature_extraction import extract_features_for_numpy
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Cell 2 — RAVDESS emotion code mapping : I'm importing the tool that will measure where each audio file sits in the raw 52-dimensional space
EMOTION_MAP = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}

# Cell 3 — Load all files and extract features : This matrix X created here is my  raw data manifold : 
# 500 points floating in 52-dimensional space. Right now, if I plotted them (after PCA to 2D), I'd 
#see a meaningless cloud. Happy, sad, angry, neutral are all mixed together. This is exactly the 
#"crumpled ball of papers" from the topology blog. The network's entire purpose is to reshape this 
#space until those four groups separate.
features = []
labels   = []
data_dir = '../data/ravdess_subset'
#EDIT HERE PUT ACTUAL ADDRESS AFTER DOWNLOADING REPO IN LAPTOP
print("Loading audio files...")
for fname in sorted(os.listdir(data_dir)):
    if fname.endswith('.wav'):
        emotion_code = fname.split('-')[2]
        if emotion_code in EMOTION_MAP:
            fpath = os.path.join(data_dir, fname)
            try:
                feat = extract_features_for_numpy(fpath)
                features.append(feat)
                labels.append(EMOTION_MAP[emotion_code])
                print(f"   {fname} → {EMOTION_MAP[emotion_code]}")
            except Exception as e:
                print(f"   Skipped {fname}: {e}")

X = np.array(features)
labels = np.array(labels)
print(f"\nFeature matrix shape: {X.shape}")
print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# Cell 4 — One-hot encode labels : This establishes the 'Ground Truth' (y) for the backpropagation engine. 
# By using one-hot encoding, we ensure the network treats each emotion as a distinct, 
# non-ordinal direction in the output manifold, facilitating the 'untangling' process 
# during training.
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()                     #The LabelEncoder takes the raw categorical strings from the audio dataset (like "angry", "happy", "sad") and maps them to a unique integer index, basically provides an ID. In a neural network, if we used these integers directly (0, 1, 2), the math might mistakenly assume that "sad" (2) is "greater than" or "further away" from "angry" (0) simply because the numbers are higher.
y_int = le.fit_transform(labels)        # 'angry' is 0, 'happy' is 1, etc.

ohe = OneHotEncoder(sparse=False)       #It converts those integers into a binary vector. This ensures all emotions are equidistant in the initial mathematical space. There is no hierarchy imposed on them, they are just different directions in an 8-dimensional space.
y_onehot = ohe.fit_transform(y_int.reshape(-1,1))  # shape: (n_samples, 4)

# Cell 5 — Normalize features (important for gradient descent)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Cell 6 — Save everything
np.save('../saved_models/X.npy', X_scaled.T)       # shape: (52, n_samples)
np.save('../saved_models/y.npy', y_onehot.T)       # shape: (4, n_samples)
np.save('../saved_models/labels.npy', np.array(labels))
print("Saved. Ready for training.")
