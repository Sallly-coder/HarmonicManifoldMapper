# This notebook's entire job is to convert messy audio into a clean numerical matrix 
# that becomes the input layer of my network.
# Cell 1 - Imports : I'm importing the tool that will measure where each audio file sits in the raw 52-dimensional space
import numpy as np
import os
import sys
sys.path.append('../src')
from feature_extraction import extract_features_for_numpy

# Cell 2 — RAVDESS emotion code mapping : I'm importing the tool that will measure where each audio file sits in the raw 52-dimensional space
EMOTION_MAP = {
    '01': 'neutral',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}

# Cell 3 — Load all files and extract features
features = []
labels   = []
data_dir = '../data/ravdess_subset'  # point to your data folder
#MAKE THE CHANGE HERE AFTER DOWNLOADING THIS REPO IN LAPTOP
for fname in os.listdir(data_dir):
    if fname.endswith('.wav'):
        emotion_code = fname.split('-')[2]
        if emotion_code in EMOTION_MAP:
            fpath = os.path.join(data_dir, fname)
            feat = extract_features_for_numpy(fpath)
            features.append(feat)
            labels.append(EMOTION_MAP[emotion_code])

X = np.array(features)   # shape: (num_samples, 52)
print(f"Feature matrix shape: {X.shape}")
print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

# Cell 4 — One-hot encode labels
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

le = LabelEncoder()
y_int = le.fit_transform(labels)        # 'angry' is 0, 'happy' is 1, etc.

ohe = OneHotEncoder(sparse=False)
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
