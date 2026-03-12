# 🎙️ Speech Emotion Recognition (SER) — Visualising Neural Network


---

## Overview

Objective: Build a neural network from scratch (using NumPy for the engine) to classify emotional states from audio features, focusing on visualizing the transformation of the "data manifold" at each layer.

## The Roadmap
Phase 1: Data Preparation (The High-Dimensional Input)
Before the network can "reshape" space, you need to provide the raw material. Audio signals are naturally "tangled" in their raw form.
Action: Extract MFCCs (Mel-frequency cepstral coefficients) and Chroma features from the dataset.
Mathematical Tie-in: These features represent your input vector $a^0$. You are starting with a high-dimensional space where "Happy" and "Sad" signals are mathematically interwoven.
Analogy: Think of this as a crumpled-up ball of two different colored papers. Your goal is to flatten them out without tearing them (Homeomorphism).
Phase 2: Designing the Architecture (Width & Topology)
Here, you apply the constraints of Width and Homeomorphism discussed in the core concepts.
Action: Define the number of hidden layers and neurons. Ensure the "Width" is sufficient (at least 3 neurons if you want to visualize 3D transformations) so the network doesn't run into a topological "dead end" where it cannot lift the data high enough to separate it.
Mathematical Tie-in: Define your weight matrices $w^l$ and bias vectors $b^l$ for each layer.
Phase 3: The "Hadamard" Engine (Backpropagation from Scratch)
Instead of using a library, you will implement the learning process manually to master the notation $z_j^l$ and $a_j^l$.
Action: Write the forward pass using the weighted input formula: $z^l = w^la^{l-1} + b^l$.
The Gradient: Implement the backpropagation using the Hadamard Product ($s \odot t$) to calculate the error $\delta^l$.
Logic: This ensures you are calculating the "sensitivity" of each individual neuron’s error without mixing them into a single sum prematurely.
Phase 4: Navigating the Landscape (Optimization)
Now you focus on how the network "finds" the solution in the 13,000+ dimensional cost landscape.
Action: Implement Gradient Descent with a configurable learning rate ($\eta$).
Experiment: Observe how the cost function behaves. If $\eta$ is too high, you’ll "jump" over the valley of the solution; if it's too low, you’ll crawl too slowly.
Goal: Reach the "global minimum" where the network has successfully learned the weights that separate the emotions.
Phase 5: Manifold Visualization (The Solution)
This is the final proof that the neural network has "solved" the problem by reshaping space.
Action: Use a technique like Principal Component Analysis (PCA) or t-SNE to take the activations ($a^l$) from the final hidden layer and plot them in 2D or 3D.
The Result: You should see that while the input data was a messy cloud, the output manifold is clean, with "Angry," "Happy," and "Neutral" signals residing in distinct, linearly separable regions of space.

---

## 🗂️ Project Structure

```
HarmonicManifoldMapper/
│
├── data/
│   └── README.md                  
│
├── src/
│   ├── audio_processing.py        
│   ├── feature_extraction.py      
│   │
│   ├── numpy_network.py           ( entire NumPy engine)
│   ├── activations.py             (sigmoid, relu, their derivatives)
│   └── cost.py                    (cost function + its derivative)
│
├── experiments/
│   ├── 01_feature_prep.ipynb      (prepare features as NumPy arrays)
│   ├── 02_train_numpy_net.ipynb   (train + watch cost fall)
│   └── 03_manifold_visualization.ipynb  (the PCA/t-SNE payoff)
│
├── saved_models/
│   └── .gitkeep                   (empty folder to save trained weights)
│
├── 01_signal_processing_demo.ipynb  (reference)
├── 02_feature_extraction.ipynb      (reference)
├── requirements.txt             
└── README.md                     
```

---

## 🛴 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/Sallly-coder/speech-emotion-recognition-pbl2.git
cd speech-emotion-recognition-pbl2
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
See `data/README.md` for instructions (RAVDESS / TESS).

### 4. Run training
```bash
python src/train_eval.py --data_dir data/ravdess_subset --emotions happy sad neutral
```

### 5. Explore notebooks
```bash
jupyter notebook notebooks/
```

---

## 🎯 Emotions Targeted (Mid-Term)

| Label     | Code |
|-----------|------|
| Neutral   | 01   |
| Happy     | 03   |
| Sad       | 04   |
| Angry     | 05   |

*(RAVDESS emotion codes)*

---

## 📊 Mid-Term Results (Baseline)

| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~65–70%  |
| Random Forest       | ~70–75%  |
| MLP Classifier      | ~72–78%  |

> *Results on 3-emotion subset (happy/sad/neutral), MFCC mean+std features*

---

## 🛣️ Roadmap

- [x] **Sem 4 (Mid-Term):** Pipeline setup, MFCC extraction, baseline classifiers
- [ ] **Sem 5:** 1D-CNN / LSTM models, more emotions
- [ ] **Sem 6:** Real-time audio input
- [ ] **Sem 7-8:** Deployment, UI polish, report

---

## 🔗 Links

- **GitHub Pages / Presentation:** [Link here]
- **Dataset:** [RAVDESS on Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio)

---

## 📚 References

- [Librosa Documentation](https://librosa.org/doc/)
- [RAVDESS Dataset](https://zenodo.org/record/1188976)
- [TechVidvan – Speech Emotion Recognition](https://techvidvan.com/tutorials/python-project-speech-emotion-recognition/)
- Scikit-learn User Guide: https://scikit-learn.org/stable/
