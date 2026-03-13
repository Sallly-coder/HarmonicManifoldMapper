import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

sys.path.append('../src')
from numpy_network import NeuralNetwork

# Load Data and Trained Weights
X      = np.load('../saved_models/X.npy')
y      = np.load('../saved_models/y.npy')
labels = np.load('../saved_models/labels.npy')

print(f'X shape      : {X.shape}')
print(f'y shape      : {y.shape}')
print(f'labels shape : {labels.shape}')
print(f'Unique labels: {np.unique(labels)}')

nn = NeuralNetwork(layer_sizes=[52, 64, 32, 4])
nn.load('../saved_models/trained_weights.npz')
print('Network loaded successfully.')

# Color map for all plots
colors = {
    'neutral': 'gray',
    'happy':   'gold',
    'sad':     'royalblue',
    'angry':   'crimson'
}

# Plot 1: Raw Input Space (before the network)
print('\nGenerating Plot 1 — Raw Input Space...')
pca  = PCA(n_components=2)
X_2d = pca.fit_transform(X.T)   # X.T → (n_samples, 52) for PCA

print(f'Explained variance by 2 PCs: {pca.explained_variance_ratio_.sum()*100:.1f}%')

plt.figure(figsize=(8, 6))
for emotion, color in colors.items():
    mask = labels == emotion
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=color, label=emotion, alpha=0.6, s=40, edgecolors='none')
plt.title('RAW INPUT SPACE — Emotions Tangled Together', fontsize=13, fontweight='bold')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../saved_models/plot_raw_input_space.png', dpi=150)
plt.show()
print('Saved: plot_raw_input_space.png')

# Plot 2: Latent Hidden Space (after the network untangles)
print('\nGenerating Plot 2 — Latent Hidden Space (PCA)...')
hidden     = nn.get_hidden_representation(X, layer=-2)
print(f'Hidden representation shape: {hidden.shape}')

pca_hidden = PCA(n_components=2)
hidden_2d  = pca_hidden.fit_transform(hidden.T)

print(f'Explained variance by 2 PCs: {pca_hidden.explained_variance_ratio_.sum()*100:.1f}%')

plt.figure(figsize=(8, 6))
for emotion, color in colors.items():
    mask = labels == emotion
    plt.scatter(hidden_2d[mask, 0], hidden_2d[mask, 1],
                c=color, label=emotion, alpha=0.6, s=40, edgecolors='none')
plt.title('LATENT SPACE — Emotion Manifolds Untangled by the Network',
          fontsize=13, fontweight='bold')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../saved_models/plot_latent_space_pca.png', dpi=150)
plt.show()
print('Saved: plot_latent_space_pca.png')

# Plot 3: t-SNE of Latent Space 
print('\nGenerating Plot 3 — t-SNE (this takes ~30 seconds, please wait)...')
tsne        = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
hidden_tsne = tsne.fit_transform(hidden.T)

plt.figure(figsize=(8, 6))
for emotion, color in colors.items():
    mask = labels == emotion
    plt.scatter(hidden_tsne[mask, 0], hidden_tsne[mask, 1],
                c=color, label=emotion, alpha=0.6, s=40, edgecolors='none')
plt.title('t-SNE of Latent Space — Emotion Clusters',
          fontsize=13, fontweight='bold')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('../saved_models/plot_latent_space_tsne.png', dpi=150)
plt.show()
print('Saved: plot_latent_space_tsne.png')

# Plot 4: Neuron Sensitivity Heatmap 
print('\nGenerating Plot 4 — Sensitivity Heatmap...')
nn.forward(X)
hidden_acts  = nn.activations[-2]   # (32, n_samples)
emotion_list = ['neutral', 'happy', 'sad', 'angry']
heatmap_data = np.zeros((len(emotion_list), hidden_acts.shape[0]))

print('\nAverage neuron activation per emotion (top 3 most active neurons):')
print('-' * 60)
for i, emotion in enumerate(emotion_list):
    mask                = labels == emotion
    avg                 = np.mean(hidden_acts[:, mask], axis=1)
    heatmap_data[i]     = avg
    top3_idx            = np.argsort(avg)[-3:][::-1]
    top3_vals           = avg[top3_idx].round(4)
    print(f'  {emotion:8s} | top neurons: {top3_idx} | activations: {top3_vals}')

plt.figure(figsize=(14, 4))
plt.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', interpolation='nearest')
plt.colorbar(label='Mean Activation')
plt.yticks(range(len(emotion_list)), emotion_list)
plt.xlabel('Hidden Neuron Index')
plt.title('Neuron Sensitivity Heatmap — Which Neurons Fire for Which Emotion?',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('../saved_models/plot_sensitivity_heatmap.png', dpi=150)
plt.show()
print('Saved: plot_sensitivity_heatmap.png')


print('\n All 4 plots saved to saved_models/')
print('  plot_raw_input_space.png')
print('  plot_latent_space_pca.png')
print('  plot_latent_space_tsne.png')
print('  plot_sensitivity_heatmap.png')
