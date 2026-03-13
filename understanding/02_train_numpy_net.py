import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append('../src')
from numpy_network import NeuralNetwork

#Load Data 
X = np.load('../saved_models/X.npy')   # shape: (52, n_samples)
y = np.load('../saved_models/y.npy')   # shape: (4, n_samples)
print(f"X shape: {X.shape}, y shape: {y.shape}")

#Build Network 
#52 inputs ==> 64 hidden ==> 32 hidden ==> 4 outputs (one per emotion)
nn = NeuralNetwork(layer_sizes=[52, 64, 32, 4])

#Train
#Try changing learning_rate to see different behaviours:
#mu = 0.5   → oscillates, overshoots the valley
#mu = 0.001 → crawls painfully slowly
#mu = 0.05  → the sweet spot
cost_history = nn.train(X, y, epochs=2000, learning_rate=0.05)

# Plot Cost Curve 
plt.figure(figsize=(10, 5))
plt.plot(cost_history)
plt.xlabel('Epoch')
plt.ylabel('Cost C')
plt.title('Gradient Descent: Cost Falling Toward the Valley')
plt.grid(True)
plt.tight_layout()
plt.savefig('../saved_models/cost_curve.png', dpi=150)
plt.show()
print("Cost curve saved to ../saved_models/cost_curve.png")

#Evaluate Accuracy 
output      = nn.forward(X)
predictions = np.argmax(output, axis=0)
true_labels = np.argmax(y, axis=0)
accuracy    = np.mean(predictions == true_labels)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

#Save Trained Weights 
nn.save('../saved_models/trained_weights')
print("Weights saved to ../saved_models/trained_weights.npz")
