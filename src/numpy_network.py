import numpy as np
from activations import sigmoid, sigmoid_prime, softmax
from cost import quadratic_cost, quadratic_cost_derivative


class NeuralNetwork:
    """
    A Neural Network built from scratch using only NumPy.
    Every operation maps directly to the four BP equations.
    """

    def __init__(self, layer_sizes):
        """
        layer_sizes: list of integers
        e.g. [52, 64, 32, 4] means:
             52 input features to 64 neurons to 32 neurons to 4 emotion outputs
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # Initialize weights and biases
        # Xavier initialization: prevents neurons from saturating immediately
        # (saturated neurons → (sigma)' ≈ 0 so learning dies, as we studied in BP1)
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) \
                * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)

        # Storage for forward pass values (needed by backprop)
        self.zs = []           # weighted inputs at each layer
        self.activations = []  # activations at each layer


    # ─────────────────────────────────────────────
    # FORWARD PASS
    # Equation: z^l = w^l * a^(l-1) + b^l
    #           a^l = sigma(z^l)
    # ─────────────────────────────────────────────
    def forward(self, X):
        """
        X: input matrix, shape (num_features, num_samples)
        Returns: output activations, shape (num_classes, num_samples)
        """
        self.zs = []
        self.activations = [X]   # a^0 = input itself

        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b          # z^l = w^l * a^(l-1) + b^l
            self.zs.append(z)

            # Use sigmoid for hidden layers, softmax for output
            if i == self.num_layers - 2:
                a = softmax(z)            # output layer
            else:
                a = sigmoid(z)            # hidden layers

            self.activations.append(a)

        return a   # final output = a^L


    # ─────────────────────────────────────────────
    # BACKWARD PASS — The Four BP Equations
    # ─────────────────────────────────────────────
    def backward(self, y_true):
        """
        y_true: correct labels, one-hot encoded
                shape (num_classes, num_samples)
        Returns: gradients for all weights and biases
        """
        m = y_true.shape[1]   # number of training samples

        # Initialize gradient storage
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # BP1: Error at the output layer
        # Start at the output. Compute how wrong each output neuron is, weighted by how sensitive it currently is to change. That's the error delta
        # delta^L = delC/dela^L <hardmard> (sigma)'(z^L)
        # For softmax + quadratic cost this simplifies to (a - y)
        delta = quadratic_cost_derivative(self.activations[-1], y_true)

        # BP4: Weight gradient at output layer 
        #Every weight gradient in the entire network is just the product of two numbers: what was coming in and how wrong the output was
        # delC/delw = a_in * delta_out
        nabla_w[-1] = np.dot(delta, self.activations[-2].T) / m

        # BP3: Bias gradient at output layer
        #the amount you need to slide each neuron's space is exactly measured by how wrong that neuron currently is.
        # delC/delb = delta
        nabla_b[-1] = np.sum(delta, axis=1, keepdims=True) / m

        # BP2: Propagate error BACKWARDS through all layers 
        # We already started with BP1 at the output. Apply BP2 once to get delta at layer L-1. Apply BP2 again to get delta at layer L-2. Keep going. You now have delta (the error) at every single neuron in the entire network. 
        # delta^l = ((w^(l+1))^T * delta^(l+1)) <Hardmard> (sigma)'(z^l)
        for l in range(2, self.num_layers):
            z = self.zs[-l]
            sp = sigmoid_prime(z)                              # σ'(z^l)
            delta = np.dot(self.weights[-l+1].T, delta) * sp  # <Hardmard> is * in NumPy

            nabla_w[-l] = np.dot(delta, self.activations[-l-1].T) / m  # BP4
            nabla_b[-l] = np.sum(delta, axis=1, keepdims=True) / m     # BP3

        return nabla_w, nabla_b


    # ─────────────────────────────────────────────
    # GRADIENT DESCENT UPDATE
    # for w it is w-(mu)*(del)C/(del)w
    # for b it is b-(mu)*(del)C/(del)b
    # ─────────────────────────────────────────────
    def update(self, nabla_w, nabla_b, learning_rate):
        self.weights = [w - learning_rate * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases  = [b - learning_rate * nb
                        for b, nb in zip(self.biases, nabla_b)]


    # ─────────────────────────────────────────────
    # FULL TRAINING LOOP
    # ─────────────────────────────────────────────
    def train(self, X_train, y_train, epochs, learning_rate, verbose=True):
        """
        X_train: shape (num_features, num_samples)
        y_train: one-hot encoded, shape (num_classes, num_samples)
        """
        cost_history = []

        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X_train)

            # Compute cost
            cost = quadratic_cost(output, y_train)
            cost_history.append(cost)

            # Backward pass
            nabla_w, nabla_b = self.backward(y_train)

            # Update weights and biases
            self.update(nabla_w, nabla_b, learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Cost: {cost:.6f}")

        return cost_history


    # ─────────────────────────────────────────────
    # SAVE / LOAD WEIGHTS
    # ─────────────────────────────────────────────
    def save(self, filepath):
        np.savez(filepath,
                 weights=np.array(self.weights, dtype=object),
                 biases=np.array(self.biases, dtype=object))

    def load(self, filepath):
        data = np.load(filepath, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases  = list(data['biases'])


    # ─────────────────────────────────────────────
    # GET HIDDEN LAYER ACTIVATIONS
    # (used for manifold visualization)
    # ─────────────────────────────────────────────
    def get_hidden_representation(self, X, layer=-2):
        """
        Returns activations at the specified layer.
        layer=-2 means the last hidden layer (before output).
        This is what you feed into PCA/t-SNE to visualize manifolds.
        """
        self.forward(X)
        return self.activations[layer]
