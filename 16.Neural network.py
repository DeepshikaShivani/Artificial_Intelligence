import random

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + pow(2.71828, -x))  # Approximation of the sigmoid function using Euler's number

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(self.hidden_size)] for _ in range(self.input_size)]
        self.bias_input_hidden = [random.uniform(-1, 1) for _ in range(self.hidden_size)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(self.output_size)] for _ in range(self.hidden_size)]
        self.bias_hidden_output = [random.uniform(-1, 1) for _ in range(self.output_size)]

    def forward(self, X):
        # Input to hidden layer
        hidden_layer_activation = [0] * self.hidden_size
        for i in range(self.hidden_size):
            activation = self.bias_input_hidden[i]
            for j in range(self.input_size):
                activation += X[j] * self.weights_input_hidden[j][i]
            hidden_layer_activation[i] = sigmoid(activation)
        
        # Hidden to output layer
        output = [0] * self.output_size
        for i in range(self.output_size):
            activation = self.bias_hidden_output[i]
            for j in range(self.hidden_size):
                activation += hidden_layer_activation[j] * self.weights_hidden_output[j][i]
            output[i] = sigmoid(activation)
        return output

    def backward(self, X, y, output, learning_rate):
        output_error = [0] * self.output_size
        for i in range(self.output_size):
            output_error[i] = y[i] - output[i]

        output_delta = [0] * self.output_size
        for i in range(self.output_size):
            output_delta[i] = output_error[i] * sigmoid_derivative(output[i])

        hidden_layer_error = [0] * self.hidden_size
        for i in range(self.hidden_size):
            error = 0
            for j in range(self.output_size):
                error += output_delta[j] * self.weights_hidden_output[i][j]
            hidden_layer_error[i] = error * sigmoid_derivative(hidden_layer_activation[i])

        # Update weights and biases
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] += hidden_layer_activation[i] * output_delta[j] * learning_rate
            self.bias_hidden_output[i] += output_delta[j] * learning_rate

        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += X[i] * hidden_layer_error[j] * learning_rate
            self.bias_input_hidden[j] += hidden_layer_error[j] * learning_rate

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            for i in range(len(X)):
                output = self.forward(X[i])
                self.backward(X[i], y[i], output, learning_rate)
                if epoch % 1000 == 0:
                    loss = sum([(y[i][j] - output[j]) ** 2 for j in range(len(y[i]))])
                    print(f"Epoch {epoch}: Loss {loss:.4f}")

# Example usage:
# Create a simple dataset
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [[0], [1], [1], [0]]

# Initialize and train the neural network
input_size = 2
hidden_size = 4
output_size = 1
epochs = 10000
learning_rate = 0.1

nn = NeuralNetwork(input_size, hidden_size, output_size)
nn.train(X, y, epochs, learning_rate)

# Make predictions
predictions = [nn.forward(x) for x in X]
print("\nFinal Predictions:")
print(predictions)
