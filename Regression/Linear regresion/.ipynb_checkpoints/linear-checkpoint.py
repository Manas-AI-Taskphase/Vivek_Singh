import numpy as np

def cost_function(X, y, theta):
    m = len(y)
    h = np.dot(X, theta)
    error = h - y
    cost = (1/(2 * m)) * np.sum(error**2)
    return cost

def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = np.dot(X, theta)
        error = h - y
        gradient = (1/m) * np.dot(X.T, error)
        theta -= learning_rate * gradient

        # Calculate and record the cost for analysis
        cost = cost_function(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

# Example data
np.random.seed(0)
m = 100  # Number of examples
n = 3    # Number of features (excluding the bias term)

# Generate random features and target variable
X = np.hstack([np.ones((m, 1)), 2 * np.random.rand(m, n)])
true_theta = np.array([[3], [4], [5]])  # True parameters
y = X.dot(true_theta) + np.random.randn(m, 1)

# Initialize parameters
initial_theta = np.zeros((n + 1, 1))

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Run gradient descent
X_transpose = X.T  # Transpose of X for easier matrix multiplication
theta_learned, cost_history = gradient_descent(X, y, initial_theta, learning_rate, num_iterations)

# Print the learned parameters
print("Learned Parameters:")
print(theta_learned)

# Print the cost history for analysis
print("\nCost History during Training:")
print(cost_history[-10:])  # Print the last 10 cost values for brevity
