import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

from google.colab import drive
drive.mount('/content/drive/')

### ** Caution ** I imported data multiple times please change import location for smooth working or wait just change the address below
test_add = '/Users/aayus/Desktop/FMLA1Q1Data_test.csv'
train_add = '/Users/aayus/Desktop/FMLA1Q1Data_train.csv'
df = pd.read_csv(train_add)
df.columns = ['col1', 'col2', 'col3']
df
import statsmodels.api as sm

# Define X and y
X = df[['col1', 'col2']]
y = df['col3']

# Add constant term to X
X = sm.add_constant(X)

# Fit the model
model = sm.OLS(y, X).fit()

# Print the summary
print(model.summary())



# `Problem 1 (i)`


y =  y.values #df['col3'].values
X =  X.values #np.c_[np.ones(X.shape[0]), X]  # Add a column of ones for the intercept

# Calculate the OLS estimate
w_ml = np.linalg.inv(X.T @ X) @ X.T @ y

# Print the estimated coefficients
print("Estimated coefficients:", w_ml)


df.describe()
# `Problem 1 (ii)`


# Convert to numpy arrays
X = np.array(X)
y = np.array(y)


# Initialize parameters (weights) for Gradient Descent
weights = np.zeros(X.shape[1])

# Define hyperparameters
learning_rate = 0.05
n_iterations = 60

# Array to store the distance between GD weights and ML weights
distances = []

# Gradient Descent
for i in range(n_iterations):
    # Calculate predictions
    predictions = np.dot(X, weights)

    # Calculate the error
    error = predictions - y

    # Calculate the MSE (Mean Squared Error)
    mse = np.mean(error ** 2)

    # Calculate the gradient
    gradient = (2 / X.shape[0]) * np.dot(X.T, error)  # Note the factor of 2 for MSE

    # Update the weights
    weights -= learning_rate * gradient

    # Calculate and store the distance between GD weights and ML weights
    distance = np.linalg.norm(weights - w_ml)
    distances.append(distance)

    # Print MSE every 100 iterations (optional)
    if i % 10 == 0:
        print(f"Iteration {i}: MSE = {mse}, Distance to w_ml = {distance}")

print("#" * 20)
# Output the weights
print("Weights after Gradient Descent:", weights)
print("#" * 20)



plt.plot(range(n_iterations), distances, marker='o')
plt.xlabel('Iteration')
plt.ylabel(r'$||w^t - w_{ML}||_2$')
plt.title(r'$||w^t - w_{ML}||_2$ Distance Over Iterations')
plt.show()

df_train = pd.read_csv(train_add)
df_test = pd.read_csv(test_add)
df_train.columns = ['col1', 'col2', 'col3']
df_test.columns = ['col1', 'col2', 'col3']

X = df_train[['col1', 'col2']]
y = df_train['col3']
X_test = df_test[['col1', 'col2']]
y_test = df_test['col3']



# Assuming X_test is your test feature matrix and w_ml is your learned weight vector from linear regression
# If you haven't added an intercept column (all ones), you should do that now:
# Add a column of ones to X_test for the intercept term
X_test_with_intercept = np.c_[np.ones(X_test.shape[0]), X_test]

# Predict y values using w_ml
y_hat = np.dot(X_test_with_intercept, w_ml)






# Plot predictions vs actual values
plt.figure(figsize=(6.4, 4.8))
plt.scatter(y_test, y_hat, color='blue', label='Predictions vs Actual')
plt.xlabel('Actual Values (y)')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual Values (Linear Regression)')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--', label='Perfect Prediction')
plt.legend()
plt.show()

# Plot the error
plt.figure(figsize=(6.4, 4.8))
plt.plot(y_test - y_hat, label='Error')
plt.xlabel('Data Point')
plt.ylabel('Error')
plt.title(r'Error over Data Points (Linear Regression)')
plt.legend()
plt.show()

# `Problem 1 (iii)`
df = pd.read_csv(train_add)
df.columns = ['col1', 'col2', 'col3']

# Define X and y
X = df[['col1', 'col2']]
y = df['col3']

y =  y.values #df['col3'].values
X =  X.values #np.c_[np.ones(X.shape[0]), X]

X = np.c_[np.ones(X.shape[0]), X]

# SGD with batch size of 100
batch_size = 100



# Initialize parameters (weights)
weights_sgd = np.zeros(X.shape[1])  # Include intercept term

# Shuffle the data initially
data = np.hstack((X, y.reshape(-1, 1)))
np.random.shuffle(data)


# Define the learning rate and number of epochs
learning_rate_sgd = 0.05
n_epochs = 60

# Array to store the distance between SGD weights and ML weights
distances = []

# SGD
for epoch in range(n_epochs):
    # Shuffle the data at the start of each epoch
    np.random.shuffle(data)
    X_shuffled = data[:, :-1]
    y_shuffled = data[:, -1]

    for i in range(0, len(y), batch_size):
        # Define the current batch
        X_batch = X_shuffled[i:i + batch_size]
        y_batch = y_shuffled[i:i + batch_size]

        # Calculate predictions for the batch
        predictions_batch = np.dot(X_batch, weights_sgd)

        # Calculate the error for the batch
        error_batch = predictions_batch - y_batch

        # Calculate the MSE for the batch
        mse_batch = np.mean(error_batch ** 2)

        # Calculate the gradient for the batch
        gradient_batch = (2 / batch_size) * np.dot(X_batch.T, error_batch)  # Note the factor of 2 for MSE

        # Update the weights
        weights_sgd -= learning_rate_sgd * gradient_batch

    # Calculate and store the distance between SGD weights and ML weights
    distance = np.linalg.norm(weights_sgd - w_ml)
    distances.append(distance)

    # Print MSE every 10 epochs (optional)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: MSE = {mse_batch}, Distance to w_ml = {distance}")

print("#"*20)
# Output the weights
print("Weights after Stochastic Gradient Descent:", weights_sgd)
print("#"*20)

# Plot the distance over epochs
plt.plot(range(n_epochs), distances, marker='o')
plt.xlabel('Epoch')
plt.ylabel(r'$||w^t - W_{ML}||_2$')
plt.title(r'$||w^t - W_{ML}||_2$ Estimate Over Epoch')
plt.show()

# `Problem 1 (iv)`
df_train = pd.read_csv(train_add)
df_test = pd.read_csv(test_add)
df_train.columns = ['col1', 'col2', 'col3']
df_test.columns = ['col1', 'col2', 'col3']
X = df_train[['col1', 'col2']]
y = df_train['col3']
X_test = df_test[['col1', 'col2']]
y_test = df_test['col3']

X.shape

# Gradient Descent for Ridge Regression
def ridge_regression_gradient_descent(X, y, lam, learning_rate=0.05, num_iterations=1000):
    m, n = X.shape
    w = np.zeros(n)
    for i in range(num_iterations):
        predictions = X.dot(w)
        errors = predictions - y
        gradient = (1/m) * X.T.dot(errors) + (2 * lam * w)
        w = w - learning_rate * gradient
    return w

# Compute Mean Squared Error (MSE)
def compute_mse(X, y, w):
    predictions = X.dot(w)
    error = predictions - y
    mse = (1/len(y)) * np.sum(error**2)
    return mse

# Cross-validation to find best lambda (regularization parameter)
def cross_validation(X, y, lambdas, learning_rate=0.001, num_iterations=1000, folds=5):
    m, n = X.shape
    fold_size = m // folds
    validation_errors = []

    # Shuffle data for cross-validation
    indices = np.random.permutation(m)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Perform cross-validation
    for lam in lambdas:
        fold_errors = []
        for fold in range(folds):
            # Validation set for this fold
            X_val = X_shuffled[fold * fold_size:(fold + 1) * fold_size]
            y_val = y_shuffled[fold * fold_size:(fold + 1) * fold_size]

            # Training set for this fold
            X_train = np.concatenate((X_shuffled[:fold * fold_size], X_shuffled[(fold + 1) * fold_size:]))
            y_train = np.concatenate((y_shuffled[:fold * fold_size], y_shuffled[(fold + 1) * fold_size:]))

            # Train the model on the training set
            w = ridge_regression_gradient_descent(X_train, y_train, lam, learning_rate, num_iterations)

            # Compute validation error
            val_error = compute_mse(X_val, y_val, w)
            fold_errors.append(val_error)

        validation_errors.append(np.mean(fold_errors))

    return lambdas, validation_errors

# Train final model with the best lambda
def train_final_model(X_train, y_train, best_lambda, learning_rate=0.001, num_iterations=1000):
    return ridge_regression_gradient_descent(X_train, y_train, best_lambda, learning_rate, num_iterations)



X_train = X[:600].values
y_train = y[:600].values
X_val = X[600:].values
y_val = y[600:].values
X_test = X_test
y_test = y_test

# Define a range of lambdas for cross-validation
lambdas = np.logspace(-2, 2, 50)

# Perform cross-validation
lambdas, validation_errors = cross_validation(X_train, y_train, lambdas)

# Plot validation error as a function of lambda
plt.plot(lambdas, validation_errors)
plt.xscale('log')
plt.xlabel('Lambda (Regularization Parameter)')
plt.ylabel('Validation Error (MSE)')
plt.title('Validation Error vs Lambda')
plt.show()

# Find the best lambda
best_lambda_index = np.argmin(validation_errors)
best_lambda = lambdas[best_lambda_index]

print(f'Best lambda: {best_lambda}')

# Train final model with best lambda
wR = train_final_model(X_train, y_train, best_lambda)

# Compute test error
test_error = compute_mse(X_test, y_test, wR)
print(f'Test error: {test_error}')

# Define X and y for training
X_train = df_train[['col1', 'col2']]
y_train = df_train['col3']

# Add a column of ones to X for the intercept term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

# Define lambda value
lambda_value = 0.01

# Calculate w_r using the closed-form solution
n_samples, n_features = X_train.shape
identity_matrix = np.eye(n_features)
identity_matrix[0, 0] = 0  # Don't regularize the intercept term

# Compute w_r
XTX = np.dot(X_train.T, X_train)
XTy = np.dot(X_train.T, y_train)
w_r = np.linalg.inv(XTX + lambda_value * identity_matrix) @ XTy

print(f"Weights for lambda = {lambda_value}:")
print(w_r)

print(w_r)
print(w_ml)
from sklearn.metrics import mean_squared_error

# Define X and y for testing
X_test = df_test[['col1', 'col2']]
y_test = df_test['col3']

# Add a column of ones to X for the intercept term
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Convert to numpy arrays
X_test = np.array(X_test)
y_test = np.array(y_test)

# Calculate predictions for w_r
y_pred_wr = np.dot(X_test, w_r)

# Calculate predictions for w_ml
y_pred_wml = np.dot(X_test, w_ml)

# Calculate test error for w_r
test_error_wr = mean_squared_error(y_test, y_pred_wr)

# Calculate test error for w_ml
test_error_wml = mean_squared_error(y_test, y_pred_wml)

# Calculate the difference in test error
difference_in_test_error = test_error_wr - test_error_wml

# Print the difference in test error
print(f"Difference in test error (w_r - w_ml): {difference_in_test_error}")

# Plot the difference in test error
plt.plot(difference_in_test_error, label='Difference in Test Error')
plt.xlabel('Data Point')
plt.ylabel('Difference in Test Error')
plt.title('Difference in Test Error Between w_r and w_ml')
plt.legend()
plt.show()

print(test_error_wr)
print(test_error_wml)
# `Problem 1 (v)`




df_train = pd.read_csv(train_add)
df_test = pd.read_csv(test_add)
df_train.columns = ['col1', 'col2', 'col3']
df_test.columns = ['col1', 'col2', 'col3']

# Define X and y for testing
X = df_train[['col1', 'col2']].values
y = df_train['col3'].values
X_test = df_test[['col1', 'col2']].values
y_test = df_test['col3'].values

# import numpy as np
# sigma = 1.0  # Set the sigma value for the Gaussian kernel

# for i in range(X.shape[0]):
#     for j in range(X.shape[0]):
#         kernel_matrix[i, j] = np.exp(-np.linalg.norm(X[i] - X[j]) ** 2 / (2 * sigma ** 2))

# print("Kernel Matrix:")
# print(kernel_matrix)

X_test[0], y_test[0]

# -----------------------------------------------------------

def rbf(x1, x2, gamma):
  # x1, x2 are 1-D arrays/vectors
  # rbf = exp( -gamma * ||x1 - x2||^2 )

  dist = np.linalg.norm(x1 - x2)  # Euclidean distance
  return np.exp( -gamma * (dist**2) )

  # dim = len(x1)  # less efficient but more clear
  # sum = 0.0
  # for i in range(dim):
  #   sum += (x1[i] - x2[i]) * (x1[i] - x2[i])
  # return np.exp( -gamma * sum )

# -----------------------------------------------------------

def compute_output(w, x, X, gamma):
  # x is a 1D array / vector of predictors
  # w is an array of weights
  # X is a 2D matrix of all training data

  N = len(X)  # number of train items
  sum = 0.0
  for i in range(N):
    xx = X[i]  # train item as 1 vector
    k = rbf(x, xx, gamma)  # kernel value
    sum += w[i] * k
  return sum

# -----------------------------------------------------------

print("\nLightweight kernel ridge regression from scratch ")
np.random.seed(1)  # not used -- no randomness
np.set_printoptions(precision=4, suppress=True)



print("\nX values: ")
print(X)
print("\nTarget y values: ")
print(y)

# wK = y
# wKK' = yK' where K' is inverse(K)
# w = yK'

# 1. make kernel matrix K
N = len(X)
gamma = 1.0
alpha = 0.001  # regularization
print("\nSetting alpha = %0.4f gamma = %0.2f " \
  % (alpha, gamma))

K = np.zeros((N,N))
for i in range(N):
  for j in range(N):
    K[i][j] = rbf(X[i], X[j], gamma)

print("\nK = " )
print(K)

# 2. add regularization term on diagonal
for i in range(N):
  K[i][i] += alpha

# 3. compute inverse of modified K matrix
Kinv = np.linalg.inv(K)
print("\nKinv = ")
# print(Kinv)                   Please Uncomment and print the value

# 4. compute model weights using K inverse
wts = np.matmul(y, Kinv)
print("\nwts = ")
# print(wts)                      Please Uncomment and print the value


# 5. use trained model to make predictions
print("\nActual y values: ")
print(y)

print("\nPredicted y values: ")
for i in range(N):
  x = X[i]
  y_pred = compute_output(wts, x, X, gamma=gamma)
  print("%0.4f" % y_pred)


### `Predicting on unseen X`


y_pd = []
print("\nPredicting for y ")
for i in range(99):
  x = X_test[i]
  print(x.shape)
  print(X.shape)
  y_pred = compute_output(wts, x, X, gamma=gamma)

  y_pd.append(y_pred)
  print("\nPredicted y = %0.4f Actual y = %0.4f " % (y_pred, y_test[i]))
  print("---------")




plt.scatter(y_test, np.array(y_pd), label='Predictions vs Actual')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values (Kernel Regression)")
plt.legend()
plt.show()


# Calculate the test error
test_error = mean_squared_error(y_test, y_pd)
print("Test Error:", test_error)

plt.plot(y_test-y_pd, label='Error')


plt.xlabel('Data Point')
plt.ylabel('Error')
plt.title('Error over Data Points (Kernel Regression)')
plt.legend()
plt.show()
