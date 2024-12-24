
class PCA(object):
    def __init__(self, n_components=None):
        """
        Initialize the PCA object.

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit the PCA model to the dataset X using SVD.

        Parameters:
        X (numpy.ndarray): The data matrix with shape (n_samples, n_features).
        """
        # Step 1: Standardize the data (mean centering)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Perform Singular Value Decomposition
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Step 3: Select the top n_components from Vt (principal components)
        if self.n_components is None or self.n_components > X.shape[1]:
            self.n_components = X.shape[1]

        self.components = Vt[:self.n_components, :]

        # Step 4: Calculate explained variance
        variance_explained = (S ** 2) / (X.shape[0] - 1)
        total_variance = np.sum(variance_explained)
        self.explained_variance = variance_explained[:self.n_components] / total_variance

    def transform(self, X):
        """
        Project the data onto the top n_components principal components.

        Parameters:
        X (numpy.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Transformed data with shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components.T)
        return X_transformed

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the data back to its original space.

        Parameters:
        X_transformed (numpy.ndarray): The transformed data with shape (n_samples, n_components).

        Returns:
        numpy.ndarray: Reconstructed data with shape (n_samples, n_features).
        """
        X_reconstructed = np.dot(X_transformed, self.components)
        X_reconstructed += self.mean
        return X_reconstructed

from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

# Load the MNIST dataset
mnist = load_dataset("mnist")

# Create a dataset with 100 images from each class
dataset = []
for i in range(10):
    class_data = mnist['train'].filter(lambda example: example['label'] == i).select(range(100))
    dataset.extend(class_data)

# Extract the images and labels
images = np.array([np.array(example['image']).reshape(28 * 28) for example in dataset])
labels = np.array([example['label'] for example in dataset])



# Apply PCA
n_components = 10
pca = PCA(n_components=n_components)
pca.fit(images)

# Visualize the top 10 principal components as images
fig, axes = plt.subplots(2, 5, figsize=(15, 6))  # 2 rows, 5 columns for 10 components
axes = axes.ravel()  # Flatten the 2D array of axes for easier indexing

# Loop through the top 10 components
for i in range(n_components):
    component = pca.components[i].reshape(28, 28)  # Reshape to 28x28 image
    axes[i].imshow(component, cmap='gray')
    axes[i].set_title(f'Principal Component: {i + 1}')
    axes[i].axis('off')  # Hide axes for cleaner visualization

plt.tight_layout()  # Adjust layout to avoid overlap
plt.show()

# Calculate and print the explained variance ratio
explained_variance_ratio = pca.explained_variance
for i, ratio in enumerate(explained_variance_ratio):
    print(f"Principal Component {i + 1} explains {ratio:.4f} of the variance")


# Load the MNIST dataset
mnist = load_dataset("mnist")

# Create a dataset with 100 images from each class
dataset = []
for i in range(10):
    class_data = mnist['train'].filter(lambda example: example['label'] == i).select(range(100))
    dataset.extend(class_data)

# Extract the images and labels
images = np.array([np.array(example['image']).reshape(28 * 28) for example in dataset])
labels = np.array([example['label'] for example in dataset])


class PCA:
    def __init__(self, n_components=None):
        """
        Initialize the PCA object.

        Parameters:
        n_components (int): Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X):
        """
        Fit the PCA model to the dataset X using SVD.

        Parameters:
        X (numpy.ndarray): The data matrix with shape (n_samples, n_features).
        """
        # Step 1: Standardize the data (mean centering)
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Step 2: Perform Singular Value Decomposition
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Step 3: Select the top n_components from Vt (principal components)
        if self.n_components is None or self.n_components > X.shape[1]:
            self.n_components = X.shape[1]

        self.components = Vt[:self.n_components, :]

    def transform(self, X):
        """
        Project the data onto the top n_components principal components.

        Parameters:
        X (numpy.ndarray): The data matrix with shape (n_samples, n_features).

        Returns:
        numpy.ndarray: Transformed data with shape (n_samples, n_components).
        """
        X_centered = X - self.mean
        X_transformed = np.dot(X_centered, self.components.T)
        return X_transformed

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the data back to its original space.

        Parameters:
        X_transformed (numpy.ndarray): The transformed data with shape (n_samples, n_components).

        Returns:
        numpy.ndarray: Reconstructed data with shape (n_samples, n_features).
        """
        X_reconstructed = np.dot(X_transformed, self.components)
        X_reconstructed += self.mean
        return X_reconstructed


# Reconstruct the dataset using different dimensions and plot original vs reconstructed images
for n_dims in [2, 10, 100, 200]:
    print(f"Reducing to {n_dims} dimensions...")

    # Apply PCA
    pca_reduced = PCA(n_components=n_dims)
    pca_reduced.fit(images)
    reduced_images = pca_reduced.transform(images)
    reconstructed_images = pca_reduced.inverse_transform(reduced_images)

    # Visualize the original vs reconstructed images
    plt.figure(figsize=(10, 4))
    for i in range(2):  # Show two examples for each dimension
        # Original Image
        plt.subplot(2, 2, i * 2 + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Original - Label: {labels[i]}')
        plt.axis('off')

        # Reconstructed Image
        plt.subplot(2, 2, i * 2 + 2)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.title(f'Reconstructed (dim={n_dims})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

import pandas as pd
df = pd.read_csv('cm_dataset_2.csv', header=None)



plt.figure(figsize=(8, 6))
plt.scatter(df[0], df[1])
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('Scatter Plot of Dataset')
plt.show()





class KMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        """
        Initialize KMeans object.

        Parameters:
        n_clusters (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        Fit the KMeans algorithm on the dataset X.

        Parameters:
        X (numpy.ndarray): The data matrix with shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize centroids randomly from the data points
        # np.random.seed(42)  # For reproducibility
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Step 2: Assign each point to the nearest centroid
            labels = self._assign_clusters(X)

            # Step 3: Compute new centroids as the mean of the assigned points
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Check for convergence (if centroids do not change significantly)
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        Assign each sample to the nearest centroid.

        Parameters:
        X (numpy.ndarray): The data matrix.

        Returns:
        numpy.ndarray: Array of cluster labels.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids

    def predict(self, X):
        """
        Predict the cluster for each sample in X.

        Parameters:
        X (numpy.ndarray): The data matrix.

        Returns:
        numpy.ndarray: Array of predicted cluster labels.
        """
        return self._assign_clusters(X)


# Convert DataFrame to NumPy array
data = df.values

# Step 2: Run K-means algorithm
kmeans = KMeans(n_clusters=2)  # Set number of clusters (e.g., 3)
kmeans.fit(data)
labels = kmeans.predict(data)

# Step 3: Visualize the clustered data
plt.figure(figsize=(8, 6))
for i in range(kmeans.n_clusters):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

# Plot centroids
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()

# def plot_kmeans_with_different_initializations(data, n_clusters, num_trials=5):
#     """
#     Runs KMeans with different random initializations and plots the error and clusters.
#     """
#     for trial in range(num_trials):
#         kmeans = KMeans(n_clusters=n_clusters)
#         kmeans.fit(data)
#         labels = kmeans.predict(data)

#         # Plot the clusters
#         plt.figure(figsize=(8, 6))
#         for i in range(kmeans.n_clusters):
#             cluster_points = data[labels == i]
#             plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')

#         # Plot centroids
#         plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], s=300, c='red', marker='X', label='Centroids')
#         plt.xlabel('Column 1')
#         plt.ylabel('Column 2')
#         plt.title(f'K-means Clustering (Trial {trial + 1})')
#         plt.legend()
#         plt.show()


# # Convert DataFrame to NumPy array
# data = df.values

# # Run KMeans with different initializations and plot
# plot_kmeans_with_different_initializations(data, n_clusters=2)

# def plot_eroor_kmeans_with_different_initializations(data, n_clusters, num_trials=5):
#     """
#     Runs KMeans with different random initializations and plots the error and clusters.
#     """
#     for trial in range(num_trials):
#         kmeans = KMeans(n_clusters=n_clusters, n_init=1, random_state=trial, max_iter=300)
        
#         # Fit KMeans
#         kmeans.fit(data)
        
#         # Collect errors (inertia) during the iterations
#         errors = []  # Store errors for each iteration
#         for iteration in range(kmeans.n_iter_):
#             # Get the labels of each data point
#             labels = kmeans.predict(data)
#             # Compute the sum of squared distances to the cluster centers (inertia)
#             distances = np.linalg.norm(data - kmeans.cluster_centers_[labels], axis=1)
#             error = np.sum(distances ** 2)  # Sum of squared distances for error
#             errors.append(error)
        
#         # Plot the error function w.r.t iterations for this trial
#         plt.figure(figsize=(8, 6))
#         plt.plot(range(1, len(errors) + 1), errors, marker='o')
#         plt.xlabel('Iterations')
#         plt.ylabel('Error (Sum of Squared Distances)')
#         plt.title(f'K-means Error Function (Trial {trial + 1})')
#         plt.grid(True)
#         plt.show()

# # Example dataset for testing (You can replace it with your own data)
# data, _ = make_blobs(n_samples=300, centers=3)

# # Run KMeans with different initializations and plot the error
# plot_kmeans_with_different_initializations(data, n_clusters=2)

import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        """
        Initialize KMeans object.

        Parameters:
        n_clusters (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        """
        Fit the KMeans algorithm on the dataset X.

        Parameters:
        X (numpy.ndarray): The data matrix with shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape

        # Step 1: Initialize centroids randomly from the data points
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centroids = X[random_indices]

        for i in range(self.max_iters):
            # Step 2: Assign each point to the nearest centroid
            labels = self._assign_clusters(X)

            # Step 3: Compute new centroids as the mean of the assigned points
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.n_clusters)])

            # Check for convergence (if centroids do not change significantly)
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X):
        """
        Assign each sample to the nearest centroid.

        Parameters:
        X (numpy.ndarray): The data matrix.

        Returns:
        numpy.ndarray: Array of cluster labels.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        """
        Predict the cluster for each sample in X.

        Parameters:
        X (numpy.ndarray): The data matrix.

        Returns:
        numpy.ndarray: Array of predicted cluster labels.
        """
        return self._assign_clusters(X)


def plot_kmeans_with_different_initializations(data, n_clusters, num_trials=5):
    """
    Runs KMeans with different random initializations and plots the error and clusters side by side.
    """
    for trial in range(num_trials):
        kmeans = KMeans(n_clusters=n_clusters, max_iters=300)
        
        # Fit KMeans
        kmeans.fit(data)
        
        # Collect errors (inertia) during the iterations
        errors = []  # Store errors for each iteration
        for iteration in range(kmeans.max_iters):
            # Get the labels of each data point
            labels = kmeans.predict(data)
            # Compute the sum of squared distances to the cluster centers (inertia)
            distances = np.linalg.norm(data - kmeans.centroids[labels], axis=1)
            error = np.sum(distances ** 2)  # Sum of squared distances for error
            errors.append(error)
            
            # Stop if convergence is achieved earlier (based on tolerance)
            if np.linalg.norm(data - kmeans.centroids[labels]) < 1e-4:
                break

        # Create subplots for error function and clusters
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 1 row, 2 columns
        
        # Plot the error function (Inertia) w.r.t iterations on the first subplot
        axes[0].plot(range(1, len(errors) + 1), errors, marker='o')
        axes[0].set_xlabel('Iterations')
        axes[0].set_ylabel('Error (Sum of Squared Distances)')
        axes[0].set_title(f'K-means Error Function (Trial {trial + 1})')
        axes[0].grid(True)
        
        # Plot the clusters and centroids on the second subplot
        labels = kmeans.predict(data)
        for i in range(kmeans.n_clusters):
            cluster_points = data[labels == i]
            axes[1].scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i + 1}')
        
        axes[1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                        s=300, c='red', marker='X', label='Centroids')
        axes[1].set_xlabel('Column 1')
        axes[1].set_ylabel('Column 2')
        axes[1].set_title(f'K-means Clustering (Trial {trial + 1})')
        axes[1].legend()
        
        # Show the side-by-side plots
        plt.tight_layout()
        plt.show()


# Example dataset for testing (You can replace it with your own data)
data, _ = make_blobs(n_samples=300, centers=3)

# Run KMeans with different initializations and plot the error and clusters side by side
plot_kmeans_with_different_initializations(df.values, n_clusters=2)


from scipy.spatial import Voronoi, voronoi_plot_2d

# KMeans from scratch
class KMeansScratch:
    def __init__(self, n_clusters=3, max_iters=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)
        # Randomly initialize centroids
        random_idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centroids = X[random_idx]

        for _ in range(self.max_iters):
            # Step 1: Assign labels to each point based on closest centroid
            labels = self._assign_labels(X)

            # Step 2: Compute new centroids as the mean of the points assigned to each cluster
            new_centroids = self._compute_centroids(X, labels)

            # Step 3: Check for convergence (if centroids do not change)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
            self.centroids = new_centroids

        self.labels_ = labels
        return self

    def _assign_labels(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return centroids


# Load the dataset
df = pd.read_csv('cm_dataset_2.csv', header=None)

# Convert DataFrame to numpy array for KMeans
X = df.values

for i in [2, 3, 4, 5]:
    # Apply KMeans from scratch
    kmeans = KMeansScratch(n_clusters=i)
    kmeans.fit(X)
    km = kmeans.labels_
    centers = kmeans.centroids

    # Plotting the Voronoi diagram and K-means clusters
    plt.figure(figsize=(15, 10))

    # Only plot Voronoi diagram if we have more than 2 centroids
    if i > 2:
        # Create Voronoi diagram using cluster centers
        vor = Voronoi(centers)
        voronoi_plot_2d(vor, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5)

    # Plot the data points, colored by their cluster label
    plt.scatter(X[:, 0], X[:, 1], c=km, s=100, cmap='viridis', marker='o', label='Data Points')

    # Plot the cluster centers with a star marker
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=300, alpha=1, marker='x', label='Centroids')

    # Add labels, titles, and hide the box around the plot

    plt.title(f'K-means Clustering with Voronoi Diagram (K={i})')
    plt.xlabel('Column 1')
    plt.ylabel('Column 2')
    plt.legend()
    plt.box(False)
    plt.show()

