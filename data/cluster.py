import numpy as np
import matplotlib.pyplot as plt

# Convert similarity matrix to data pointsa
def similarity_to_points(similarity_matrix):
    """
    Convert a similarity matrix to 2D points using a simple method:
    Treat the rows as features and reduce dimensions via PCA-like logic.
    """
    # Standardize the matrix (mean = 0)
    standardized_matrix = similarity_matrix - np.mean(similarity_matrix, axis=0)
    # Calculate covariance matrix
    covariance_matrix = np.cov(standardized_matrix)
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # Take top 2 eigenvectors
    top_indices = np.argsort(eigenvalues)[-2:][::-1]
    top_components = eigenvectors[:, top_indices]
    # Transform data to 2D
    points_2d = np.dot(standardized_matrix, top_components)
    return points_2d

# K-means clustering
def kmeans(data, k, max_iter=100):
    """
    Args:
    - data: Array of shape (n_samples, n_features).
    - k: Number of clusters.
    - max_iter: Maximum number of iterations.
    
    Returns:
    - clusters: Cluster assignments for each point.
    - centers: Final cluster centers.
    """
    np.random.seed(42)
    n_samples = data.shape[0]
    
    # Randomly initialize cluster centers
    centers = data[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iter):
        # Assign each point to the nearest center
        clusters = []
        for point in data:
            distances = np.linalg.norm(point - centers, axis=1) # Calculate the distance from data point to the center
            clusters.append(np.argmin(distances)) # Assign the point to the nearest center
        clusters = np.array(clusters)
        
        # Update cluster centers
        new_centers = np.array([data[clusters == i].mean(axis=0) for i in range(k)])
        
        # Check for convergence
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    
    return clusters, centers

# Visualization
def visualize_clustering(data, labels, centers, n_clusters):
    plt.figure(figsize=(10, 8))
    
    for i in range(n_clusters):
        cluster_points = data[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], s=100, alpha=0.7, label=f'Cluster {i+1}')
    
    # Plot cluster centers
    plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X', label='Centers')
    
    plt.title('K-means Clustering')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

input_matrix = np.random.rand(16, 16)

# Convert similarity matrix to 2D points
points = similarity_to_points(input_matrix)

# K-means clustering
n_clusters = 2
labels, centers = kmeans(points, n_clusters)

# Visualize clustering
visualize_clustering(points, labels, centers, n_clusters)
