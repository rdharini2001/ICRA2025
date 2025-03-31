import numpy as np
from numpy.linalg import det, inv
import scipy.spatial.distance as dist
from scipy.spatial import cKDTree

def load_point_cloud(filename):
    """
    Load point cloud data from a .pcl file.
    Assumes the file has three columns: x, y, z.
    """
    return np.loadtxt(filename)

def load_splat_file(filename):
    """
    Load Gaussian splatting data from a .splat file.
    Assumes each row is:
    mean_x, mean_y, mean_z, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, weight
    """
    data = np.loadtxt(filename, delimiter=',')
    means = data[:, :3]
    covs = []
    for row in data:
        # Construct the symmetric covariance matrix:
        # [ sigma_xx, sigma_xy, sigma_xz ]
        # [ sigma_xy, sigma_yy, sigma_yz ]
        # [ sigma_xz, sigma_yz, sigma_zz ]
        cov = np.array([[row[3], row[4], row[5]],
                        [row[4], row[6], row[7]],
                        [row[5], row[7], row[8]]])
        covs.append(cov)
    covs = np.array(covs)
    weights = data[:, 9]
    # Normalize weights
    weights = weights / np.sum(weights)
    return means, covs, weights

def compute_pssi(points, gaussian_means, gaussian_weights):
    """
    Compute the Probabilistic Structural Similarity Index (PSSI).
    """
    # Compute centroids
    c_P = np.mean(points, axis=0)
    c_G = np.average(gaussian_means, axis=0, weights=gaussian_weights)
    
    # Compute distances from centroids
    d_P = np.linalg.norm(points - c_P, axis=1)
    d_G = np.linalg.norm(gaussian_means - c_G, axis=1)
    
    mu_P = np.mean(d_P)
    mu_G = np.average(d_G, weights=gaussian_weights)
    
    sigma_P = np.std(d_P)
    sigma_G = np.sqrt(np.average((d_G - mu_G)**2, weights=gaussian_weights))
    
    # For covariance: assign each point the distance of the nearest Gaussian mean
    tree = cKDTree(gaussian_means)
    _, indices = tree.query(points)
    d_G_assigned = d_G[indices]
    sigma_PG = np.mean((d_P - mu_P) * (d_G_assigned - mu_G))
    
    # Stability constants
    c1 = 1e-6
    c2 = 1e-6
    pssi = ((2 * mu_P * mu_G + c1) / (mu_P**2 + mu_G**2 + c1)) * \
           ((2 * sigma_PG + c2) / (sigma_P**2 + sigma_G**2 + c2))
    return pssi

def compute_gdd(points, gaussian_means, covs, gaussian_weights, h=0.1):
    """
    Compute the Geometric Density Divergence (GDD) between the point cloud and the Gaussian splatting map.
    """
    N = points.shape[0]
    # Empirical density for the point cloud using KDE
    factor = 1 / ((2 * np.pi * h**2)**(1.5))
    # Compute pairwise distances between points
    dists = dist.squareform(dist.pdist(points))
    kde_vals = np.mean(factor * np.exp(-dists**2 / (2 * h**2)), axis=1)
    
    # Density from the Gaussian mixture for each point in the point cloud
    rho_G = np.zeros(N)
    for j in range(len(gaussian_means)):
        mean_j = gaussian_means[j]
        cov_j = covs[j]
        inv_cov = inv(cov_j)
        det_cov = det(cov_j)
        norm_const = gaussian_weights[j] / ((2 * np.pi)**1.5 * np.sqrt(det_cov))
        diff = points - mean_j
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        rho_G += norm_const * np.exp(exponent)
    
    # Avoid division by zero
    rho_G[rho_G == 0] = 1e-12
    gdd = np.mean(np.log(kde_vals / rho_G))
    return gdd

def compute_psci(points, gaussian_means, covs):
    """
    Compute the Probabilistic Surface Coverage Index (PSCI).
    """
    N = points.shape[0]
    psci_vals = np.zeros(N)
    for i, p in enumerate(points):
        max_val = 0
        for j in range(len(gaussian_means)):
            mean_j = gaussian_means[j]
            cov_j = covs[j]
            inv_cov = inv(cov_j)
            diff = p - mean_j
            exponent = -0.5 * (diff.T @ inv_cov @ diff)
            val = np.exp(exponent)
            if val > max_val:
                max_val = val
        psci_vals[i] = max_val
    psci = np.mean(psci_vals)
    return psci

def compute_sue(covs, gaussian_weights):
    """
    Compute the Spatial Uncertainty Entropy (SUE) for the Gaussian splatting map.
    """
    sue = 0
    for j in range(len(covs)):
        cov_j = covs[j]
        det_cov = det(cov_j)
        entropy = 0.5 * (3 * np.log(2 * np.pi * np.e) + np.log(det_cov))
        sue += gaussian_weights[j] * entropy
    return sue

if __name__ == '__main__':
    # Load point cloud and Gaussian splatting data
    points = load_point_cloud('map.pcl')
    gaussian_means, covs, gaussian_weights = load_splat_file('map.splat')
    
    # Compute metrics
    pssi = compute_pssi(points, gaussian_means, gaussian_weights)
    gdd = compute_gdd(points, gaussian_means, covs, gaussian_weights, h=0.1)
    psci = compute_psci(points, gaussian_means, covs)
    sue = compute_sue(covs, gaussian_weights)
    
    # Print results
    print("Probabilistic Structural Similarity Index (PSSI):", pssi)
    print("Geometric Density Divergence (GDD):", gdd)
    print("Probabilistic Surface Coverage Index (PSCI):", psci)
    print("Spatial Uncertainty Entropy (SUE):", sue)
