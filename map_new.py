import numpy as np
from numpy.linalg import det, inv
import scipy.spatial.distance as dist
from scipy.spatial import cKDTree

def load_point_cloud(filename):
    return np.loadtxt(filename)

def load_splat_file(filename):
    data = np.loadtxt(filename, delimiter=',')
    means = data[:, :3]
    covs = np.array([
        [[row[3], row[4], row[5]],
         [row[4], row[6], row[7]],
         [row[5], row[7], row[8]]]
        for row in data
    ])
    weights = data[:, 9]
    weights /= np.sum(weights)
    return means, covs, weights

def compute_pssi(points, gaussian_means, gaussian_weights):
    c_P = np.mean(points, axis=0)
    c_G = np.average(gaussian_means, axis=0, weights=gaussian_weights)

    d_P = np.linalg.norm(points - c_P, axis=1)
    d_G = np.linalg.norm(gaussian_means - c_G, axis=1)

    mu_P, mu_G = np.mean(d_P), np.average(d_G, weights=gaussian_weights)
    sigma_P, sigma_G = np.std(d_P), np.sqrt(np.average((d_G - mu_G)**2, weights=gaussian_weights))

    indices = cKDTree(gaussian_means).query(points)[1]
    sigma_PG = np.mean((d_P - mu_P) * (d_G[indices] - mu_G))

    c1, c2 = 1e-6, 1e-6
    return ((2 * mu_P * mu_G + c1) / (mu_P**2 + mu_G**2 + c1)) * ((2 * sigma_PG + c2) / (sigma_P**2 + sigma_G**2 + c2))

def compute_gdd(points, gaussian_means, covs, gaussian_weights, h=0.1):
    N = points.shape[0]
    kde_vals = np.mean(np.exp(-dist.squareform(dist.pdist(points))**2 / (2 * h**2)), axis=1)
    kde_vals *= 1 / ((2 * np.pi * h**2)**1.5)

    rho_G = np.zeros(N)
    for mean_j, cov_j, w_j in zip(gaussian_means, covs, gaussian_weights):
        inv_cov = inv(cov_j)
        norm_const = w_j / ((2 * np.pi)**1.5 * np.sqrt(det(cov_j)))
        diff = points - mean_j
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        rho_G += norm_const * np.exp(exponent)

    rho_G[rho_G == 0] = 1e-12
    return np.mean(np.log(kde_vals / rho_G))

def compute_psci(points, gaussian_means, covs):
    psci_vals = np.zeros(points.shape[0])
    for j in range(len(gaussian_means)):
        mean_j = gaussian_means[j]
        inv_cov = inv(covs[j])
        diff = points - mean_j
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        psci_vals = np.maximum(psci_vals, np.exp(exponent))
    return np.mean(psci_vals)

def compute_sue(covs, gaussian_weights):
    entropies = [0.5 * (3 * np.log(2 * np.pi * np.e) + np.log(det(cov))) for cov in covs]
    return np.sum(gaussian_weights * entropies)

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
