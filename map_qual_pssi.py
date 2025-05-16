#Author: Srijan Dokania
#Description: This script computes the Probabilistic Structural Similarity Index (PSSI) and Geometric Density Divergence (GDD) between a point cloud and a Gaussian splatting map.
# It loads point cloud data from a .ply file and Gaussian splatting data from a .splat file.
# It computes the PSSI and GDD metrics, which are useful for evaluating the quality of Gaussian splatting splat and pointcloud representations.
import numpy as np
from numpy.linalg import det, inv
import scipy.spatial.distance as dist
from scipy.spatial import cKDTree, KDTree
from sklearn.neighbors import KernelDensity
import open3d as o3d
import re

def load_point_cloud(filename):
    """
    Load point cloud data from a .pcl file.
    Assumes the file has three columns: x, y, z.
    """
    pcd = o3d.io.read_point_cloud(filename)  
    return np.asarray(pcd.points)
def load_splat(filename):
    """
    Load Gaussian splatting data from a .splat file.
    Assumes each row is:
    mean_x, mean_y, mean_z, sigma_xx, sigma_xy, sigma_xz, sigma_yy, sigma_yz, sigma_zz, weight
    """
    means = []
    covs = []
    weights = []
    data = []
    with open(filename, 'r') as f:
        for line in f:
            # Remove tensor notation and split components
            components = line.strip().replace('tensor(', '').replace(')', '') \
                                     .replace("device='cuda:0'", '').split(',')
            
            # Extract float values from string components
            cleaned = []
            # print(components)
            for comp in components:
                if comp.strip() != '':
                    comp = comp.strip()
                    if comp.startswith('['):  # Handle weight array syntax
                        comp = comp[1:-1]
                    cleaned.append(float(comp))
            data.append(cleaned)
    data = np.array(data)
    # print(data[0:5])
    for row in data:  # Each row has [X,Y,Z, σ_xx,σ_xy,σ_xz,σ_yy,σ_yz,σ_zz, weight]
            cov_matrix = np.array([
                [row[3], row[4], row[5]],    # σ_xx, σ_xy, σ_xz
                [row[4], row[6], row[7]],    # σ_xy, σ_yy, σ_yz 
                [row[5], row[7], row[8]]     # σ_xz, σ_yz, σ_zz
            ], dtype=np.float32)
            covs.append(cov_matrix)
    print("covs", covs[0:5])
    means = data[:, :3]
    print("means", means[0:5])
    covs = np.array(covs)
    weights = data[:, 9]
    print("weights", weights[0:5])
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

def compute_gdd_old(points, gaussian_means, covs, gaussian_weights, h=0.1):
    """
    Compute the Geometric Density Divergence (GDD) between the point cloud and the Gaussian splatting map.
    """
    N = points.shape[0]
    # Empirical density for the point cloud using KDE
    factor = 1 / ((2 * np.pi * h**2)**(1.5))
    # Compute pairwise distances between points
    dists = dist.squareform(dist.pdist(points))
    kde_vals = np.mean(factor * np.exp(-dists**2 / (2 * h**2)), axis=1)
    kde_vals = np.zeros(N)
    for i in range(N):
        # Query points within 3h bandwidth
        dists = tree.query_ball_point(points[i], r=3*h)
        kde_vals[i] = np.exp(-np.sum((points[dists] - points[i])**2, axis=1)/(2*h**2)).sum()
    kde_vals /= (N * (2*np.pi*h**2)**1.5)
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


if __name__ == '__main__':
    # Load point cloud and Gaussian splatting data
    points = load_point_cloud('rtab_map_cloud.ply')
    gaussian_means, covs, gaussian_weights = load_splat('splat_model.splat')
    
    # Compute metrics
    pssi = compute_pssi(points, gaussian_means, gaussian_weights)
    print("Probabilistic Structural Similarity Index (PSSI):", pssi)

    # gdd = compute_gdd(points, gaussian_means, covs, gaussian_weights, h=0.1)
    # print("Geometric Density Divergence (GDD):", gdd)


