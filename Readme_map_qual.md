# ICRA2025
Codebase for ICRA 2025

The load_point_cloud function reads the point cloud from a simple text file with three columns.
The load splat file function reads the Gaussian parameters from a CSV file. Each Gaussian is represented by its mean, a symmetric covariance matrix (constructed from the six unique entries), and an associated weight.
    
1. PSSI Computation
Computes centroids and average distances (as proxies for structural characteristics) for both maps.
Uses a nearest-neighbor search (via \texttt{cKDTree}) to pair each point with its closest Gaussian.
Computes the final PSSI value using the provided equation.
       
2. GDD Computation
Uses a kernel density estimate (KDE) to approximate the density at each point in the point cloud.
Computes the density at each point from the Gaussian mixture.
Computes the divergence as the average log-ratio between the two densities.

3. PSCI Computation
For each point, finds the maximum response from all Gaussian components.
A higher average value indicates better surface coverage by the Gaussian splat map.
    
4. SUE Computation
For each Gaussian, calculates the entropy of a multivariate Gaussian distribution.
Computes a weighted sum (using the Gaussian weights) to give an overall uncertainty measure.
