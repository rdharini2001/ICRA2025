# ICRA2025
Codebase for ICRA 2025


        \item \texttt{load\_point\_cloud}: Reads the point cloud from a simple text file with three columns.
        \item \texttt{load\_splat\_file}: Reads the Gaussian parameters from a CSV file. Each Gaussian is represented by its mean (\( \mu_j \)), a symmetric covariance matrix \( \Sigma_j \) (constructed from the six unique entries), and an associated weight \( w_j \).
    
   PSSI Computation
    \begin{itemize}
        \item Computes centroids and average distances (as proxies for structural characteristics) for both maps.
        \item Uses a nearest-neighbor search (via \texttt{cKDTree}) to pair each point with its closest Gaussian.
        \item Computes the final PSSI value using the provided equation.
    \end{itemize}
    
    \item \textbf{GDD Computation:}
    \begin{itemize}
        \item Uses a kernel density estimate (KDE) to approximate the density at each point in the point cloud.
        \item Computes the density at each point from the Gaussian mixture.
        \item Computes the divergence as the average log-ratio between the two densities.
    \end{itemize}
    
    \item \textbf{PSCI Computation:}
    \begin{itemize}
        \item For each point, finds the maximum response from all Gaussian components.
        \item A higher average value indicates better surface coverage by the Gaussian splat map.
    \end{itemize}
    
    \item \textbf{SUE Computation:}
    \begin{itemize}
        \item For each Gaussian, calculates the entropy of a multivariate Gaussian distribution.
        \item Computes a weighted sum (using the Gaussian weights) to give an overall uncertainty measure.
    \end{itemize}
