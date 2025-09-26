# Mathematical foundation
## 1. Objective

Find a new axis 𝑣 (unit vector) so that when projecting data 𝑥𝑖 onto it, the variance of the projected data is maximized.

Reduce dimensionality by keeping axes with the largest variance.

## 2. Data Standardization

Gaussian standardization: 𝑥𝑖 = (𝑥𝑖−mean(𝑥𝑖)) / std(𝑥𝑖).

After this step, mean(𝑥𝑖) = 0.

Meaning: centering data around the origin helps identify the “direction of maximum spread” and balances features with different ranges.

## 3. Projecting Data onto the New Axis

For one point: 𝑧𝑖 = 𝑣^𝑇.𝑥𝑖

For all data: 𝑍 = 𝑋.𝑣

## 4. Variance of Projected Data

Var(z) = (1/n).Σ(zi − z̄)²

After centering (μ=0): Var(z) = (1/n).Σ(zi²) = (1/n).(Xv)^T.(Xv) = v^T.A.v

where A = (1/n).X^T.X (covariance matrix)

## 5. Constrained Optimization

Maximize J(v) = v^T.A.v

Constraint: v^T.v = 1 (unit vector has |v| = 1)

Lagrangian: L(v, λ) = v^T.A.v − λ.(v^T.v − 1)

Derivative to find the extremum to maximize with respect to 𝑣: A.v = λ.v

## 6. Finding Eigenvalues & Eigenvectors

Equation: (A − λI).v = 0

Non-trivial solutions exist only if det(A − λI) = 0

Solve for λ (eigenvalues) and v (eigenvectors), then normalize v

## 7. Interpretation of Results

Eigenvector v = direction of the new axis

Eigenvalue λ = variance along that axis (because Var(z) = v^T.A.v and A.v = λ.v => Var(z) = v^T.λ.v = λ.(v^T.v), but v^T.v = 1 => Var(z) = λ

## 8. Principal Components

Sort eigenvalues in descending order

Choose k eigenvectors with largest λ → Vk

Project data: Z = X.Vk

#
