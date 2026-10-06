# Optimization Methods Comparison

| Method | Curvature used | Memory | Where it fits |
| :--- | :--- | :--- | :--- |
| **Gradient descent** | None | $O(n)$ | Large DNNs |
| **Newton** | Exact Hessian | $O(n^2)$ | Small models, teaching |
| **Gauss–Newton / LM** | $J^T J$ approximation | $O(n^2)$ or less | Least-squares and curve fitting |
| **L-BFGS** | Implicit, from $m$ past gradients | $O(mn)$ | Full-batch, medium problems |
| **Hessian-free** | Implicit, via $Hv$ products | $O(n)$ | Deep research models |