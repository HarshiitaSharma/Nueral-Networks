# Second-Order Optimization Methods

Second-order optimization methods are mathematical optimization algorithms that utilize both the gradient (first derivative) and the Hessian matrix (second derivative) of an objective function to find optimal parameter values. By incorporating the second derivative, these methods capture the curvature of the loss landscape, allowing them to make highly informed, adaptive updates that converge in fewer iterations than standard first-order methods like Gradient Descent or Adam.

## OBSERVE
Second-order methods win on iteration count in every task above, and lose on cost per iteration everywhere
except the smallest problems. That is precisely why SGD, Momentum and Adam still dominate large-scale deep
learning, while L-BFGS and Hessian-free methods survive in scientific machine learning, fine-tuning and full-batch
problems where high accuracy matters more than throughput.

## The Intuition Behind Curvature
While first-order methods only see the direction of the steepest descent, they treat all directions relatively equally, often leading to oscillations in steep ravines or slow progress on flat plateaus.
- Large Curvature (Steep Valleys): Second-order methods take smaller, cautious steps to avoid overshooting.
- Small Curvature (Flat Plateaus): They take larger, aggressive steps to traverse the flat region quickly.
- Saddle Points: They use curvature directions to escape saddle points where first-order gradients drop to zero.

## Running it

```bash
pip install torch matplotlib
jupyter notebook second_order.ipynb
```
# Optimization Methods Comparison

| Method | Curvature used | Memory | Where it fits |
| :--- | :--- | :--- | :--- |
| **Gradient descent** | None | $O(n)$ | Large DNNs |
| **Newton** | Exact Hessian | $O(n^2)$ | Small models, teaching |
| **Gauss–Newton / LM** | $J^T J$ approximation | $O(n^2)$ or less | Least-squares and curve fitting |
| **L-BFGS** | Implicit, from $m$ past gradients | $O(mn)$ | Full-batch, medium problems |
| **Hessian-free** | Implicit, via $Hv$ products | $O(n)$ | Deep research models |



## Lab exercise
1. Apply the Task 1 code to L(w) = 3w² − 12w + 7 starting from w = 4. Report g, H and w₁, and confirm by hand that the method reaches the optimum in one step.
2. Change A in Task 2 to [[20, 0], [0, 0.05]] and rerun. How many gradient-descent iterations are now needed to match one Newton step, and how does that number relate to the condition number of A?
3. Replace L(w) = w² − 4w + 5 in Task 1 with L(w) = w⁴ − 3w² + w and start Newton’s method from w = 0.1.Explain what goes wrong and connect it to the sign of H.
4. Extend the Task 4 code into adaptive Levenberg–Marquardt: multiply λ by 10 when a step increases the SSE and divide it by 10 when the step succeeds. Report the λ trajectory over ten iterations.
5. Fit the Task 5 problem with a small MLP instead of the two-parameter exponential, and compare L-BFGS against Adam on both full-batch and mini-batch data. Which optimizer degrades when the batches change?
6. Modify conjugate_gradient in Task 6 to stop after a fixed number of iterations, and measure how the quality of the resulting direction changes for a 20-dimensional random quadratic.