"""
Generate train.txt and test.txt for the nondimensionalized pendulum.

Equations:
    dp/dt = -sin(theta)
    dtheta/dt = p

Initial condition: [p, theta] = [0, 1]
Stepsize h = 0.1, n = 140 points.

Stormer-Verlet (a 2nd-order symplectic integrator) is the canonical
symplectic integrator for separable Hamiltonians H = T(p) + V(q).
For H = p^2/2 + (1 - cos theta), one Stormer-Verlet step is:
    p_{n+1/2} = p_n     - (h/2) * sin(theta_n)
    theta_{n+1} = theta_n + h * p_{n+1/2}
    p_{n+1}   = p_{n+1/2} - (h/2) * sin(theta_{n+1})
"""
import numpy as np

def stormer_verlet_step(p, theta, h):
    p_half = p - 0.5 * h * np.sin(theta)
    theta_new = theta + h * p_half
    p_new = p_half - 0.5 * h * np.sin(theta_new)
    return p_new, theta_new

def generate_trajectory(p0, theta0, h, n):
    """Return arrays p[n], theta[n] with x[0] = (p0, theta0)."""
    p = np.zeros(n)
    th = np.zeros(n)
    p[0], th[0] = p0, theta0
    for i in range(n - 1):
        p[i + 1], th[i + 1] = stormer_verlet_step(p[i], th[i], h)
    return p, th

def main():
    np.random.seed(0)
    # As in the project description
    h = 0.1
    n_total = 140
    p, theta = generate_trajectory(p0=0.0, theta0=1.0, h=h, n=n_total)

    # Stack as [p, theta] columns (column 1 = p, column 2 = theta)
    data = np.column_stack([p, theta])

    # As specified, 40 for training and the remaining 100 for testing.
    train_data = data[:40]
    test_data = data[40:]

    np.savetxt("../data/train.txt", train_data, fmt="%.8e")
    np.savetxt("../data/test.txt", test_data, fmt="%.8e")
    np.savetxt("../data/full_trajectory.txt", data, fmt="%.8e")
    print(f"Wrote train.txt ({train_data.shape}) and test.txt ({test_data.shape})")
    print(f"p range: [{p.min():.4f}, {p.max():.4f}]")
    print(f"theta range: [{theta.min():.4f}, {theta.max():.4f}]")

if __name__ == "__main__":
    import os
    os.makedirs("../data", exist_ok=True)
    main()
