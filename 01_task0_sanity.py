"""
Task 0: Physics sanity checks.

For the nondimensional pendulum (m = l = g = 1):
    dp/dt     = -sin(theta)
    dtheta/dt = p

The Hamiltonian (kinetic + potential, with V chosen so V(0) = 0) is
    H(theta, p) = p^2 / 2  +  (1 - cos theta)

Hamilton's equations:
    dtheta/dt =  dH/dp     =  p
    dp/dt     = -dH/dtheta = -sin theta

So with z = (theta, p)^T and J = [[0, 1], [-1, 0]] (standard symplectic
matrix), dz/dt = J grad H.

Conserved/qualitative properties:
  (i)  Energy: H is conserved exactly by the continuous flow.
  (ii) Periodicity / closed orbits: for E < 2, trajectories are closed
       loops in phase space. Our IC has E = 1 - cos(1) approx 0.4597 < 2.
  (iii) Phase volume (Liouville): the flow is area-preserving in (theta, p).
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("../figures", exist_ok=True)

# Make matplotlib aesthetics consistent across all figures
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
})

def H(theta, p):
    return 0.5 * p**2 + (1.0 - np.cos(theta))

def main():
    full = np.loadtxt("../data/full_trajectory.txt")
    p_arr, theta_arr = full[:, 0], full[:, 1]
    h = 0.1
    t = np.arange(len(p_arr)) * h

    energy = H(theta_arr, p_arr)
    E0 = energy[0]
    drift = energy - E0

    # ----- Energy vs time -----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(t, energy, color="#1C7293")
    ax[0].axhline(E0, color="gray", linestyle="--", alpha=0.7,
                  label=f"E(0) = {E0:.5f}")
    ax[0].set_xlabel("time t")
    ax[0].set_ylabel(r"$H(\theta, p_\theta)$")
    ax[0].set_title("Discrete energy of the data")
    ax[0].legend(frameon=False)

    ax[1].plot(t, drift, color="#B85042")
    ax[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel(r"$H_i - H_0$")
    ax[1].set_title("Energy drift (Stormer-Verlet bounded oscillation)")

    plt.tight_layout()
    plt.savefig("../figures/task0_energy.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ----- Phase portrait of the data -----
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(theta_arr, p_arr, "-", color="#065A82", lw=1.5, alpha=0.8)
    ax.plot(theta_arr, p_arr, "o", color="#065A82", ms=3)
    ax.plot(theta_arr[0], p_arr[0], "s", color="#B85042",
            ms=10, label="initial $(\\theta_0, p_0)$")
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"$p_\theta$")
    ax.set_title("Phase portrait of the data (140 points, h = 0.1)")
    ax.legend(frameon=False)
    ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("../figures/task0_phase_portrait.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"E(0) = {E0:.6f}")
    print(f"max |H_i - H_0| = {np.abs(drift).max():.3e}")
    print(f"std(H_i) = {energy.std():.3e}")
    print(f"relative drift = {np.abs(drift).max() / E0:.3e}")

if __name__ == "__main__":
    main()
