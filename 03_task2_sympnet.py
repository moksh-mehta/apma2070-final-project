"""
Task 2: SympNet.

A SympNet (Jin et al., 2020) factorises the learned map into symplectic
sublayers so the resulting map is symplectic-by-construction. For a 2D
phase space z = (p, q)^T with the symplectic form omega = dq ^ dp
(equivalently the matrix J = [[0, -1], [1, 0]]) and Hamiltonian flow,
a map phi: R^2 -> R^2 is symplectic iff its Jacobian D phi satisfies
    D phi(z)^T J D phi(z) = J,
which in 2D reduces to det(D phi) = 1 (area preservation).

LA-SympNet uses the elementary symplectic shears
    P_a:  p -> p + grad V(q), q -> q
    Q_a:  p -> p,             q -> q + grad K(p)
where V, K are scalar functions (here parametrised by small MLPs).
Composing alternating P_a and Q_a layers yields a symplectic map that is
universal for symplectic diffeomorphisms.

We implement the gradient-module variant: V(q) and K(p) are scalar MLPs
of the form sum_i a_i * sigma(k_i * x + b_i), and their derivatives wrt
the input are obtained analytically (or via autograd).
"""
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
})

DEVICE = "cpu"
os.makedirs("../figures", exist_ok=True)
os.makedirs("../results", exist_ok=True)


# ------------- SympNet building blocks -------------
class GradientModule(nn.Module):
    """Scalar function of a 1D input parameterised so its gradient is easy.

    f(x) = sum_i a_i * sigma(k_i x + b_i)
    f'(x) = sum_i a_i * k_i * sigma'(k_i x + b_i)

    sigma = tanh, so sigma' = 1 - tanh^2.
    """
    def __init__(self, width=32):
        super().__init__()
        self.a = nn.Parameter(0.1 * torch.randn(width))
        self.k = nn.Parameter(0.1 * torch.randn(width))
        self.b = nn.Parameter(0.1 * torch.randn(width))

    def derivative(self, x):
        # x shape: (N, 1) -> output shape (N, 1)
        z = self.k * x + self.b                # (N, width)
        sig_prime = 1.0 - torch.tanh(z) ** 2   # (N, width)
        return (self.a * self.k * sig_prime).sum(dim=-1, keepdim=True)


class UpShear(nn.Module):
    """(p, q) -> (p + V'(q), q).  Symplectic (det Jacobian = 1)."""
    def __init__(self, width=32):
        super().__init__()
        self.V = GradientModule(width)

    def forward(self, x):
        p, q = x[..., 0:1], x[..., 1:2]
        return torch.cat([p + self.V.derivative(q), q], dim=-1)


class LowShear(nn.Module):
    """(p, q) -> (p, q + K'(p)).  Symplectic."""
    def __init__(self, width=32):
        super().__init__()
        self.K = GradientModule(width)

    def forward(self, x):
        p, q = x[..., 0:1], x[..., 1:2]
        return torch.cat([p, q + self.K.derivative(p)], dim=-1)


class SympNet(nn.Module):
    """Alternating LowShear/UpShear stack — guaranteed symplectic map."""
    def __init__(self, n_layers=8, width=32):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(LowShear(width) if i % 2 == 0 else UpShear(width))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ------------- training -------------
def main():
    full = np.loadtxt("../data/full_trajectory.txt")
    H_STEP = 0.1
    X_train = full[:40]
    Y_train = full[1:41]
    X_test = full[40:-1]
    Y_test = full[41:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    model = SympNet(n_layers=8, width=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
    EPOCHS = 20000
    for ep in range(EPOCHS):
        opt.zero_grad()
        pred = model(X_train_t)
        loss = ((pred - Y_train_t) ** 2).mean()
        loss.backward()
        opt.step()
        scheduler.step()

    # ---------- evaluation ----------
    with torch.no_grad():
        Y_pred_tf = model(X_test_t).numpy()
    tf_err = ((Y_pred_tf - Y_test) ** 2).mean(axis=1)
    tf_mse = tf_err.mean()

    # rollout
    pred_traj = np.zeros((100, 2))
    pred_traj[0] = full[40]
    x = torch.tensor(full[40], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for k in range(99):
            x = model(x)
            pred_traj[k + 1] = x.squeeze(0).numpy()

    true_traj = full[40:]
    rollout_err = ((pred_traj - true_traj) ** 2).mean(axis=1)
    rollout_mse = rollout_err.mean()

    print(f"SympNet teacher-forcing MSE: {tf_mse:.3e}")
    print(f"SympNet rollout MSE over 99 steps: {rollout_mse:.3e}")
    print(f"SympNet rollout MSE at final step: {rollout_err[-1]:.3e}")

    # save
    np.savetxt("../results/sympnet_rollout.txt", pred_traj)
    np.savetxt("../results/sympnet_tf_err.txt", tf_err)
    np.savetxt("../results/sympnet_rollout_err.txt", rollout_err)
    torch.save(model.state_dict(), "../results/sympnet_model.pt")

    # ---------- symplecticity test (det Jacobian) ----------
    # In 2D, symplectic <=> area-preserving <=> det(J) = 1.
    # We sample 200 random phase-space points and check det(D phi).
    rng = np.random.default_rng(0)
    samples = rng.uniform(low=[-1.0, -1.2], high=[1.0, 1.2], size=(200, 2))
    dets = []
    for s in samples:
        x = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        # compute Jacobian via autograd
        y = model(x)
        J = torch.zeros(2, 2)
        for i in range(2):
            grad = torch.autograd.grad(y[i], x, retain_graph=(i == 0))[0]
            J[i] = grad
        dets.append(np.linalg.det(J.numpy()))
    dets = np.array(dets)
    print(f"SympNet det(Jacobian): mean = {dets.mean():.6f}, "
          f"max |det - 1| = {np.abs(dets - 1).max():.2e}")

    # ---------- plots ----------
    t_test = np.arange(40, 140) * H_STEP

    # FNN rollout for comparison
    fnn_traj = np.loadtxt("../results/fnn_rollout.txt")
    fnn_rollout_err = np.loadtxt("../results/fnn_rollout_err.txt")
    fnn_tf_err = np.loadtxt("../results/fnn_tf_err.txt")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(full[:, 1], full[:, 0], "-", color="#065A82",
               alpha=0.4, label="true full")
    ax[0].plot(true_traj[:, 1], true_traj[:, 0], "o", color="#065A82",
               ms=3, label="true (test)")
    ax[0].plot(fnn_traj[:, 1], fnn_traj[:, 0], "x", color="#B85042",
               ms=4, alpha=0.8, label="FNN rollout")
    ax[0].plot(pred_traj[:, 1], pred_traj[:, 0], "+", color="#2C5F2D",
               ms=6, alpha=0.9, label="SympNet rollout")
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_ylabel(r"$p_\theta$")
    ax[0].set_title("Phase portrait: FNN vs SympNet rollout")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_aspect("equal")

    ax[1].semilogy(t_test[1:], fnn_rollout_err[1:], "x-", color="#B85042",
                   ms=4, label="FNN rollout")
    ax[1].semilogy(t_test[1:], rollout_err[1:], "+-", color="#2C5F2D",
                   ms=5, label="SympNet rollout")
    ax[1].semilogy(t_test[1:], fnn_tf_err, ":", color="#B85042",
                   alpha=0.5, label="FNN teacher-forcing")
    ax[1].semilogy(t_test[1:], tf_err, ":", color="#2C5F2D",
                   alpha=0.5, label="SympNet teacher-forcing")
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel("per-step MSE")
    ax[1].set_title("Rollout MSE vs time")
    ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig("../figures/task2_sympnet.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---------- energy comparison ----------
    def H(theta, p): return 0.5 * p**2 + (1.0 - np.cos(theta))
    E_true = H(true_traj[:, 1], true_traj[:, 0])
    E_fnn  = H(fnn_traj[:, 1], fnn_traj[:, 0])
    E_sym  = H(pred_traj[:, 1], pred_traj[:, 0])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_test, E_true, color="#065A82", label="true")
    ax.plot(t_test, E_fnn,  color="#B85042", label="FNN")
    ax.plot(t_test, E_sym,  color="#2C5F2D", label="SympNet")
    ax.axhline(H(1.0, 0.0), color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$H(\theta, p_\theta)$")
    ax.set_title("Energy along the rollout (lower drift = better structure)")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("../figures/task2_energy.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"FNN energy max drift:     {np.abs(E_fnn - H(1.0, 0.0)).max():.3e}")
    print(f"SympNet energy max drift: {np.abs(E_sym - H(1.0, 0.0)).max():.3e}")

    # ---------- symplecticity histogram ----------
    # Also compute the FNN Jacobian determinants for comparison
    import importlib.util
    spec = importlib.util.spec_from_file_location("fnn_mod", "02_task1_fnn.py")
    fnn_mod = importlib.util.module_from_spec(spec)
    # Build a fresh FNN architecture and load weights
    class _FNN(nn.Module):
        def __init__(self, hidden=64, layers=3):
            super().__init__()
            mods = [nn.Linear(2, hidden), nn.Tanh()]
            for _ in range(layers - 1):
                mods += [nn.Linear(hidden, hidden), nn.Tanh()]
            mods += [nn.Linear(hidden, 2)]
            self.net = nn.Sequential(*mods)
        def forward(self, x): return self.net(x)
    fnn = _FNN(); fnn.load_state_dict(torch.load("../results/fnn_model.pt"))
    fnn_dets = []
    for s in samples:
        x = torch.tensor(s, dtype=torch.float32, requires_grad=True)
        y = fnn(x)
        J = torch.zeros(2, 2)
        for i in range(2):
            grad = torch.autograd.grad(y[i], x, retain_graph=(i == 0))[0]
            J[i] = grad
        fnn_dets.append(np.linalg.det(J.numpy()))
    fnn_dets = np.array(fnn_dets)
    print(f"FNN det(Jacobian):   mean = {fnn_dets.mean():.4f}, "
          f"std = {fnn_dets.std():.4f}, max |det - 1| = {np.abs(fnn_dets - 1).max():.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # FNN dets vs 1
    ax[0].hist(fnn_dets, bins=25, color="#B85042", alpha=0.85)
    ax[0].axvline(1, color="black", linestyle="--", label="symplectic (det = 1)")
    ax[0].set_xlabel(r"$\det(D\phi)$")
    ax[0].set_ylabel("count")
    ax[0].set_title("FNN: Jacobian determinants (200 random states)")
    ax[0].legend(frameon=False)

    # SympNet — show |det - 1| on log scale since they're ~1e-7
    ax[1].hist(np.log10(np.abs(dets - 1) + 1e-16), bins=25,
               color="#2C5F2D", alpha=0.85)
    ax[1].set_xlabel(r"$\log_{10} |\det(D\phi) - 1|$")
    ax[1].set_ylabel("count")
    ax[1].set_title("SympNet: deviation from det = 1 (machine precision)")
    plt.tight_layout()
    plt.savefig("../figures/task2_symplecticity.png", dpi=160, bbox_inches="tight")
    plt.close()
    np.savetxt("../results/sympnet_dets.txt", dets)
    np.savetxt("../results/fnn_dets.txt", fnn_dets)


if __name__ == "__main__":
    main()
