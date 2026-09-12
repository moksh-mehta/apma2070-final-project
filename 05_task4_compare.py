"""
Task 4: Comparative analysis.

We re-train FNN and SympNet on data with additive Gaussian noise (sigma = 0.02
and sigma = 0.05) and compare:
  - one-step error
  - rollout error at the final step
  - energy drift

The PINN we already have from Task 3 — we re-run a noisy version too,
showing the regularising effect of the physics residual.

We also generate a summary table figure.
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

os.makedirs("../figures", exist_ok=True)
os.makedirs("../results", exist_ok=True)


# ---------- shared utilities ----------
def H(theta, p): return 0.5 * p**2 + (1.0 - np.cos(theta))


# ---------- FNN ----------
class FNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        mods = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)


# ---------- SympNet building blocks (replicated minimal) ----------
class GradientModule(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.a = nn.Parameter(0.1 * torch.randn(width))
        self.k = nn.Parameter(0.1 * torch.randn(width))
        self.b = nn.Parameter(0.1 * torch.randn(width))
    def derivative(self, x):
        z = self.k * x + self.b
        sig_prime = 1.0 - torch.tanh(z) ** 2
        return (self.a * self.k * sig_prime).sum(dim=-1, keepdim=True)


class UpShear(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.V = GradientModule(width)
    def forward(self, x):
        p, q = x[..., 0:1], x[..., 1:2]
        return torch.cat([p + self.V.derivative(q), q], dim=-1)


class LowShear(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.K = GradientModule(width)
    def forward(self, x):
        p, q = x[..., 0:1], x[..., 1:2]
        return torch.cat([p, q + self.K.derivative(p)], dim=-1)


class SympNet(nn.Module):
    def __init__(self, n_layers=8, width=32):
        super().__init__()
        self.layers = nn.ModuleList(
            [LowShear(width) if i % 2 == 0 else UpShear(width) for i in range(n_layers)]
        )
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_pair_model(model_cls, X_train, Y_train, epochs=15000, lr=1e-3, seed=0, **kw):
    torch.manual_seed(seed)
    model = model_cls(**kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    Yt = torch.tensor(Y_train, dtype=torch.float32)
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(Xt)
        loss = ((pred - Yt) ** 2).mean()
        loss.backward()
        opt.step()
        sch.step()
    return model


def rollout(model, x0, n_steps):
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = x0
    x = torch.tensor(x0, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        for k in range(n_steps):
            x = model(x)
            traj[k + 1] = x.squeeze(0).numpy()
    return traj


# ---------- main experiment ----------
def main():
    full = np.loadtxt("../data/full_trajectory.txt")    # (140, 2)
    h = 0.1
    t_arr = np.arange(140) * h
    clean_train_X = full[:40]
    clean_train_Y = full[1:41]
    true_test = full[40:]

    results = {}
    rng = np.random.default_rng(42)

    for sigma in [0.0, 0.02, 0.05]:
        noise = rng.normal(0, sigma, size=full.shape)
        noisy_data = full + noise
        Xn = noisy_data[:40]
        Yn = noisy_data[1:41]

        fnn = train_pair_model(FNN, Xn, Yn, hidden=64, layers=3, seed=0)
        sym = train_pair_model(SympNet, Xn, Yn, n_layers=8, width=32, seed=0)

        x0 = full[40]    # use TRUE x0 so rollout error is isolated
        fnn_traj = rollout(fnn, x0, 99)
        sym_traj = rollout(sym, x0, 99)

        results[sigma] = {
            "FNN": {
                "traj": fnn_traj,
                "rollout_mse": ((fnn_traj - true_test)**2).mean(),
                "final_mse": ((fnn_traj[-1] - true_test[-1])**2).mean(),
                "energy_drift": np.abs(H(fnn_traj[:, 1], fnn_traj[:, 0]) - H(1, 0)).max(),
            },
            "SympNet": {
                "traj": sym_traj,
                "rollout_mse": ((sym_traj - true_test)**2).mean(),
                "final_mse": ((sym_traj[-1] - true_test[-1])**2).mean(),
                "energy_drift": np.abs(H(sym_traj[:, 1], sym_traj[:, 0]) - H(1, 0)).max(),
            },
        }
        print(f"sigma = {sigma}")
        for k in ("FNN", "SympNet"):
            r = results[sigma][k]
            print(f"  {k:8s}: rollout MSE = {r['rollout_mse']:.3e}, "
                  f"final = {r['final_mse']:.3e}, "
                  f"energy drift = {r['energy_drift']:.3e}")

    # ---------- summary plots ----------
    sigmas = [0.0, 0.02, 0.05]
    fnn_mse = [results[s]["FNN"]["rollout_mse"] for s in sigmas]
    sym_mse = [results[s]["SympNet"]["rollout_mse"] for s in sigmas]
    fnn_E = [results[s]["FNN"]["energy_drift"] for s in sigmas]
    sym_E = [results[s]["SympNet"]["energy_drift"] for s in sigmas]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(sigmas))
    w = 0.35
    ax[0].bar(x - w/2, fnn_mse, w, color="#B85042", label="FNN")
    ax[0].bar(x + w/2, sym_mse, w, color="#2C5F2D", label="SympNet")
    ax[0].set_yscale("log")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"$\\sigma={s}$" for s in sigmas])
    ax[0].set_ylabel("rollout MSE over 99 steps")
    ax[0].set_title("Noise robustness: rollout error")
    ax[0].legend(frameon=False)

    ax[1].bar(x - w/2, fnn_E, w, color="#B85042", label="FNN")
    ax[1].bar(x + w/2, sym_E, w, color="#2C5F2D", label="SympNet")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"$\\sigma={s}$" for s in sigmas])
    ax[1].set_ylabel(r"max $|H - H_0|$ in rollout")
    ax[1].set_title("Noise robustness: energy drift")
    ax[1].legend(frameon=False)

    plt.tight_layout()
    plt.savefig("../figures/task4_noise_robustness.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---------- phase portraits under noise ----------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i, s in enumerate(sigmas):
        ax = axes[i]
        ax.plot(full[:, 1], full[:, 0], "-", color="#065A82",
                alpha=0.4, label="true")
        ax.plot(results[s]["FNN"]["traj"][:, 1],
                results[s]["FNN"]["traj"][:, 0],
                "x", color="#B85042", ms=4, alpha=0.8, label="FNN")
        ax.plot(results[s]["SympNet"]["traj"][:, 1],
                results[s]["SympNet"]["traj"][:, 0],
                "+", color="#2C5F2D", ms=5, alpha=0.9, label="SympNet")
        ax.set_xlabel(r"$\theta$"); ax.set_ylabel(r"$p_\theta$")
        ax.set_title(f"$\\sigma$ = {s}")
        ax.legend(frameon=False, fontsize=8)
        ax.set_aspect("equal")
    plt.tight_layout()
    plt.savefig("../figures/task4_noise_phase.png", dpi=160, bbox_inches="tight")
    plt.close()

    # save summary
    with open("../results/task4_summary.txt", "w") as f:
        f.write("sigma,model,rollout_mse,final_mse,energy_drift\n")
        for s in sigmas:
            for k in ("FNN", "SympNet"):
                r = results[s][k]
                f.write(f"{s},{k},{r['rollout_mse']:.6e},"
                        f"{r['final_mse']:.6e},{r['energy_drift']:.6e}\n")


if __name__ == "__main__":
    main()
