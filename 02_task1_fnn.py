"""
Task 1: Train an FNN x_{i-1} -> x_i on 40 train points, evaluate
        (a) teacher-forcing one-step error on 100 test pairs,
        (b) rollout error from x_40 forward.

We use PyTorch. No physics structure is imposed: a generic MLP.
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

# ---------- data ----------
full = np.loadtxt("../data/full_trajectory.txt")    # columns: p, theta
# project description: "first column = p_theta, second column = theta"
# train: indices 0..39, test: 40..139
N = full.shape[0]
H_STEP = 0.1

# inputs x_{i-1}, targets x_i
X_all = full[:-1]    # (139, 2)
Y_all = full[1:]     # (139, 2)

# Train: (x_{i-1}, x_i) for i=1..40 -> 40 pairs
X_train = X_all[:40]
Y_train = Y_all[:40]
# Test: rest
X_test = X_all[40:]
Y_test = Y_all[40:]

X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=DEVICE)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32, device=DEVICE)
Y_test_t  = torch.tensor(Y_test,  dtype=torch.float32, device=DEVICE)


# ---------- FNN ----------
class FNN(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        mods = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*mods)

    def forward(self, x):
        return self.net(x)


def train_fnn(seed=0, epochs=20000, lr=1e-3):
    torch.manual_seed(seed)
    model = FNN(hidden=64, layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)
    losses = []
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_train_t)
        loss = ((pred - Y_train_t) ** 2).mean()
        loss.backward()
        opt.step()
        scheduler.step()
        if ep % 1000 == 0:
            losses.append((ep, loss.item()))
    return model, losses


def rollout(model, x0, n_steps):
    """Apply model recursively n_steps times starting from x0 (1D np array of size 2)."""
    traj = np.zeros((n_steps + 1, 2))
    traj[0] = x0
    x = torch.tensor(x0, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        for k in range(n_steps):
            x = model(x)
            traj[k + 1] = x.squeeze(0).cpu().numpy()
    return traj


def H(theta, p):
    return 0.5 * p**2 + (1.0 - np.cos(theta))


def main():
    model, losses = train_fnn()

    # ---------- evaluation ----------
    # (a) teacher forcing on the 99 test pairs (indices 40..138 -> 41..139)
    with torch.no_grad():
        Y_pred_test = model(X_test_t).cpu().numpy()
    tf_err = ((Y_pred_test - Y_test) ** 2).mean(axis=1)   # per-step
    tf_mse = tf_err.mean()

    # (b) rollout starting from x_40 for 99 steps -> predicted x_41..x_139
    x40 = full[40]
    pred_traj = rollout(model, x40, 99)    # shape (100, 2)
    true_traj = full[40:]                  # shape (100, 2)
    rollout_err = ((pred_traj - true_traj) ** 2).mean(axis=1)
    rollout_mse = rollout_err.mean()

    print(f"FNN teacher-forcing MSE on 99 test pairs: {tf_mse:.3e}")
    print(f"FNN rollout MSE over 99 steps:            {rollout_mse:.3e}")
    print(f"FNN rollout MSE at final step:            {rollout_err[-1]:.3e}")

    # save predictions for cross-method comparison
    np.savetxt("../results/fnn_rollout.txt", pred_traj)
    np.savetxt("../results/fnn_tf_err.txt", tf_err)
    np.savetxt("../results/fnn_rollout_err.txt", rollout_err)

    # ---------- plots ----------
    t_test = np.arange(40, 140) * H_STEP

    # Phase portrait
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(full[:, 1], full[:, 0], "-", color="#065A82",
               alpha=0.4, label="true full trajectory")
    ax[0].plot(true_traj[:, 1], true_traj[:, 0], "o", color="#065A82",
               ms=3, label="true (test)")
    ax[0].plot(pred_traj[:, 1], pred_traj[:, 0], "x", color="#B85042",
               ms=4, label="FNN rollout")
    ax[0].plot(x40[1], x40[0], "s", color="black", ms=8, label="start")
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_ylabel(r"$p_\theta$")
    ax[0].set_title("Phase portrait: FNN rollout vs true")
    ax[0].legend(frameon=False, fontsize=9)
    ax[0].set_aspect("equal")

    ax[1].semilogy(t_test[1:], tf_err, "o-", color="#1C7293",
                   ms=3, label="teacher-forcing (one-step)")
    ax[1].semilogy(t_test[1:], rollout_err[1:], "s-", color="#B85042",
                   ms=3, label="rollout (recursive)")
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel("per-step MSE")
    ax[1].set_title("FNN: one-step vs rollout error")
    ax[1].legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig("../figures/task1_fnn.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Energy diagnostic
    E_true = H(true_traj[:, 1], true_traj[:, 0])
    E_fnn  = H(pred_traj[:, 1], pred_traj[:, 0])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t_test, E_true, color="#065A82", label="true")
    ax.plot(t_test, E_fnn,  color="#B85042", label="FNN rollout")
    ax.axhline(H(1.0, 0.0), color="gray", linestyle="--", alpha=0.7,
               label="initial energy")
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$H(\theta, p_\theta)$")
    ax.set_title("Energy in rollout: FNN does NOT conserve H")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("../figures/task1_fnn_energy.png", dpi=160, bbox_inches="tight")
    plt.close()

    print(f"FNN energy drift over rollout: max |E - E0| = {np.abs(E_fnn - H(1.0, 0.0)).max():.3e}")

    # save model
    torch.save(model.state_dict(), "../results/fnn_model.pt")

if __name__ == "__main__":
    main()
