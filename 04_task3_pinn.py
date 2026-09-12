"""
Task 3: PINN for parameter estimation of (m, l) from trajectory data.

The pendulum ODE (with g = 1 fixed) is
    dtheta/dt = p_theta / (m l^2)
    dp_theta/dt = -m * 1 * l * sin(theta)
                = -(m l) sin(theta)

So only two SCALAR combinations of (m, l) appear in the dynamics:
    alpha := m l^2     (in the kinematic equation)
    beta  := m l       (in the momentum equation)

This means we have a 2-dim parameter space being projected from a
2-dim parameter space via (m, l) -> (alpha, beta). Solving back:
    l = alpha / beta,    m = beta^2 / alpha.
Hence m and l ARE individually identifiable from data on (theta, p_theta)
once both alpha and beta are pinned down. So with g fixed, both m and l
are identifiable in principle — but identifiability degrades if the
trajectory is short/noise-dominated.

(Aside: from (theta, p_theta) the angular frequency for small
oscillations is omega^2 = g / l = 1 / l, so l is identifiable purely
from the period; m drops out of theta(t) at small amplitude, but enters
p_theta(t) = m l^2 dtheta/dt and so is identifiable from p_theta.)

PINN setup:
    Network: t -> (theta_hat(t), p_hat(t))
    Data loss: MSE between network and 40 training points
    Physics loss: residuals of the two ODEs with learnable (m, l)
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


class PINN(nn.Module):
    """Network t -> (theta, p_theta)."""
    def __init__(self, hidden=64, layers=4):
        super().__init__()
        mods = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            mods += [nn.Linear(hidden, hidden), nn.Tanh()]
        mods += [nn.Linear(hidden, 2)]
        self.net = nn.Sequential(*mods)

    def forward(self, t):
        return self.net(t)


def main():
    full = np.loadtxt("../data/full_trajectory.txt")
    h = 0.1
    t_all = (np.arange(140) * h).reshape(-1, 1)
    # training: first 40 points
    t_train = t_all[:40]
    p_train = full[:40, 0:1]
    th_train = full[:40, 1:2]

    t_train_t = torch.tensor(t_train, dtype=torch.float32)
    p_train_t = torch.tensor(p_train, dtype=torch.float32)
    th_train_t = torch.tensor(th_train, dtype=torch.float32)

    # collocation points for physics loss — denser than data
    t_coll = torch.linspace(0, t_train[-1, 0], 200).unsqueeze(1)
    t_coll.requires_grad_(True)

    # learnable parameters m, l (log-parameterised to enforce positivity)
    # initial guesses purposely off
    log_m = nn.Parameter(torch.tensor(np.log(1.5), dtype=torch.float32))
    log_l = nn.Parameter(torch.tensor(np.log(1.5), dtype=torch.float32))

    model = PINN(hidden=64, layers=4)
    params = list(model.parameters()) + [log_m, log_l]
    opt = torch.optim.Adam(params, lr=2e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5000, gamma=0.5)

    LAMBDA_PHYS = 1.0
    history = {"data": [], "phys": [], "m": [], "l": []}

    EPOCHS = 25000
    for ep in range(EPOCHS):
        opt.zero_grad()

        m = torch.exp(log_m)
        l = torch.exp(log_l)

        # data fit
        pred = model(t_train_t)
        loss_data = ((pred[:, 0:1] - p_train_t) ** 2).mean() \
                  + ((pred[:, 1:2] - th_train_t) ** 2).mean()

        # physics: dtheta/dt = p / (m l^2),   dp/dt = -m l sin(theta)
        out = model(t_coll)
        p_hat = out[:, 0:1]
        th_hat = out[:, 1:2]
        dp_dt = torch.autograd.grad(p_hat, t_coll,
                                    grad_outputs=torch.ones_like(p_hat),
                                    create_graph=True)[0]
        dth_dt = torch.autograd.grad(th_hat, t_coll,
                                     grad_outputs=torch.ones_like(th_hat),
                                     create_graph=True)[0]
        res_th = dth_dt - p_hat / (m * l**2)
        res_p  = dp_dt + m * l * torch.sin(th_hat)   # g = 1
        loss_phys = (res_th**2).mean() + (res_p**2).mean()

        loss = loss_data + LAMBDA_PHYS * loss_phys
        loss.backward()
        opt.step()
        scheduler.step()

        if ep % 500 == 0:
            history["data"].append(loss_data.item())
            history["phys"].append(loss_phys.item())
            history["m"].append(torch.exp(log_m).item())
            history["l"].append(torch.exp(log_l).item())

        if ep % 5000 == 0:
            print(f"  ep {ep:5d}: data {loss_data.item():.3e} "
                  f"phys {loss_phys.item():.3e} "
                  f"m {torch.exp(log_m).item():.4f} "
                  f"l {torch.exp(log_l).item():.4f}")

    m_hat = float(torch.exp(log_m).detach())
    l_hat = float(torch.exp(log_l).detach())
    print(f"\nLearned: m_hat = {m_hat:.4f}, l_hat = {l_hat:.4f}")
    print(f"True:    m = 1.0000,    l = 1.0000")
    print(f"Combinations learned: m_hat * l_hat^2 = {m_hat * l_hat**2:.4f}, "
          f"m_hat * l_hat = {m_hat * l_hat:.4f}")

    # ---------- forward integrate with learned parameters ----------
    def sv_step(p, th, h, m, l):
        # general pendulum: dp/dt = -m l sin(th), dth/dt = p/(m l^2)
        p_half = p - 0.5 * h * m * l * np.sin(th)
        th_new = th + h * p_half / (m * l**2)
        p_new = p_half - 0.5 * h * m * l * np.sin(th_new)
        return p_new, th_new

    def integrate(m, l, p0, th0, h, n):
        p = np.zeros(n); th = np.zeros(n)
        p[0], th[0] = p0, th0
        for i in range(n - 1):
            p[i + 1], th[i + 1] = sv_step(p[i], th[i], h, m, l)
        return p, th

    p_pred, th_pred = integrate(m_hat, l_hat, 0.0, 1.0, h, 140)
    pred_traj = np.column_stack([p_pred, th_pred])
    mse_per_step = ((pred_traj - full)**2).mean(axis=1)

    # ---------- sensitivity: +-1% perturbations ----------
    pert_results = {}
    for name, (dm, dl) in [("m+1%", (0.01, 0)),
                           ("m-1%", (-0.01, 0)),
                           ("l+1%", (0, 0.01)),
                           ("l-1%", (0, -0.01))]:
        p_p, th_p = integrate(m_hat * (1 + dm), l_hat * (1 + dl), 0.0, 1.0, h, 140)
        pert_results[name] = np.column_stack([p_p, th_p])

    # save
    np.savetxt("../results/pinn_trajectory.txt", pred_traj)
    np.savetxt("../results/pinn_mse_per_step.txt", mse_per_step)
    with open("../results/pinn_params.txt", "w") as f:
        f.write(f"m_hat = {m_hat:.6f}\nl_hat = {l_hat:.6f}\n")

    # ---------- plots ----------
    t_arr = np.arange(140) * h

    # Trajectory comparison
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(full[:, 1], full[:, 0], "-", color="#065A82",
               alpha=0.5, label="true")
    ax[0].plot(pred_traj[:, 1], pred_traj[:, 0], "--", color="#B85042",
               alpha=0.8, label="PINN parameters")
    ax[0].axvline(full[40, 1], color="gray", linestyle=":",
                  alpha=0.5)
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_ylabel(r"$p_\theta$")
    ax[0].set_title("Phase portrait with learned (m, l)")
    ax[0].legend(frameon=False)
    ax[0].set_aspect("equal")

    ax[1].semilogy(t_arr, mse_per_step + 1e-16, color="#B85042",
                   label="PINN integrate")
    ax[1].axvline(t_train[-1, 0], color="gray", linestyle=":",
                  alpha=0.7, label="end of training window")
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel("per-step MSE")
    ax[1].set_title("PINN: trajectory error from integrating learned (m, l)")
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig("../figures/task3_pinn.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Parameter convergence
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs_rec = np.arange(len(history["m"])) * 500
    ax[0].plot(epochs_rec, history["m"], color="#B85042", label=r"$\hat m$")
    ax[0].plot(epochs_rec, history["l"], color="#2C5F2D", label=r"$\hat l$")
    ax[0].axhline(1.0, color="gray", linestyle="--",
                  label="true value")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("parameter value")
    ax[0].set_title("Convergence of learned (m, l)")
    ax[0].legend(frameon=False)

    ax[1].semilogy(epochs_rec, history["data"], color="#065A82",
                   label="data MSE")
    ax[1].semilogy(epochs_rec, history["phys"], color="#B85042",
                   label="physics residual")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss")
    ax[1].set_title("PINN losses")
    ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig("../figures/task3_convergence.png", dpi=160, bbox_inches="tight")
    plt.close()

    # Sensitivity plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].plot(full[:, 1], full[:, 0], "-", color="#065A82",
               alpha=0.7, lw=2, label="true / nominal")
    cols = {"m+1%": "#B85042", "m-1%": "#F96167",
            "l+1%": "#2C5F2D", "l-1%": "#69A297"}
    for name, traj in pert_results.items():
        ax[0].plot(traj[:, 1], traj[:, 0], "--", color=cols[name],
                   alpha=0.7, label=name)
    ax[0].set_xlabel(r"$\theta$")
    ax[0].set_ylabel(r"$p_\theta$")
    ax[0].set_title(r"Trajectory under $\pm 1\%$ perturbation of $\hat m, \hat l$")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_aspect("equal")

    for name, traj in pert_results.items():
        err = ((traj - full)**2).mean(axis=1)
        ax[1].semilogy(t_arr, err + 1e-16, color=cols[name], label=name)
    ax[1].set_xlabel("time t")
    ax[1].set_ylabel("per-step MSE")
    ax[1].set_title("Long-time MSE under parameter perturbation")
    ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig("../figures/task3_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close()

    return m_hat, l_hat, pred_traj, mse_per_step


if __name__ == "__main__":
    main()
