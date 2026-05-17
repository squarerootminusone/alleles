import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Fixed parameters
mu = 1e-8
p_target = 0.5

# Parameter ranges
Ne_vals = np.logspace(4, 9, 180)    # 10^4 to 10^9
s_vals = np.logspace(-5, 0, 180)    # 10^-5 to 10^0 = 1
Ne_grid, s_grid = np.meshgrid(Ne_vals, s_vals)

# Exact establishment probability
P_est = (1 - np.exp(-2 * s_grid)) / (1 - np.exp(-4 * Ne_grid * s_grid))

# Establishment frequency approximation
p_est = 1 / (2 * Ne_grid * s_grid)

# Valid region: growth to target only makes sense if p_est < p_target
valid = (P_est > 0) & (p_est > 0) & (p_est < p_target)

# Exact proportion R = T_phase1 / T_phase2
R = np.full_like(Ne_grid, np.nan, dtype=float)

T_phase1 = (
    1 / (2 * Ne_grid[valid] * mu * P_est[valid])
    + np.log(1 / s_grid[valid]) / s_grid[valid]
)

T_phase2 = (
    1 / s_grid[valid]
) * np.log(
    (p_target * (1 - p_est[valid])) / (p_est[valid] * (1 - p_target))
)

R[valid] = T_phase1 / T_phase2

# Clip values above 100 for display
R_clipped = np.clip(R, None, 100)

# Log coordinates for 3D plotting
X = np.log10(Ne_grid)
Y = np.log10(s_grid)
Z = np.ma.masked_invalid(np.log10(R_clipped))  # z = log10(clipped ratio)

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")

# Main surface
ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, alpha=0.95)

# Transparent plane at ratio = 1
# Since z-axis is log10(ratio), ratio = 1 corresponds to z = 0
X_plane, Y_plane = np.meshgrid(
    np.linspace(4, 9, 30),
    np.linspace(-5, 0, 30)
)
Z_plane = np.zeros_like(X_plane)

ax.plot_surface(
    X_plane,
    Y_plane,
    Z_plane,
    color="red",
    alpha=0.20,
    linewidth=0,
    shade=False
)

# ---------------------------------------------------------
# Green line 1: cross-section at fixed s = 0.1, varying Ne
# ---------------------------------------------------------
s_fixed = 0.1
Ne_line1 = np.logspace(4, 9, 500)

P_est_line1 = (1 - np.exp(-2 * s_fixed)) / (1 - np.exp(-4 * Ne_line1 * s_fixed))
p_est_line1 = 1 / (2 * Ne_line1 * s_fixed)

valid1 = (P_est_line1 > 0) & (p_est_line1 > 0) & (p_est_line1 < p_target)

R_line1 = np.full_like(Ne_line1, np.nan, dtype=float)
R_line1[valid1] = (
    1 / (2 * Ne_line1[valid1] * mu * P_est_line1[valid1])
    + np.log(1 / s_fixed) / s_fixed
) / (
    (1 / s_fixed) * np.log(
        (p_target * (1 - p_est_line1[valid1])) / (p_est_line1[valid1] * (1 - p_target))
    )
)

R_line1 = np.clip(R_line1, None, 100)

ax.plot(
    np.log10(Ne_line1[valid1]),
    np.full(np.sum(valid1), np.log10(s_fixed)),
    np.log10(R_line1[valid1]),
    color="limegreen",
    linewidth=3,
    label=r"$s = 0.1$"
)

# ---------------------------------------------------------
# Green line 2: cross-section at fixed Ne = 10^6, varying s
# ---------------------------------------------------------
Ne_fixed = 1e6
s_line2 = np.logspace(-5, 0, 500)

P_est_line2 = (1 - np.exp(-2 * s_line2)) / (1 - np.exp(-4 * Ne_fixed * s_line2))
p_est_line2 = 1 / (2 * Ne_fixed * s_line2)

valid2 = (P_est_line2 > 0) & (p_est_line2 > 0) & (p_est_line2 < p_target)

R_line2 = np.full_like(s_line2, np.nan, dtype=float)
R_line2[valid2] = (
    1 / (2 * Ne_fixed * mu * P_est_line2[valid2])
    + np.log(1 / s_line2[valid2]) / s_line2[valid2]
) / (
    (1 / s_line2[valid2]) * np.log(
        (p_target * (1 - p_est_line2[valid2])) / (p_est_line2[valid2] * (1 - p_target))
    )
)

R_line2 = np.clip(R_line2, None, 100)

ax.plot(
    np.full(np.sum(valid2), np.log10(Ne_fixed)),
    np.log10(s_line2[valid2]),
    np.log10(R_line2[valid2]),
    color="green",
    linewidth=3,
    linestyle="--",
    label=r"$N_e = 10^6$"
)

# Labels
ax.set_xlabel(r"$\log_{10}(N_e)$")
ax.set_ylabel(r"$\log_{10}(s)$")
ax.set_zlabel(
    r"$\log_{10}\!\left(\min\left(\frac{T_{\mathrm{phase\ 1}}}{T_{\mathrm{phase\ 2}}},\,100\right)\right)$"
)
ax.set_title(
    "Exact proportion of phase-1 time to phase-2 time\n"
    r"($\mu = 10^{-8}$, target allele frequency $p = 0.5$, clipped at 100)"
)

# Ticks
ax.set_xticks([4, 5, 6, 7, 8, 9])
ax.set_xticklabels([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"])

ax.set_yticks([-5, -4, -3, -2, -1, 0])
ax.set_yticklabels([r"$10^{-5}$", r"$10^{-4}$", r"$10^{-3}$", r"$10^{-2}$", r"$10^{-1}$", r"$10^0$"])

ax.set_zticks([-1, 0, 1, 2])  # corresponds to 0.1, 1, 10, 100
ax.set_zticklabels([r"$10^{-1}$", r"$10^0$", r"$10^1$", r"$10^2$"])

# View angle
ax.view_init(elev=28, azim=-60)

ax.legend(loc="upper left")
plt.tight_layout()
plt.show()