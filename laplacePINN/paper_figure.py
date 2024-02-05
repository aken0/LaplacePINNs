from importlib import reload

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from matplotlib import colors
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.stats import chi2
from tueplots import figsizes, bundles, axes


plt.rcParams.update(bundles.iclr2024(nrows=0.98, ncols=1))
plt.rcParams.update(axes.lines())
plt.rcParams.update(axes.grid())
plt.rcParams.update(axes.spines())
# plt.rcParams["axes.unicode_minus"] = False
# plt.rcParams["figure.constrained_layout.use"] = False

###############################################################################
# Load Data
###############################################################################
data = scipy.io.loadmat("data/burgers_shock.mat")
t_ = torch.tensor(data["t"].flatten()).float()
x_ = torch.tensor(data["x"].flatten()).float()
Exact = np.real(data["usol"]).T

# X,T=torch.meshgrid(x_,t_,indexing='xy')
# X,T=torch.meshgrid(x_,t_)
X, T = np.meshgrid(x_, t_, indexing="xy")
X = torch.tensor(X)
T = torch.tensor(T)
X_exact = X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

t = torch.linspace(0, 1, 251).float()
x = torch.linspace(-1, 1, 501).float()
bc_l = torch.vstack([-1 * torch.ones_like(t), t]).T
bc_r = torch.vstack([torch.ones_like(t), t]).T
ic = torch.vstack([x, torch.zeros_like(x)]).T
data_grid = torch.vstack([bc_l, bc_r, ic])
x = data_grid[:, 0]
t = data_grid[:, 1]

eee = np.load(
    "data/experiment2,power2equalGrid2.npz",
    allow_pickle=True,
)
errs = eee["arr_0"].squeeze()[:-1, :]
var_cpu = eee["arr_2"].squeeze()[:-1, :]
mean_cpu = eee["arr_1"].squeeze()[:-1, :]
covariances_cpu = eee["arr_3"].squeeze()[:-1, :]
grids = eee["arr_4"].squeeze()[:-1]


###############################################################################
# Plot
###############################################################################
cmap = "Spectral_r"
index = idx = [2, 4, 6, 9]
arr = np.array(
    [
        mean_cpu[idx, :],
        errs[idx, :] ** 2,
        np.sqrt(var_cpu[idx, :] ** 2),
        ((errs**2) / var_cpu)[idx, :],
    ]
).transpose((1, 0, -1))
arr = np.vstack(arr)

rows = 4
cols = 4
fig, axes = plt.subplots(
    nrows=rows,
    ncols=cols,
    sharex=True,
    sharey=True,
    constrained_layout=False,
)

alpha = 0.05
index = range(rows * cols)
for i in range(rows):
    for j in range(cols):
        ax = axes[i, j]
        arr_idx = i * 4 + j
        if j == 3:
            c1, c2 = chi2.ppf([alpha / 2, 1 - alpha / 2], 1)
            bounds = [c1, c2]
            divnorm = colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend="both")
            U_Y = griddata(X_exact, arr[arr_idx, :], (X, T), method="nearest")
            # gs0 = gridspec.GridSpec(1, 2)
            # gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
            h = ax.imshow(
                U_Y.T,
                interpolation="nearest",
                norm=divnorm,
                # cmap="seismic",
                cmap=cmap,
                extent=[t.min(), t.max(), x.min(), x.max()],
                origin="lower",
                aspect="auto",
            )
        else:
            U_Y = griddata(X_exact, arr[arr_idx], (X, T), method="nearest")
            # gs0 = gridspec.GridSpec(1, 2)
            # gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
            h = ax.imshow(
                (U_Y).T,
                interpolation="nearest",
                cmap=cmap,
                extent=[t.min(), t.max(), x.min(), x.max()],
                origin="lower",
                aspect="auto",
                vmin=[-1, 0, 0][j],
                vmax=[1, None, None][j],
            )

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.02)

            cbformat = ScalarFormatter()  # create the formatter
            cbformat.set_powerlimits((-2, 2))
            plt.colorbar(h, cax=cax, orientation="vertical", format=cbformat)

            cax.tick_params(length=1.5, labelsize=8, pad=1)

        plot_points = grids[idx[i]]
        x_col = plot_points[:, 0]
        t_col = plot_points[:, 1]
        ax.plot(t_col, x_col, "kx", markersize=0.2, clip_on=False)
        ax.set_xticks([0, 1])

axes[0, 0].set_title("PDE Solution", pad=15)
axes[0, 1].set_title("Squared Error", pad=15)
axes[0, 2].set_title("Predicted Variance", pad=15)
axes[0, 3].set_title("Calibration", pad=15)
# axes[0,0].set_ylabel(idx[0],rotation=0,labelpad=-1)
# axes[1,0].set_ylabel(idx[1],rotation=0,labelpad=-1)
# axes[2,0].set_ylabel(idx[2],rotation=0,labelpad=-1)
# axes[3,0].set_ylabel(idx[3],rotation=0,labelpad=-1)
axes[0, 0].set_ylabel("$N=2^2$", labelpad=-1)
axes[1, 0].set_ylabel("$N=2^4$", labelpad=-1)
axes[2, 0].set_ylabel("$N=2^6$", labelpad=-1)
axes[3, 0].set_ylabel("$N=2^{10}$", labelpad=-1)


fig.supxlabel("$t$")
fig.supylabel("$x$")

# fig.tight_layout()
fig.subplots_adjust(
    top=0.8,
    hspace=0.4,
    wspace=0.4,
)
fig.savefig(
    "../../figures/fig.pdf",
    bbox_inches="tight",
)
plt.close()
