import numpy as np
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import torch

from .utils import get_new_model_and_trainer, get_dataloader

cwd = Path(__file__).parent

# feature distributions: circle and ellipse
def circle_dist(n_size=10000):
    # x**2 + y**2 = r**2
    r = 1
    x = (np.random.ranf(size=n_size) - 0.5) * 2
    y = (np.random.ranf(size=n_size) - 0.5) * 2
    idx = (np.square(x) + np.square(y)) < r**2
    print(sum(idx))
    return np.stack([x[idx], y[idx]], axis=1)


def ellipse_dist(n_size=10000):
    # x^2/a^2 + y^2/b^2 = r^2
    a, b, r = 2, 0.5, 1
    x = (np.random.ranf(size=n_size) - 0.5) * 4
    y = np.random.ranf(size=n_size) - 0.5
    idx = (np.square(x) / a**2 + np.square(y) / b**2) < r**2
    print(sum(idx))
    return np.stack([x[idx], y[idx]], axis=1)


def get_label(x: np.ndarray, y: np.ndarray, noise: float=0.1):
    # label function y>x
    return ((x**2+np.random.normal(0, noise, size=(len(y))))*6< y+0.6).astype(int)


def plot_data_points(
    dists: list[np.ndarray], colors: Optional[list[list[int]]] = None, title=None
):
    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)
    lim_val = 1
    colors = (
        cm.plasma(np.arange(0, 1, 1 / (len(dists) + 1))) if colors is None else colors
    )
    colors = np.array(colors)
    for ii, dist in enumerate(dists):
        # x: dist[:,0], y: dist[:,1], label: dist[:,2]
        lab = np.unique(dist[:, 2])
        for jj, ll in enumerate(lab):
            idx_ll = np.where(dist[:, 2] == ll)[0]
            ax.scatter(
                dist[idx_ll, 0],
                dist[idx_ll, 1],
                alpha=0.1,
            )
        lim_val = max(lim_val, np.max(dist))
    plt.xlim([-lim_val, lim_val])
    plt.ylim([-lim_val, lim_val])
    plt.title(title)
    return fig


if __name__ == "__main__":
    cir = circle_dist(10000)
    cir_tar = get_label(cir[:, 0], cir[:, 1])
    ell = ellipse_dist(10000)
    ell_tar = get_label(ell[:, 0], ell[:, 1])
    n_cir = 6000
    n_ell = 10

    fig = plot_data_points(
        [
            np.concatenate([cir[:n_cir], cir_tar[:n_cir].reshape((-1, 1))], axis=1),
            np.concatenate([ell[:n_ell], ell_tar[:n_ell].reshape((-1, 1))], axis=1),
        ],
        [[1, 0.3, 0.3], [0.3, 1, 0.3]],
        title="trainset",
    )
    fig.savefig(cwd / "results" / "trainset.png")

    # train model
    train_loader = get_dataloader(
        np.concatenate([cir[:n_cir, :], ell[:n_ell, :]], axis=0),
        np.concatenate([cir_tar[:n_cir], ell_tar[:n_ell]], axis=0),
        shuffle=True
    )
    val_loader = get_dataloader(
        np.concatenate([cir[n_cir:, :], ell[n_ell:n_ell+500, :]], axis=0),
        np.concatenate([cir_tar[n_cir:], ell_tar[n_ell:n_ell+500]], axis=0),
        shuffle=True
    )
    test_loader = get_dataloader(
        np.concatenate([cir[n_cir:, :], ell[n_ell:n_ell+500, :]], axis=0) + np.random.normal(size=(len(cir)-n_cir+500,2)),
        np.concatenate([cir_tar[n_cir:], ell_tar[n_ell:n_ell+500]], axis=0),
        shuffle=True
    )
    model, trainer = get_new_model_and_trainer(save_dir=cwd/"results"/"models", n_epochs=100)
    model_ckpt = cwd/"results"/"models"/ "checkpoints"/"last.ckpt"
    if model_ckpt.exists():
        print(f"Found checkpoint in {model_ckpt}, resuming training.")
    else:
        model_ckpt = None
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=model_ckpt,
        )
    test_dist = np.concatenate([cir[n_cir:, :], ell[n_ell:n_ell+500, :]], axis=0) + np.random.normal(0, 0.3,size=(len(cir)-n_cir+500,2))
    test_tar = np.concatenate([cir_tar[n_cir:], ell_tar[n_ell:n_ell+500]], axis=0)
    est = model(torch.FloatTensor(test_dist)).cpu().numpy()[:,0]
    est_val = ((est>0.5).astype(int)==test_tar).astype(int).reshape((-1,1))

    fig = plot_data_points([np.concatenate([test_dist, (est>0.5).astype(int).reshape((-1,1))], axis=1)])
    fig.savefig(cwd / "results" / "testset_lab.png")

    fig = plot_data_points([np.concatenate([test_dist, est_val], axis=1)])
    fig.savefig(cwd / "results" / "testset_acc.png")


