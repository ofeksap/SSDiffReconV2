#!/usr/bin/env python3
"""Plot zero-filled reference images from the test_reference folder."""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser(description="Plot .npy images from a folder")
    ap.add_argument("folder", nargs="?", default="./results/ixi_samples_iter1/test_reference",
                    help="Folder containing im_*samples.npy files")
    ap.add_argument("--indices", type=str, default="0,100,500,1000",
                    help="Comma-separated indices to plot (e.g. 0,100,500)")
    ap.add_argument("--rows", type=int, default=2, help="Number of rows in grid")
    ap.add_argument("--save", type=str, default=None, help="Save figure to path instead of showing")
    ap.add_argument("--dpi", type=int, default=100, help="DPI for saved figure (higher = larger file, 1:1 pixels at 100)")
    args = ap.parse_args()

    indices = [int(x.strip()) for x in args.indices.split(",")]
    n = len(indices)
    ncols = (n + args.rows - 1) // args.rows
    nrows = min(args.rows, n)

    # Get image shape from first available file (for 1:1 pixel mapping in saved PNG)
    img_shape = None
    for idx in indices:
        path = os.path.join(args.folder, f"im_{idx}samples.npy")
        if os.path.isfile(path):
            img_shape = np.squeeze(np.load(path)).shape
            break
    if img_shape is None:
        img_shape = (256, 256)  # fallback
    h, w = img_shape[0], img_shape[1]
    # Figure size in inches so each subplot is (w/dpi) x (h/dpi) -> saved at 1:1 pixel
    figsize = (ncols * w / args.dpi, nrows * h / args.dpi)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)

    for i, idx in enumerate(indices):
        path = os.path.join(args.folder, f"im_{idx}samples.npy")
        if not os.path.isfile(path):
            print("Missing:", path)
            continue
        # img = np.squeeze(np.load(path))
        img = np.load(path)  # keeps (256,256); handles (1,256,256) from other sources
        ax = axes.flat[i]
        ax.imshow(img, cmap="gray")
        ax.set_title(f"index {idx}")
        ax.axis("off")

    for i in range(n, nrows * ncols):
        axes.flat[i].axis("off")

    plt.suptitle("Zero-filled reference (test_reference)")
    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print("Saved:", args.save)
    else:
        plt.show()

if __name__ == "__main__":
    main()
