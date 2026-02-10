"""
Analyze evaluation results: PSNR, SSIM, NMSE between reconstructions and reference.

For the original model outputs in results/ixi_samples/test/:

  1. Export reference images (same order as eval) once:
       python export_test_reference.py
     This creates data/ixi_test_reference/ with im_*samples.npy (not inside results/).

  2. Run this script:
       python run_analysis.py
     Or with custom paths:
       python run_analysis.py --recon results/ixi_samples/test --gt data/ixi_test_reference
"""
import argparse
import numpy as np
import os
import glob
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def analyze_results(recon_dir, gt_dir):
    # Find all .npy files in the reconstruction directory
    recon_files = sorted(glob.glob(os.path.join(recon_dir, "*.npy")))
    
    psnr_list = []
    ssim_list = []
    nmse_list = []

    for r_file in recon_files:
        # Construct path to the corresponding ground truth file
        filename = os.path.basename(r_file)
        g_file = os.path.join(gt_dir, filename)
        
        if not os.path.exists(g_file):
            continue

        # Load data
        recon = np.squeeze(np.load(r_file))
        gt = np.squeeze(np.load(g_file))
        if recon.shape != gt.shape:
            continue

        # 1. Normalize both to 0-1 range to ensure fair comparison
        # (MRI intensity can vary between scans)
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-8)
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)

        # 2. Calculate Metrics
        p = psnr(gt_norm, recon_norm, data_range=1.0)
        s = ssim(gt_norm, recon_norm, data_range=1.0)
        gt_norm_fro_sq = np.sum(gt_norm ** 2)
        # NMSE = ||recon - gt||^2 / ||gt||^2; only when reference has enough energy
        min_denom = 1e-6 * gt_norm.size  # avoid explosion when gt is near-constant
        n = (np.linalg.norm(gt_norm - recon_norm) ** 2) / max(gt_norm_fro_sq, min_denom)

        psnr_list.append(p)
        ssim_list.append(s)
        nmse_list.append(n)

    if len(psnr_list) == 0:
        print("No pairs found. Ensure recon and GT have matching filenames.")
        print("  Recon dir:", recon_dir)
        print("  GT dir:  ", gt_dir)
        print("  To create GT from test set: python export_test_reference.py")
        return
    print(f"--- Analysis Results (Average over {len(psnr_list)} slices) ---")
    print(f"Average PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_list):.4f}")
    print(f"Average NMSE: {np.mean(nmse_list):.4f}")

def plot_comparison(gt, recon):
    diff = np.abs(gt - recon)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1); plt.imshow(gt, cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1, 3, 2); plt.imshow(recon, cmap='gray'); plt.title("Reconstruction")
    plt.subplot(1, 3, 3); plt.imshow(diff, cmap='hot'); plt.title("Error Map")
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze recon vs reference (PSNR/SSIM/NMSE)")
    parser.add_argument("--recon", default="./results/ixi_samples/test",
                        help="Directory of reconstruction .npy files")
    parser.add_argument("--gt", default="./data/ixi_test_reference",
                        help="Directory of ground-truth/reference .npy files (default: ./data/ixi_test_reference)")
    args = parser.parse_args()
    analyze_results(args.recon, args.gt)
