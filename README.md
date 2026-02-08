# SSDiffReconV2  — Retraining SSDiffRecon model on generated dataset

This repo is the [SSDiffRecon](https://github.com/icon-lab/SSDiffRecon) codebase (MICCAI 2023 — self-supervised MRI reconstruction with unrolled diffusion). 
Below is the gist of what was done on top of it: 
**Retraining on IXI SSDiffRecon generated images, running evaluation, and comparing reconstructions to a reference** (zero-filled, since no index→GT mapping is published).

---

## 1. **Generating dataset

---

## 2. Training
- **Training:** Retrained the single-coil IXI model with the same data setup; checkpoints and logs live in **`results/ixi_retrain_iter1_data_trial2`** (or the `--exp_name` you use).
- **NaN handling:** Training was hitting NaN gradients, which corrupted weights. The fix is in **`SSDiffRecon/diffusion/diffusion_tf/utils.py`**: gradients are sanitized (NaN/Inf replaced with zeros before `apply_gradients`) and optional checks keep the loss finite.
- **Logging:** Per-step NaN gradient prints were removed to avoid log flooding; only a critical message is printed when gradient norm is suspiciously small.

---

## 3. Evaluation

- **Command (from repo root):**
  ```bash
  cd SSDiffRecon && python run_ixi.py --eval --results_dir ./results
  ```
- **Checkpoint:** Uses the **latest** checkpoint in `results/<exp_name>/` unless you pass `--eval_checkpoint <STEP>` (e.g. `--eval_checkpoint 445831`).
- **Output:** Reconstructions are written as **`results/<exp_name>/test/im_0samples.npy`**, `im_1samples.npy`, … (one per test sample, 6480 total for IXI test).

---

## 4. Reference for Metrics (Zero-Filled)

The paper does not provide an index→ground-truth mapping for the test set, so **fully-sampled IXI GT** is not available by index here. Instead we use **zero-filled (ZF) reconstructions** as the reference so we can still run PSNR/SSIM/NMSE.

- **Export ZF reference** (same order as eval, so filenames align):
  ```bash
  python export_test_reference.py
  ```
  Default output: **`data/ixi_test_reference/`** (`im_0samples.npy`, …).  
  Options: `--out DIR`, `--max N`. See **`IXI_GT_SYNC.md`** if you later obtain a real GT list or index mapping.

---

## 5. Analysis (PSNR / SSIM / NMSE)

- **Command:**
  ```bash
  python run_analysis.py --recon ./results/ixi_retrain_iter1_data_trial2/test --gt ./data/ixi_test_reference
  ```
- **Defaults:** `--recon ./results/ixi_samples_iter1/test`, `--gt ./data/ixi_test_reference` (so you can omit both if using those paths).
- **Metrics:** PSNR, SSIM, NMSE (recon vs reference). NMSE uses a stable denominator to avoid division-by-zero on constant references.

**Note:** These numbers are **recon vs zero-filled**, not vs real GT. The paper reports recon vs GT, so values are not directly comparable; they still show how the model compares to ZF.

---

## 6. Plotting Reference / Reconstructions

- **`plot_reference_images.py`** — plot `.npy` images from a folder (e.g. ZF reference or eval outputs):
  ```bash
  python plot_reference_images.py [FOLDER] --indices 0,100,500 --save analysis/ref.png
  ```
  Default folder: `results/ixi_samples_iter1/test_reference`. Options: `--out`, `--dpi`, `--rows`.

---

## 7. File Overview (Added / Modified)

| Item | Description |
|------|-------------|
| **`export_test_reference.py`** | Export ZF images from test tfrecords → `im_*samples.npy`. |
| **`run_analysis.py`** | PSNR/SSIM/NMSE between a recon folder and a reference folder. |
| **`plot_reference_images.py`** | Plot selected `.npy` images (e.g. ZF or recon). |
| **`SSDiffRecon/diffusion/diffusion_tf/utils.py`** | Gradient sanitization and reduced NaN logging. |

---

## 8. Original Repo

For dataset setup, citation, and the original run commands, see **`SSDiffRecon/README.md`**.

