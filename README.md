# Dental Project (Django + Multi-View Dental Index Model)

This project hosts a Django web app plus a multi-view deep learning model that predicts three dental indices (MGI 0–4, OHI 0–3, GEI 0–2) from three photographs (frontal, left lateral, right lateral). A Jupyter notebook provides the full training loop.

## Project Structure
- `manage.py` / `dental_project/` — Django project
- `analysis/` — App with upload/processing/results views
- `ml/Train_Model.ipynb` — Training notebook (GPU-first, CPU fallback)
- `ml/model.py` — Multi-view DINOv2 backbone with three classification heads
- `ml/dataset.py` — CSV + image loader for Thesis_Data
- `ml/transforms.py` — DINOv2-compatible image transforms with heavy augmentation
- `ml/inference.py` — Inference module with TTA and ensemble support
- `ml/losses.py` — CORAL ordinal loss (alternative loss function)
- `Thesis_Data/` — CSV + photographs (Frontal, Left_Lateral, Right_Lateral)

## Environment & Kernel
- Python 3.12+, venv located at `venv/` (or `.venv/`), registered as Jupyter kernel **"Dental Project (venv)"** (`dental_venv`).
- Activate venv (PowerShell): `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass ; .\venv\Scripts\activate`

## GPU Setup (Required for Training)

The model uses DINOv2 (ViT-B/14, 86M params) which is too large for practical CPU training. An NVIDIA GPU is required.

**Step 1: Verify your GPU**
```bash
nvidia-smi
```

**Step 2: Install CUDA-enabled PyTorch** (replaces CPU-only build)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

**Step 3: Verify CUDA works**
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

> **Note:** No WSL needed — PyTorch CUDA works natively on Windows with the NVIDIA driver.

## Running the Web App
1) Activate venv (see above).
2) Install deps (already installed, but if needed): `pip install -r requirements.txt`
3) Apply migrations: `python manage.py migrate`
4) Run server: `python manage.py runserver 8000`
5) Open http://127.0.0.1:8000/ — upload the three images + patient name, submit, wait for processing, view scores and Grad-CAM overlays.

## Training the Model (Notebook)
- Open `ml/Train_Model.ipynb` and select kernel **"Dental Project (venv)"**.
- Key config in the notebook `CONFIG` dict:
  - `image_size`: default 336 (= 14×24, DINOv2-compatible). Use 518 for maximum quality if VRAM allows.
  - `n_folds`: default 3 (cross-validation folds).
  - `warmup_epochs`: default 5 (head-only training with frozen backbone).
  - `finetune_epochs`: default 25 (with last 4 transformer blocks unfrozen).
  - `batch_size`: default 8 (fits 12 GB VRAM with 336px images).
  - `patience`: default 8 (early stopping tolerance).
  - `unfreeze_blocks`: default 4 (last 4 of 12 DINOv2 transformer blocks).
- Training loop uses GPU with mixed precision (AMP); falls back to CPU if unavailable (not recommended).
- Checkpoints and history save to `ml/checkpoints/` (`best_model.pth`, `fold_*_head.pth`, `training_history.json`).

## Backbone: DINOv2 (ViT-B/14)
DINOv2 is a self-supervised Vision Transformer pre-trained on 142M images by Meta AI. It produces rich visual features that transfer effectively even with very small datasets like ours (201 samples).

The training strategy:
1. **Phase 1 (Warmup):** Freeze entire backbone, train only the classifier head.
2. **Phase 2 (Fine-tuning):** Unfreeze last 4 transformer blocks with 10-50× lower learning rate than the head.

## Data Notes
- CSV: `Thesis_Data/Thesis_Results.csv` contains labels; parser expects patterns `MGI-x`, `OHI-y`, `GEI-z`.
- Images: place patient photos in `Thesis_Data/Thesis_Photographs/Frontal`, `Left_Lateral`, `Right_Lateral` with filenames `F<number>`, `L<number>`, `R<number>` (case-insensitive extensions .JPG/.jpg/.png/.jpeg).
- Samples are included only when all three views exist.

## Inference in the Web App
- The Django app loads the best checkpoint (`ml/checkpoints/best_model.pth`) during inference.
- If fold checkpoints exist (`fold_*_head.pth`), it uses ensemble averaging for better predictions.
- If no checkpoint exists, the app will not return scores; train first.

## Troubleshooting
- **Training appears stuck / infinite loop:** Most likely the CPU-only PyTorch is installed. Install CUDA PyTorch (see GPU Setup above).
- **Out-of-memory on GPU:** Reduce `batch_size` (e.g., 4 or 2), lower `image_size` (e.g., 224).
- **Slow training:** Ensure GPU is being used (`Device: cuda` printed at start). Lower `image_size` if needed.
- **Class imbalance:** Handled via inverse-frequency class weights in the loss function.
- **Windows multiprocessing errors:** `num_workers` is set to 0 by default to avoid Windows fork issues.

## References
- DINOv2 paper: https://arxiv.org/abs/2304.07193 (self-supervised ViT pre-training)
- timm model zoo: https://huggingface.co/timm (DINOv2 variants available)
