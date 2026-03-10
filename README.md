# Dental Project (Django + Multi-View Dental Index Model)

This project hosts a Django web app plus a multi-view deep learning model that predicts three dental indices (MGI 0–4, OHI 0–3, GEI 0–2) from three photographs (frontal, left lateral, right lateral). A Jupyter notebook provides the full training loop.

## Project Structure
- `manage.py` / `dental_project/` — Django project
- `analysis/` — App with upload/processing/results views
- `ml/Train_Model.ipynb` — Training notebook (GPU-first, CPU fallback)
- `ml/model.py` — Multi-view EfficientNet backbone with three ordinal heads
- `ml/dataset.py` — CSV + image loader for Thesis_Data
- `ml/losses.py` — CORAL ordinal loss
- `Thesis_Data/` — CSV + photographs (Frontal, Left_Lateral, Right_Lateral)

## Environment & Kernel
- Python 3.12, venv located at `venv/` (or `.venv/`), registered as Jupyter kernel **“Dental Project (venv)”** (`dental_venv`).
- Activate venv (PowerShell): `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass ; .\venv\Scripts\activate`

## Running the Web App
1) Activate venv (see above).
2) Install deps (already installed, but if needed): `pip install -r requirements.txt`
3) Apply migrations: `python manage.py migrate`
4) Run server: `python manage.py runserver 8000`
5) Open http://127.0.0.1:8000/ — upload the three images + patient name, submit, wait for processing, view scores and Grad-CAM overlays.

## Training the Model (Notebook)
- Open `ml/Train_Model.ipynb` and select kernel **“Dental Project (venv)”**.
- Key config in the notebook `CONFIG` dict:
  - `backbone_name`: default `efficientnet_b3` (strong, moderate size). You can switch to `tf_efficientnetv2_s` for a faster/more accurate backbone if GPU VRAM allows.
  - `image_size`: default 300 (match backbone). For EfficientNetV2-S, use 300–384.
  - `epochs`, `patience`, `unfreeze_epoch`, `backbone_lr_mult` — control schedule.
  - `resume_from`: set a checkpoint path to continue training.
- Training loop uses GPU if available (mixed precision via AMP); falls back to CPU automatically.
- Checkpoints and history save to `ml/checkpoints/` (`best_model.pth`, `final_model.pth`, `training_history.json`).

## Switching to EfficientNetV2
The current backbone is EfficientNet-B3 (good accuracy/size balance). EfficientNetV2 (paper: arXiv:2104.00298) reports higher ImageNet accuracy with faster training. To try it:
1) In `ml/Train_Model.ipynb`, set `backbone_name = 'tf_efficientnetv2_s'` (or `m`) in CONFIG.
2) Adjust `image_size` to 300–384 for `v2_s`; ~420–480 for `v2_m` if VRAM allows.
3) Keep `batch_size` small (e.g., 2–4) on modest GPUs; increase if memory permits.
4) Train as usual; checkpoints will remain in `ml/checkpoints/`.

## Data Notes
- CSV: `Thesis_Data/Thesis_Results.csv` contains labels; parser expects patterns `MGI-x`, `OHI-y`, `GEI-z`.
- Images: place patient photos in `Thesis_Data/Thesis_Photographs/Frontal`, `Left_Lateral`, `Right_Lateral` with filenames `F<number>`, `L<number>`, `R<number>` (case-insensitive extensions .JPG/.jpg/.png/.jpeg).
- Samples are included only when all three views exist.

## Inference in the Web App
- The Django app loads the best checkpoint (`ml/checkpoints/best_model.pth`) during inference.
- If no checkpoint exists, the app will not return scores; train first or place a trained checkpoint at that path.

## Troubleshooting
- Slow training: lower `image_size` or switch backbone to `efficientnet_b0/b1`.
- Out-of-memory on GPU: reduce `batch_size`, lower `image_size`, or use `efficientnetv2_s` with smaller size.
- Class imbalance: handled via class weights + weighted sampler (already in notebook).

## References
- EfficientNetV2 paper: https://arxiv.org/abs/2104.00298 (faster training, higher accuracy)
- timm model zoo: https://huggingface.co/timm (EfficientNetV2 variants available)
