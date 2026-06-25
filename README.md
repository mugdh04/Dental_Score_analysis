# DentAI Oral Health Prediction System

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Django](https://img.shields.io/badge/Django-6.0-success)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![CUDA](https://img.shields.io/badge/GPU-CUDA%20Required-green)
![Status](https://img.shields.io/badge/Model-Single%20Active%20Checkpoint-informational)

AI-assisted dental analysis platform that predicts oral health indices from three intraoral photos:

- MGI (Modified Gingival Index)
- OHI (Oral Hygiene Index)
- GEI (Gingival Enlargement Index)
- PI (Plaque Index, rule-based)

## System Flow

```mermaid
flowchart LR
    A[Upload 3 Photos\nFrontal + Left + Right] --> B[Background Processing]
    B --> C[Model Inference\nMGI/OHI/GEI]
    B --> D[Rule-Based PI]
    C --> E[Save Scores + Confidence]
    D --> E
    E --> F[Results Page + Grad-CAM]
    F --> G[Dentist Review Workflow]
```

## Architecture Snapshot

```mermaid
graph TD
    subgraph Web
      V[analysis/views.py]
      T[templates/analysis]
    end

    subgraph Runtime ML
      I[inference/predict.py]
      C[models/checkpoints/*]
    end

    subgraph New Training Stack
      TD[training/dataset.py]
      TM[training/model.py]
      TL[training/loss.py]
      TT[training/trainer.py]
      TN[training/train_model.ipynb]
      IP[inference/predict.py]
    end

    V --> I --> C
    TN --> TD
    TN --> TM
    TN --> TL
    TN --> TT
    TN --> IP
```

## Active Model Policy

Current runtime uses the checkpoints under [models/checkpoints](models/checkpoints).

Keep the runtime checkpoint bundle small and avoid committing training artifacts or private data.

## Quick Start

### 1) Environment

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_model.txt
```

### 2) Run App

```powershell
python manage.py migrate
python manage.py runserver 8000
```

Open: http://127.0.0.1:8000/

### 3) Train (GPU Required)

Use either notebook (both are kept):

- [ml/Train_Model.ipynb](ml/Train_Model.ipynb)
- [training/train_model.ipynb](training/train_model.ipynb)

The training scripts expect a dataset path provided through `DATA_DIR`. The public repo no longer bundles `Thesis_Data/`.

## Data Contract

Provide your own dataset root with this layout:

- Frontal: F{Sl No}
- Left: L{Sl No}
- Right: R{Sl No}

Only complete triplets are used for training.

## Feature Matrix

| Capability | Status | Location |
|---|---|---|
| 3-view upload and async processing | Active | [analysis/views.py](analysis/views.py) |
| MGI/OHI/GEI inference | Active | [inference/predict.py](inference/predict.py) |
| PI persistence and display | Active | [analysis/models.py](analysis/models.py), [templates/analysis/results.html](templates/analysis/results.html) |
| Startup model warm-up | Active | [analysis/apps.py](analysis/apps.py) |
| New patch-based predictor module | Ready | [inference/predict.py](inference/predict.py) |
| New training pipeline modules | Ready | [training](training) |

## Interactive Ops Notes

<details>
<summary>Model Path Resolution</summary>

Runtime resolution order:

1. MODEL_PATH environment variable
2. settings MODEL_PATH
3. Fallback to [models/checkpoints](models/checkpoints)

Configured in [dental_project/settings.py](dental_project/settings.py) and used by the runtime inference loader.

</details>

<details>
<summary>GPU Validation Checklist</summary>

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

If CUDA is False, training notebook will fail fast by design.

</details>

<details>
<summary>Troubleshooting</summary>

- Model not loading:
  - Verify your configured checkpoint path exists and `MODEL_PATH` points to it.
- Slow inference:
  - Confirm warm-up is enabled in [dental_project/settings.py](dental_project/settings.py).
- Training OOM:
  - Reduce batch size in [training/train_model.ipynb](training/train_model.ipynb).

</details>

## Repository Layout

```text
analysis/         Django app (views, models, forms, templates)
dental_project/   Django project settings and routes
training/         New modular training stack + notebook
inference/        New predictor + plaque algorithm
models/           Runtime checkpoints and calibration assets
templates/        HTML templates for the web app
```
