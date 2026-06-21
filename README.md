# AgroMind CV

A standalone tomato-leaf classifier built with TensorFlow/Keras and MobileNetV2. It trains on four PlantVillage classes and exposes predictions through a small FastAPI service.

## Recommended environment

This machine's RTX 3070 is available inside Ubuntu on WSL2. Modern TensorFlow GPU builds should run there; native Windows can still be used for CPU-only development. The commands below are PowerShell commands unless the prompt begins with `wsl`.

### 1. Set up TensorFlow with GPU

From this repository in PowerShell:

```powershell
wsl -d Ubuntu
cd /mnt/c/Users/monaz/Documents/GitHub/agromind-cv
bash scripts/setup_wsl.sh
source scripts/activate_wsl.sh
```

The final setup line prints TensorFlow's visible GPUs. It should include `GPU:0`. Use
`source scripts/activate_wsl.sh` in each new WSL shell; it activates the environment
and exposes pip-installed CUDA libraries to TensorFlow.

### 2. Download PlantVillage

The helper clones the public PlantVillage repository (roughly 1 GB):

```bash
bash scripts/download_plantvillage.sh
```

If downloading manually, place the dataset anywhere below `data/raw/PlantVillage`. These exact class directories must exist:

```text
Tomato___healthy
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
```

Raw data and trained artifacts are ignored by Git.

### 3. Prepare deterministic splits

```bash
python src/prepare_dataset.py
```

This validates images and creates a repeatable 70/15/15 split under `data/processed`.

### 4. Train and evaluate

```bash
python src/train.py
```

On an 8 GB RTX 3070, the default batch size of 32 is a sensible start. If GPU memory is tight, use `--batch-size 16`. Quick pipeline check:

```bash
python src/train.py --head-epochs 1 --fine-tune-epochs 1
```

Training writes:

- `models/agromind_mobilenetv2.keras`
- `models/class_names.json`
- `models/metadata.json`
- `models/classification_report.json`
- `models/confusion_matrix.csv` and `.png`
- `models/learning_curves.png`

Use test accuracy and macro F1 from `metadata.json`; also inspect the per-class report and confusion matrix. PlantVillage's clean backgrounds can inflate real-world performance, so test greenhouse/phone photos before presenting claims about field accuracy.

### 5. Predict one image

```bash
python src/predict.py --image path/to/tomato_leaf.jpg
```

Example response:

```json
{
  "prediction": "early_blight",
  "confidence": 0.8732,
  "severity": 0.7
}
```

`severity` is currently a fixed disease-risk proxy for the simulation, not measured lesion coverage.

### 6. Run the inference API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs`, or test from PowerShell:

```powershell
curl.exe -X POST http://localhost:8000/predict `
  -F "plantId=plant_042" `
  -F "file=@C:\path\to\tomato_leaf.jpg"
```

Endpoints:

- `GET /health` returns `ok` after model artifacts exist and `not_ready` before training.
- `POST /predict` accepts a multipart image and optional `plantId` (10 MB maximum).

## Development checks

Install test tools into the active environment, then run:

```bash
python -m pip install pytest httpx ruff
ruff check .
pytest -q
```

## Native Windows CPU fallback

Install 64-bit Python 3.11 or 3.12, then from PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```
