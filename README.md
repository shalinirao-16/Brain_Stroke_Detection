# Brain_Stroke_Detection

This project is a Flask web app that uses a Keras `.h5` model (`best_stroke_model.h5`) to predict whether an uploaded image is `Stroke` or `Normal`.

Quick steps to get reproducible, correct predictions locally:

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Evaluate preprocessing modes to find the correct one used during training (run once):

```powershell
python evaluate_preprocessing.py
```

This script tests three common preprocessing strategies:
- Mode 0: scale pixels 0..1 (arr / 255.0)
- Mode 1: scale pixels -1..1 (arr / 127.5 - 1.0)
- Mode 2: ImageNet mean subtraction (arr - mean_rgb)

Check which mode gives the best recall/precision for the `Stroke` class. Note the mode number.

3. Configure the app to use the chosen preprocessing mode by setting environment variable `PREPROCESS_MODE` (0,1,or 2). Example for PowerShell:

```powershell
$env:PREPROCESS_MODE = "1"
python app.py
```

4. Open the app in a browser (http://127.0.0.1:5000), login with `admin/admin`, upload an image and view predictions.

Notes
- The app contains a debug log line which prints predicted probabilities and preprocessing mode for each uploaded image to help diagnose problems.
- If none of the modes provides acceptable results, the model likely needs retraining with consistent preprocessing and a verified dataset.

If you'd like, I can set a default mode for you after you run `evaluate_preprocessing.py` and paste the output here. If you want I can also add a route that runs the evaluation on the server and shows a small report in the UI.

Fine-tuning and helper scripts (tools/)
--------------------------------------
Helper scripts used for evaluation and fine-tuning are located in the `tools/` folder to keep the project root clean:

- `tools/evaluate_preprocessing.py` - compare preprocessing modes (0..1, -1..1, ImageNet mean-sub).
- `tools/compute_threshold.py` - search for the best classification threshold by F1 score.
- `tools/fine_tune.py` - light transfer-learning fine-tune script that saves `fine_tuned_model.h5`.

Run them from the project root (PowerShell):
```powershell
python tools/evaluate_preprocessing.py
python tools/compute_threshold.py
python tools/fine_tune.py
```

After producing a fine-tuned model, update `MODEL_PATHS` in `app.py` or replace `best_stroke_model.h5`.
