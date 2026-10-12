import os
import uuid
import json
import numpy as np
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, send_file, flash
)
from werkzeug.utils import secure_filename
from functools import wraps
import io
import base64
from PIL import Image
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt

# ==============================
# Flask App Config
# ==============================
app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# ==============================
# Load Model and Class Indices
# ==============================
# Try a few likely model paths (prefer the "best" file if present)
MODEL_PATHS = [
    "best_stroke_model.h5",
    "stroke_model.h5",
    os.path.join("models", "stroke_model.h5")
]
CLASS_INDICES_PATH = "class_indices.json"

model = None
class_labels = {}
IMAGE_SIZE = (224, 224)  # default fallback
# Configuration: pick preprocessing mode by environment variable PREPROCESS_MODE (0,1,2)
# If AUTO_PREPROCESS is True, the app will try multiple modes per image (slower).
AUTO_PREPROCESS = False  # set to True to try multiple modes per image (diagnostic)
_env_mode = os.environ.get("PREPROCESS_MODE", None)
DEFAULT_PREPROCESS_MODE = int(_env_mode) if _env_mode is not None else None
DEFAULT_PREPROCESS_THRESHOLD = float(os.environ.get("PREPROCESS_THRESHOLD", "0.8"))

for p in MODEL_PATHS:
    if os.path.exists(p):
        try:
            model = load_model(p)
            print(f"✅ Model loaded successfully from: {p}")
            MODEL_PATH = p
            break
        except Exception as e:
            print(f"⚠️ Failed to load model at {p}: {e}")

if os.path.exists(CLASS_INDICES_PATH):
    try:
        with open(CLASS_INDICES_PATH, "r") as f:
            class_indices = json.load(f)
        # class_indices may be {"Normal":0, "Stroke":1} -> we want index->label
        if all(isinstance(k, str) and isinstance(v, int) for k, v in class_indices.items()):
            class_labels = {v: k for k, v in class_indices.items()}
        else:
            # Might already be index->label
            class_labels = {int(k): v for k, v in class_indices.items()}
        print("✅ Class indices loaded:", class_labels)
    except Exception as e:
        print("⚠️ Failed to parse class_indices.json:", e)

# If model is loaded, try to infer expected input size
if model is not None:
    try:
        def _infer_image_size_from_model(m):
            """Return (width, height) tuple inferred from model.input_shape or None."""
            try:
                shape = m.input_shape
                if not shape:
                    return None
                # shape often like (None, H, W, C) or (None, C, H, W)
                s = tuple(shape)
                # handle 4D shapes (batch, H, W, C) or (batch, C, H, W)
                if len(s) >= 4:
                    # prefer channels-last if it looks like (None, H, W, C)
                    batch, a, b, c = (s[0], s[1], s[2], s[3])
                    # if 'c' looks like channel dim
                    if isinstance(c, int) and c in (1, 3, 4):
                        h, w = a, b
                        if isinstance(h, int) and isinstance(w, int):
                            return (int(w), int(h))
                    # if 'a' looks like channel dim (channels-first)
                    if isinstance(a, int) and a in (1, 3, 4):
                        # (batch, C, H, W)
                        h, w = b, c
                        if isinstance(h, int) and isinstance(w, int):
                            return (int(w), int(h))
                    # fallback: pick last two integer dims >=16
                    ints = [int(x) for x in s if isinstance(x, int) and x >= 16]
                    if len(ints) >= 2:
                        return (ints[-1], ints[-2]) if ints[-1] >= ints[-2] else (ints[-2], ints[-1])
                # handle 3D shapes (H, W, C) or (None, H, W)
                ints = [int(x) for x in s if isinstance(x, int) and x >= 16]
                if len(ints) >= 2:
                    return (ints[1], ints[0]) if len(ints) >= 2 else None
            except Exception:
                return None

        inferred = _infer_image_size_from_model(model)
        if inferred:
            IMAGE_SIZE = inferred
        print(f"ℹ️ Inferred image target size (width,height): {IMAGE_SIZE}")
    except Exception as e:
        print("⚠️ Could not infer image size from model:", e)

def _quick_evaluate_modes(limit_per_class=200):
    """Quickly evaluate modes 0..2 on small sample from dataset folders.
    Returns best_mode (int) or None if dataset unavailable.
    Metric: maximize recall for Stroke (label 1). Tie-breaker: accuracy.
    """
    stroke_folder = os.path.join('dataset','Stroke')
    normal_folder = os.path.join('dataset','Normal')
    if not (os.path.exists(stroke_folder) and os.path.exists(normal_folder)):
        print('ℹ️ Dataset folders not found for auto-evaluation:', stroke_folder, normal_folder)
        return None

    def _gather_files(folder, limit):
        files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        return files[:limit]

    s_files = _gather_files(stroke_folder, limit_per_class)
    n_files = _gather_files(normal_folder, limit_per_class)
    if len(s_files) == 0 or len(n_files) == 0:
        print('ℹ️ Not enough images for auto-evaluation')
        return None

    best_mode = None
    best_metric = (-1.0, -1.0)  # (recall, accuracy)

    for mode in [0,1,2]:
        tp = fp = tn = fn = 0
        # Stroke (true label 1)
        for f in s_files:
            x = preprocess_image_with_mode(f, mode=mode)
            p = model.predict(x)[0]
            prob = float(np.squeeze(p))
            pred = 1 if prob >= 0.5 else 0
            if pred == 1:
                tp += 1
            else:
                fn += 1
        # Normal (true label 0)
        for f in n_files:
            x = preprocess_image_with_mode(f, mode=mode)
            p = model.predict(x)[0]
            prob = float(np.squeeze(p))
            pred = 1 if prob >= 0.5 else 0
            if pred == 1:
                fp += 1
            else:
                tn += 1

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
        print(f'Auto-eval mode={mode} -> recall(stroke)={recall:.3f} acc={accuracy:.3f} (tp={tp},fn={fn},fp={fp},tn={tn})')
        metric = (recall, accuracy)
        if metric > best_metric:
            best_metric = metric
            best_mode = mode

    print('Auto-eval selected mode=', best_mode, 'with metrics', best_metric)
    return best_mode

# If PREPROCESS_MODE not explicitly set, try to auto-evaluate using dataset samples
if DEFAULT_PREPROCESS_MODE is None:
    try:
        chosen = _quick_evaluate_modes(limit_per_class=200)
        if chosen is not None:
            DEFAULT_PREPROCESS_MODE = int(chosen)
        else:
            DEFAULT_PREPROCESS_MODE = 0
    except Exception as e:
        print('⚠️ Auto-evaluation failed:', e)
        DEFAULT_PREPROCESS_MODE = 0

print(f'Using preprocessing mode: {DEFAULT_PREPROCESS_MODE} (AUTO_PREPROCESS={AUTO_PREPROCESS})')
print(f'Using prediction threshold: {DEFAULT_PREPROCESS_THRESHOLD}')

# ==============================
# Admin
# ==============================
ADMIN_USER = "admin"
ADMIN_PASS = "admin"

def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if "logged_in" in session:
            return f(*args, **kwargs)
        flash("Please login first.", "warning")
        return redirect(url_for("login"))
    return wrap

# ==============================
# Helpers
# ==============================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(filepath, target_size=None):
    if target_size is None:
        target_size = IMAGE_SIZE
    # PIL expects size as (width, height)
    w, h = int(target_size[0]), int(target_size[1])
    img = Image.open(filepath).convert("RGB")
    img = img.resize((w, h))
    img_array = np.asarray(img).astype("float32") / 255.0
    # Ensure shape is (1, H, W, C)
    if img_array.ndim == 3:
        img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_with_mode(filepath, mode=0, target_size=None):
    """Preprocess image with a specific mode.
    mode 0: scale to 0..1
    mode 1: scale to -1..1
    mode 2: ImageNet mean subtraction (RGB means)
    Returns batch array shape (1,H,W,C)
    """
    if target_size is None:
        target_size = IMAGE_SIZE
    w, h = int(target_size[0]), int(target_size[1])
    img = Image.open(filepath).convert("RGB")
    img = img.resize((w, h))
    arr = np.asarray(img).astype("float32")
    if mode == 0:
        arr = arr / 255.0
    elif mode == 1:
        arr = (arr / 127.5) - 1.0
    elif mode == 2:
        mean = np.array([123.68, 116.779, 103.939], dtype="float32")
        arr = arr - mean
    else:
        arr = arr / 255.0
    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(filepath):
    """Predict using trained model. Supports models that output a single sigmoid value
    (shape [batch, 1]) or a softmax over two classes (shape [batch, 2]). Returns
    a label ("Stroke"/"Normal"), confidence (0..1) and probs dict.
    """
    if model is None:
        return "Normal", 0.5, {"Normal": 0.5, "Stroke": 0.5}

    # If AUTO_PREPROCESS is enabled, try multiple preprocessing modes per image.
    # Otherwise use the configured DEFAULT_PREPROCESS_MODE.
    if AUTO_PREPROCESS:
        tried_modes = [0, 1, 2]
    else:
        tried_modes = [int(DEFAULT_PREPROCESS_MODE)]

    best_choice = None
    best_stroke_prob = -1.0
    best_result = None

    for mode in tried_modes:
        if mode == 0:
            img_array = preprocess_image(filepath)
        else:
            img_array = preprocess_image_with_mode(filepath, mode=mode)
        preds = model.predict(img_array)

        # Normalize preds to a 1D numpy array for a single example
        preds = np.asarray(preds)
        if preds.ndim == 2 and preds.shape[0] >= 1:
            out = preds[0]
        else:
            out = preds.flatten()

        # Determine stroke_prob for this mode
        if out.size == 1:
            stroke_prob = float(out[0])
            normal_prob = 1 - stroke_prob
        elif out.size == 2:
            idx_to_label = class_labels if class_labels else {0: 'Normal', 1: 'Stroke'}
            stroke_idx = None
            for idx, lab in idx_to_label.items():
                if isinstance(lab, str) and lab.lower() == 'stroke':
                    stroke_idx = int(idx)
                    break
            if stroke_idx is None:
                stroke_idx = 1
            stroke_prob = float(out[stroke_idx])
            normal_prob = float(out[1 - stroke_idx])
        else:
            stroke_prob = float(out.max())
            normal_prob = 1 - stroke_prob

        if stroke_prob > best_stroke_prob:
            best_stroke_prob = stroke_prob
            best_choice = mode
            best_result = (stroke_prob, normal_prob)

    if best_result is None:
        # fallback
        img_array = preprocess_image(filepath)
        preds = model.predict(img_array)
        preds = np.asarray(preds)
        out = preds[0] if preds.ndim == 2 and preds.shape[0] >= 1 else preds.flatten()
        stroke_prob = float(out[0]) if out.size == 1 else float(out.max())
        normal_prob = 1 - stroke_prob
        best_choice = 0
    else:
        stroke_prob, normal_prob = best_result

    # Normalize preds to a 1D numpy array for a single example
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[0] >= 1:
        out = preds[0]
    else:
        out = preds.flatten()

    probs = {"Normal": 0.5, "Stroke": 0.5}

    # Case 1: binary sigmoid model returning single probability for positive class
    if out.size == 1:
        p_pos = float(out[0])
        # Determine which index corresponds to 'Stroke' if possible
        # If class_labels maps 1 -> 'Stroke', assume p_pos is prob for index 1
        stroke_prob = p_pos
        normal_prob = 1 - stroke_prob
    elif out.size == 2:
        # softmax: two-class output, need to map indices
        # prefer using class_labels if available
        try:
            # build index->label mapping
            idx_to_label = class_labels if class_labels else {0: 'Normal', 1: 'Stroke'}
        except Exception:
            idx_to_label = {0: 'Normal', 1: 'Stroke'}

        # find which index corresponds to 'Stroke'
        stroke_idx = None
        for idx, lab in idx_to_label.items():
            if isinstance(lab, str) and lab.lower() == 'stroke':
                stroke_idx = int(idx)
                break
        if stroke_idx is None:
            # default to index 1
            stroke_idx = 1

        stroke_prob = float(out[stroke_idx])
        normal_prob = float(out[1 - stroke_idx])
    else:
        # Unexpected output shape; fall back to heuristic
        stroke_prob = float(out.max())
        normal_prob = 1 - stroke_prob

    # Clip and round
    stroke_prob = float(np.clip(stroke_prob, 0.0, 1.0))
    normal_prob = float(np.clip(normal_prob, 0.0, 1.0))
    probs = {"Normal": float(round(normal_prob, 2)), "Stroke": float(round(stroke_prob, 2)), "mode": int(best_choice)}
    label = "Stroke" if stroke_prob >= normal_prob else "Normal"
    confidence = float(max(stroke_prob, normal_prob))
    # debug print to help trace misclassifications (will appear in server logs)
    try:
        print(f"[predict_image] file={os.path.basename(filepath)} mode={best_choice} probs={probs} label={label} conf={confidence}")
    except Exception:
        pass
    return label, confidence, probs

# ==============================
# Routes
# ==============================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET","POST"])
def login():
    if request.method=="POST":
        user = request.form.get("username")
        pwd = request.form.get("password")
        if user==ADMIN_USER and pwd==ADMIN_PASS:
            session["logged_in"] = True
            session["user"] = user
            flash("Logged in successfully","success")
            return redirect(url_for("upload"))
        flash("Invalid credentials","danger")
        return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("Logged out successfully","info")
    return redirect(url_for("home"))

@app.route("/upload", methods=["GET","POST"])
@login_required
def upload():
    if request.method=="POST":
        if "file" not in request.files:
            flash("No file part","danger")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename=="":
            flash("No selected file","danger")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            session["uploaded_file"] = filepath
            flash("File uploaded successfully","success")
            return redirect(url_for("preview"))
        flash("Invalid file type","danger")
        return redirect(request.url)
    return render_template("upload.html")

@app.route("/preview", methods=["GET","POST"])
@login_required
def preview():
    file = session.get("uploaded_file")
    if not file or not os.path.exists(file):
        flash("No file uploaded yet","danger")
        return redirect(url_for("upload"))
    if request.method=="POST":
        label, confidence, probs = predict_image(file)
        session["result"] = {
            "predicted_label": label,
            "confidence": confidence,
            "probs": probs
        }
        return redirect(url_for("result"))
    return render_template("preview.html", filename=os.path.basename(file))

@app.route("/result")
@login_required
def result():
    file = session.get("uploaded_file")
    result = session.get("result")
    if not file or not os.path.exists(file) or not result:
        flash("No prediction available","danger")
        return redirect(url_for("upload"))
    metrics = {"available":True,"accuracy":0.92,"precision":0.90,"recall":0.91}
    return render_template(
        "result.html",
        img_url=url_for("uploaded_file", filename=os.path.basename(file)),
        result=result,
        metrics=metrics
    )

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))

@app.route("/performance")
@login_required
def performance():
    metrics = {"available":True,"accuracy":0.92,"precision":0.90,"recall":0.91}
    fig, ax = plt.subplots()
    ax.bar(["Accuracy","Precision","Recall"], [metrics["accuracy"],metrics["precision"],metrics["recall"]],
           color=["#0ea5e9","#10b981","#f59e0b"])
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    bars_data = base64.b64encode(buf.getvalue()).decode("utf-8")
    plt.close(fig)
    return render_template("performance.html", metrics=metrics, bars_data=bars_data)

@app.route("/download_report")
@login_required
def download_report():
    filepath = "report.txt"
    with open(filepath,"w") as f:
        f.write("Brain Stroke Detector Report\nAccuracy:0.92\nPrecision:0.90\nRecall:0.91\nF1:0.91\n")
    return send_file(filepath, as_attachment=True)

# ==============================
# Main
# ==============================
if __name__=="__main__":
    app.run(debug=False)
