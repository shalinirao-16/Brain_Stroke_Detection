import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score

MODEL_PATH = 'best_stroke_model.h5'
IMAGE_SIZE = (224,224)

model = load_model(MODEL_PATH)
print('Loaded model:', MODEL_PATH)


def preprocess(img_path, mode=0):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img).astype('float32')
    if mode == 0:
        arr = arr / 255.0
    elif mode == 1:
        arr = (arr / 127.5) - 1.0
    elif mode == 2:
        mean = np.array([123.68, 116.779, 103.939], dtype='float32')
        arr = arr - mean
    arr = np.expand_dims(arr, 0)
    return arr


def gather_probs_labels(mode=0, limit_per_class=None):
    probs = []
    labels = []
    for folder, label in [('dataset/Stroke',1), ('dataset/Normal',0)]:
        files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            x = preprocess(f, mode=mode)
            p = float(np.squeeze(model.predict(x)[0]))
            probs.append(p)
            labels.append(label)
    return np.array(probs), np.array(labels)

if __name__ == '__main__':
    best_overall = (0.0, 0.5, 0)  # (f1, threshold, mode)
    for mode in [0,1,2]:
        probs, labels = gather_probs_labels(mode=mode)
        if len(probs) == 0:
            print('No data for mode', mode)
            continue
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = 0.0
        best_t = 0.5
        for t in thresholds:
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        print(f'Mode {mode}: best F1={best_f1:.3f} at thresh={best_t:.2f}')
        if best_f1 > best_overall[0]:
            best_overall = (best_f1, best_t, mode)
    print('Overall best:', best_overall)
