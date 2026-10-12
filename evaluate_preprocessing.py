import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

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


def evaluate_for_mode(mode=0, limit_per_class=None):
    y_true = []
    y_pred = []
    for label_name, label_val, folder in [('Stroke',1,'dataset/Stroke'), ('Normal',0,'dataset/Normal')]:
        files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        if limit_per_class:
            files = files[:limit_per_class]
        for f in files:
            x = preprocess(f, mode=mode)
            p = model.predict(x)[0]
            prob = float(np.squeeze(p))
            pred = 1 if prob >= 0.5 else 0
            y_true.append(label_val)
            y_pred.append(pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {'accuracy':acc, 'precision':prec, 'recall':rec, 'confusion_matrix':cm}

if __name__ == '__main__':
    for mode in [0,1,2]:
        print('=== MODE', mode, '===')
        res = evaluate_for_mode(mode=mode, limit_per_class=200)
        print('Accuracy: {:.3f} Precision: {:.3f} Recall: {:.3f}'.format(res['accuracy'], res['precision'], res['recall']))
        print('Confusion matrix:\n', res['confusion_matrix'])
        print('\n')
