import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

MODEL_PATH = 'best_stroke_model.h5'
IMAGE_SIZE = (224,224)

model = load_model(MODEL_PATH)
print('Loaded model:', MODEL_PATH)
print('input_shape:', model.input_shape, 'output_shape:', model.output_shape)


def preprocess(img_path, mode=0):
    img = Image.open(img_path).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    arr = np.asarray(img).astype('float32')
    if mode == 0:
        # scale 0..1
        arr = arr / 255.0
    elif mode == 1:
        # scale to -1..1
        arr = (arr / 127.5) - 1.0
    elif mode == 2:
        # simple mean subtraction using ImageNet mean
        mean = np.array([123.68, 116.779, 103.939], dtype='float32')
        arr = arr - mean
    arr = np.expand_dims(arr, 0)
    return arr


def run_for_folder(folder, mode=0, limit=50):
    files = [os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(('.jpg','.png','.jpeg'))][:limit]
    counts = {'Stroke':0, 'Normal':0}
    examples = []
    for f in files:
        x = preprocess(f, mode=mode)
        p = model.predict(x)[0]
        # model outputs single sigmoid value
        prob = float(np.squeeze(p))
        label = 'Stroke' if prob >= 0.5 else 'Normal'
        counts[label] += 1
        if len(examples) < 5:
            examples.append((os.path.basename(f), prob, label))
    return counts, examples


if __name__ == '__main__':
    stroke_folder = 'dataset/Stroke'
    normal_folder = 'dataset/Normal'
    for mode in range(3):
        print('--- MODE', mode, '---')
        c_s, ex_s = run_for_folder(stroke_folder, mode=mode, limit=200)
        c_n, ex_n = run_for_folder(normal_folder, mode=mode, limit=200)
        print('Stroke folder ->', c_s)
        print('Sample:', ex_s)
        print('Normal folder ->', c_n)
        print('Sample:', ex_n)
        print('\n')
