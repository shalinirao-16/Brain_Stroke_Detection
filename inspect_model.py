from tensorflow.keras.models import load_model
m = load_model('best_stroke_model.h5')
print('INPUT_SHAPE:', m.input_shape)
print('OUTPUT_SHAPE:', m.output_shape)
# Try to read activation from last layer config
last = m.layers[-1]
act = None
try:
    cfg = last.get_config()
    act = cfg.get('activation', None)
except Exception:
    act = None
print('LAST_LAYER_CLASS:', last.__class__.__name__)
print('LAST_ACTIVATION_CONFIG:', act)
import numpy as np
print('NUM_OUTPUTS:', np.prod(m.output_shape[1:]) if m.output_shape is not None else 'unknown')
