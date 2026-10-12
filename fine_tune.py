import os
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_MODEL_PATH = 'best_stroke_model.h5'
OUTPUT_MODEL_PATH = 'fine_tuned_model.h5'
IMAGE_SIZE = (224,224)
BATCH_SIZE = 16
EPOCHS = 5

# Load base model
base = load_model(BASE_MODEL_PATH)
print('Loaded base model:', BASE_MODEL_PATH)
# Remove top if necessary: if base already has output, we'll create a new head using its convolutional base
# Try to find a suitable layer to attach; often models have a GlobalAveragePooling2D before Dense

# Find last conv layer by searching backwards for a 4D output
conv_output = None
for layer in reversed(base.layers):
    out_shape = getattr(layer, 'output_shape', None)
    if out_shape and len(out_shape) == 4:
        conv_output = layer.output
        break

if conv_output is None:
    raise RuntimeError('Could not find convolutional base in provided model to fine-tune.')

x = GlobalAveragePooling2D()(conv_output)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base.input, outputs=output)

# Freeze original layers initially
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
print(model.summary())

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.15
)

train_gen = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)
val_gen = train_datagen.flow_from_directory(
    'dataset',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Train head
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

# Optionally unfreeze some top layers for fine-tuning
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

model.save(OUTPUT_MODEL_PATH)
print('Saved fine-tuned model to', OUTPUT_MODEL_PATH)
