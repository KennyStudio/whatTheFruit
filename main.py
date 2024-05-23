import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
import tensorflow
import datetime

tensorflow.config.optimizer.set_jit(False)

print("Доступно GPU: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
if tensorflow.test.is_built_with_cuda():
    print("CUDA доступен и будет использован при обучении модели")
else:
    print('CUDA недоступна, останавливаем программу...')
    sys.exit(1)


gpus = tensorflow.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tensorflow.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


base_dir = os.getcwd()
data_dir = os.path.join(base_dir, 'dataset')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'val')
model_dir = os.path.join(base_dir, 'models')
logs_dir = os.path.join(base_dir, 'logs')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(logs_dir, exist_ok=True)

batch_size = 32
img_size = 299

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    brightness_range=[0.8, 1.2])

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical')

base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(30, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
num_layers = len(model.layers)
print(f"Количество слоев в модели: {num_layers}")

model.compile(optimizer=Adam(learning_rate=0.001),  loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = os.path.join(logs_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint = ModelCheckpoint(
    os.path.join(model_dir, str(batch_size)+'_'+str(img_size)+'_'+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
    save_best_only=True,
    monitor='val_loss',
    mode='min')

history = model.fit(
    train_generator,
    steps_per_epoch=np.ceil(21000 / batch_size),
    epochs=25,
    validation_data=validation_generator,
    validation_steps=np.ceil(6000 / batch_size),
    callbacks=[checkpoint, tensorboard_callback])

for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=np.ceil(21000 / batch_size),
    epochs=15,
    validation_data=validation_generator,
    validation_steps=np.ceil(6000 / batch_size),
    callbacks=[checkpoint, tensorboard_callback])