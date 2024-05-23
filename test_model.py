from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model_path = 'models/model_bs32_img299.h5'
model = load_model(model_path)

test_data_dir = 'dataset/test'
batch_size = 32
img_height, img_width = 299, 299

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=test_generator.samples // batch_size)

print(f'Точность вашей модели на тестовых данных: {test_accuracy * 100:.2f}%')