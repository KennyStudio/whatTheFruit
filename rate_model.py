import tensorflow as tf
from tensorflow.keras.models import load_model

# Путь к файлу с сохраненной моделью
model_path = 'путь'  # Замените на путь к вашей модели

# Загрузка модели
model = load_model(model_path)

# Вывод информации о модели
model.summary()

# Вывод дополнительных характеристик модели
num_layers = len(model.layers)
total_params = model.count_params()
trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])

print(f"\nОбщее количество слоев: {num_layers}")
print(f"Общее количество параметров: {total_params}")
print(f"Количество обучаемых параметров: {trainable_params}")
print(f"Количество необучаемых параметров: {non_trainable_params}")
