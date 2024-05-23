import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tkinter import Tk, filedialog, messagebox

model_path = 'models/model_bs32_img299.h5'
model = load_model(model_path)
def load_class_names(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        class_names = [line.strip() for line in file.readlines()]
    return class_names

class_names = load_class_names('class_names.txt')

def prepare_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    return img_tensor

def classify_and_display_image(img_path):
    img_tensor = prepare_image(img_path)
    predictions = model.predict(img_tensor)
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]
    top_5_probs = predictions[0][top_5_indices]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Выбранное изображение')

    plt.subplot(1, 2, 2)
    plt.barh(range(5), top_5_probs, color='blue')
    plt.yticks(range(5), [class_names[i] for i in top_5_indices])
    plt.xlabel('Вероятность')
    plt.title('Топ-5 предсказаний')

    plt.tight_layout()
    plt.show()

def main_loop():
    while True:
        Tk().withdraw()
        img_path = filedialog.askopenfilename()

        if not img_path:
            print("Изображение не выбрано или процесс завершен пользователем.")
            if messagebox.askyesno("Завершение", "Вы хотите завершить работу?"):
                break
            else:
                continue

        classify_and_display_image(img_path)

main_loop()
