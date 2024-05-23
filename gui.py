import numpy as np
import tkinter as tk
from tkinter import filedialog
import tkinter.font as tkFont
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

try:
    model = load_model('models/model_bs32_img299.h5')
    print("Модель загружена")
except Exception as e:
    print("Модель не загружена")
    print(f"Ошибка: {e}")

def prepare_image(img_path, target_size=(299, 299)):
    img = image.load_img(img_path, target_size=target_size)
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0) / 255.0
    return img_tensor


def classify_image(img_path):
    img_tensor = prepare_image(img_path)
    predictions = model.predict(img_tensor)
    class_index = np.argmax(predictions[0])
    return class_index

def load_class_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        class_names = file.read().splitlines()
    return class_names

class_names_file = 'class_names.txt'
class_names = load_class_names(class_names_file)

def select_image():
    filepath = filedialog.askopenfilename(
        title="Выбрать картинку",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
    )
    if not filepath:
        return
    class_index = classify_image(filepath)
    fruit_name = class_names[class_index]
    result_label.config(text=f"Да это же {fruit_name}")
    display_image(filepath)


def display_image(img_path):
    img = Image.open(img_path)
    img = img.resize((600, 600), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img

app = tk.Tk()
app.title("WHAT THE FRUIT")
app.geometry("800x600") #размер
app.configure(bg='#201F1E') #цвет фона

reg_font = tkFont.Font(family="Montserrat", size=12, weight="normal")
bold_font = tkFont.Font(family="Montserrat", size=24, weight="bold")

open_button = tk.Button(app, text="Выбрать картинку", command=select_image, font=reg_font)
open_button.pack(pady=40)

result_label = tk.Label(app, text="Покажи свой фрукт", font=bold_font, bg='#201F1E', fg='#D93433')
result_label.pack(pady=40)
image_label = tk.Label(app, bg='#201F1E')
image_label.pack(pady=40)

app.mainloop()
