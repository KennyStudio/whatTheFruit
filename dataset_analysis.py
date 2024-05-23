import os
import matplotlib.pyplot as plt

data_dir = 'dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

def count_images_in_directory(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))
    return class_counts

train_counts = count_images_in_directory(train_dir)
val_counts = count_images_in_directory(val_dir)
test_counts = count_images_in_directory(test_dir)
print(train_counts)
print(val_counts)
print(test_counts)

classes = sorted(set(train_counts.keys()) | set(val_counts.keys()) | set(test_counts.keys()))
train_counts_list = [train_counts.get(cls, 0) for cls in classes]
val_counts_list = [val_counts.get(cls, 0) for cls in classes]
test_counts_list = [test_counts.get(cls, 0) for cls in classes]

x = range(len(classes))
plt.figure(figsize=(20, 10))
plt.bar(x, train_counts_list, width=0.2, label='Тренировка', align='center')
plt.bar(x, val_counts_list, width=0.4, label='Валидация', align='edge')
plt.bar(x, test_counts_list, width=0.2, label='Тест', align='edge')

plt.xlabel('Плотность классов')
plt.ylabel('Кол-во картинок')
plt.title('Оценка компонентов датасета на наполненность')
plt.xticks(ticks=x, labels=classes, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()
