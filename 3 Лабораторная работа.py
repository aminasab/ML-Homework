# Импортируем необходимые библиотеки
import tensorflow as tf
from google.colab import drive
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Подключаем Google Диск
drive.mount('/content/drive/')

# Загружаем данные MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Изменяем размерность данных
x_train = x_train.reshape((-1, 28*28))  # Вытягиваем каждую картинку в вектор
x_test = x_test.reshape((-1, 28*28))

# Создаем папку для чекпоинтов на Google Диске
path = Path("/content/drive/My Drive/Skillbox/model_1")
path.mkdir(exist_ok=True, parents=True)
assert path.exists()

cpt_filename = "best_checkpoint.hdf5"  
cpt_path = str(path / cpt_filename)

# Создаем модель
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(28*28,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Компилируем модель
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Создаем коллбек для сохранения чекпоинтов
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=cpt_path,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

# Обучаем модель
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_test, y_test),
    callbacks=[checkpoint_callback]
)

# Проверяем условия и утверждения
# < YOUR CODE STARTS HERE >

assert len(list(path.glob("*"))) != 0, f"Checkpoint dir {path}"
assert "accuracy" in history.history, "History object must contain Accuracy. Please, retrain with this metric"
assert "val_accuracy" in history.history, "Please, provide validation_data in model.fit."
assert np.max(history.history["val_accuracy"]) > 0.95, "Validation accuracy must be more than 0.95"

print("Training tests passed")

# < YOUR CODE ENDS HERE >

# Визуализируем результаты обучения
def show_progress(history):
    plt.figure(figsize=(12, 4))
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

show_progress(history.history)

# Загружаем сохраненную модель
restored_model = tf.keras.models.load_model(cpt_path)

# Оцениваем восстановленную модель на тестовых данных
loss, acc = restored_model.evaluate(x_test, y_test)
print(f"Accuracy of restored model {acc*100:.2f}%")
assert acc > 0.96

# Сравниваем предсказания с реальными классами
predicted_labels = np.argmax(restored_model.predict(x_test), axis=-1)
idxs = np.random.choice(np.arange(len(x_test)), 16, replace=False)

def show_mnist(images, true_labels, predicted_labels):
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, img, true_label, pred_label in zip(axes.flatten(), images, true_labels, predicted_labels):
        ax.imshow(img.reshape(28, 28), cmap='gray')
        ax.set_title(f'True: {true_label}, Pred: {pred_label}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_mnist(x_test[idxs], y_test[idxs], predicted_labels[idxs])
