from keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal, GlorotUniform


from preprocessing import train_generator, validation_generator, img_width, img_height

def build_model(activation='relu', initializer='glorot_uniform'):
    model = Sequential()

    # Сверточные слои
    model.add(Conv2D(32, (3, 3), activation=activation, input_shape=(img_width, img_height, 3),
                     kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation=activation, kernel_initializer=initializer))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Полносвязные слои
    model.add(Flatten())
    model.add(Dense(128, activation=activation, kernel_initializer=initializer))
    model.add(Dropout(0.5))

    model.add(Dense(train_generator.num_classes, activation='softmax'))

    # Компиляция модели
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


# Модель с активацией ReLU и инициализацией Xavier
relu_model = build_model(activation='relu', initializer=GlorotUniform())
relu_model.summary()

# Модель с активацией Sigmoid и случайной инициализацией весов
sigmoid_model = build_model(activation='sigmoid', initializer=RandomNormal(stddev=0.01))
sigmoid_model.summary()


import matplotlib.pyplot as plt

# Параметры обучения
epochs = 10  # Задайте количество эпох

# Обучение модели с ReLU активацией
history_relu = relu_model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Обучение модели с Sigmoid активацией
history_sigmoid = sigmoid_model.fit(train_generator, validation_data=validation_generator, epochs=epochs)

# Визуализация результатов
def plot_history(history, title):
    plt.figure(figsize=(12, 4))

    # Потери
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='validation loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Точность
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='validation accuracy')
    plt.title(f'{title} - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Визуализируем результаты для обеих моделей
plot_history(history_relu, 'ReLU Model')
plot_history(history_sigmoid, 'Sigmoid Model')

# Сохранение обученных моделей
relu_model.save('relu_model.h5')
sigmoid_model.save('sigmoid_model.h5')

