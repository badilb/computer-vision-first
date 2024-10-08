from keras.src.legacy.preprocessing.image import ImageDataGenerator

def create_data_generators(dataset_path, img_width, img_height, batch_size):
    """
    Создание генераторов данных для обучения и валидации.

    :param dataset_path: Путь к набору данных.
    :param img_width: Ширина изображений.
    :param img_height: Высота изображений.
    :param batch_size: Размер батча.
    :return: train_generator, validation_generator
    """
    # Data Augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 80% обучение, 20% валидация
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator
