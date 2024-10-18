from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, target_size=(224, 224), batch_size=32, validation_split=0.2):
    """
    Function to load and augment data using ImageDataGenerator with a dynamic train/validation split.
    Args:
        data_dir (str): Directory containing the dataset.
        target_size (tuple): Image size for resizing.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of the data to reserve for validation.
    Returns:
        train_generator: Data generator for the training set.
        validation_generator: Data generator for the validation set.
    """
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split
    )

    # Training generator
    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Validation generator
    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator
