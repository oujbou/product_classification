from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(num_classes, input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Function to create and compile the EfficientNetB0 model.
    Args:
        num_classes (int): Number of classes for classification.
        input_shape (tuple): Shape of the input images.
        learning_rate (float): Learning rate for the Adam optimizer.
    Returns:
        model: Compiled Keras model.
    """
    # Load EfficientNetB0 pre-trained on ImageNet, excluding the top classification layer
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze base model

    # Add custom classification layers on top of EfficientNetB0
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model with Adam optimizer and categorical cross-entropy loss
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_model(model, train_generator, validation_generator, epochs=10):
    """
    Function to train the model on the training data.
    Args:
        model: Keras model to train.
        train_generator: Data generator for training data.
        validation_generator: Data generator for validation data.
        epochs (int): Number of epochs to train for.
    Returns:
        history: Training history of the model.
    """
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=len(validation_generator)
    )
    return history