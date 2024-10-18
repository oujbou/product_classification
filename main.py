from data_loader.data_loader import load_data
from build_train.build_train import create_model, train_model
from evaluate.evaluate import evaluate_model

if __name__ == "__main__":
    # Define paths and parameters
    data_dir = '/home/toujlakh/Projects/product_classification/preprocessed_data/train'
    batch_size = 32
    epochs = 35
    target_size = (224, 224)

    # Load the data
    train_generator, validation_generator = load_data(data_dir, target_size, batch_size)

    # Create the model
    num_classes = train_generator.num_classes  # Number of coat models
    model = create_model(num_classes, input_shape=(224, 224, 3))

    # Train the model
    history = train_model(model, train_generator, validation_generator, epochs)

    # Evaluate the model on validation data
    evaluate_model(model, validation_generator)

    # Save the trained model
    model.save('efficientnet_coat_classifier.h5')