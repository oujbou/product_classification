def evaluate_model(model, validation_generator):
    """
    Function to evaluate the model on the validation data.
    Args:
        model: Trained Keras model to evaluate.
        validation_generator: Data generator for validation data.
    Returns:
        None
    """
    val_loss, val_acc = model.evaluate(validation_generator, steps=len(validation_generator))
    print(f"Validation Loss: {val_loss}")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")