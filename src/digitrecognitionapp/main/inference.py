import os
from PIL import Image

from src.digitrecognitionapp.config.config import configuration
from src.digitrecognitionapp.utils.utils import (
    load_model,
    preprocess_image,
    predict_digit,
)


def main(image_path, model_path):
    # Load the model
    model = load_model(model_path)

    image = Image.open(image_path)
    image_tensor = preprocess_image(image)

    # Make prediction
    prediction = predict_digit(model, image_tensor)

    return prediction


if __name__ == "__main__":
    # Example usage
    image_path = (
        "./data/testSample/img_300.jpg"  # Path to the image you want to predict
    )

    model_path = os.path.join(
        os.getcwd(), "dump", configuration.get("saved_path", "model"), "model.pth"
    )
    predicted_digit = main(image_path, model_path)
    print(f"Predicted digit: {predicted_digit}")
