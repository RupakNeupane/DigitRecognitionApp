import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.digitrecognitionapp.datasets.datasets import train_dataset, test_dataset
from src.digitrecognitionapp.model.model import NeuralNet
from src.digitrecognitionapp.utils.utils import train, plotcurve
from src.digitrecognitionapp.config.config import configuration


def main():
    saved_path = os.path.join(
        os.getcwd(), "dump", configuration.get("saved_path", "model")
    )
    model_path = os.path.join(saved_path, "model.pth")
    hyperparam_path = os.path.join(saved_path, "hyperparam. json")
    train_curve_path = os.path.join(saved_path, "train_curve.png")

    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataloader = DataLoader(
        train_dataset, batch_size=configuration["batch_size"], shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=configuration["batch_size"], shuffle=False
    )
    model = NeuralNet(28 * 28, 500, 10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=configuration["learning_rate"])
    train_loss, test_loss = train(
        configuration["n_epoch"],
        train_dataloader,
        test_dataloader,
        model,
        criterion,
        optimizer,
    )
    plotcurve(train_loss, test_loss, train_curve_path)

    # Save the model
    torch.save(model, model_path)

    # Save the hyperparameters
    with open(hyperparam_path, "w") as f:
        json.dump(configuration, f)

    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

    # # Check if the image is loaded correctly
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # img = train_dataset[0][0].to(device)
    # input = img.view(-1, 28*28)  # Flatten the image
    # print(f"Image shape: {img.shape}")
    # print(f"Input shape: {input.shape}")
    # model_path = os.path.join(os.getcwd(), "dump", configuration.get('saved_path'), "model.pth")
    # model = torch.load(model_path,map_location=device, weights_only=False)
    # output = model(input)
    # result = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
    # print(f"Predicted class: {result.item()}")
