import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.cnn = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 14 * 14, hidden_size)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.cnn(x)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


if __name__ == "__main__":
    # Check if the model is created correctly
    hidden_size = 500
    num_classes = 10
    model = NeuralNet(hidden_size, num_classes)

    # Print the model architecture
    print(model)

    # Check if the model can process a sample input
    sample_input = torch.randn(1, 1, 28, 28)  # Batch size of 1, 1 channel, 28x28 image
    output = model(sample_input)
    result = output.argmax(
        dim=1, keepdim=True
    )  # Get the index of the max log-probability
    print(f"Predicted class: {result.item()}")
