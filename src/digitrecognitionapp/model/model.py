import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        return out


if __name__ == "__main__":
    # Check if the model is created correctly
    input_size = 784  # 28x28
    hidden_size = 500
    num_classes = 10
    model = NeuralNet(input_size, hidden_size, num_classes)

    # Print the model architecture
    print(model)

    # Check if the model can process a sample input
    sample_input = torch.randn(1, input_size)
    output = model(sample_input)
    result = output.argmax(
        dim=1, keepdim=True
    )  # Get the index of the max log-probability
    print(f"Predicted class: {result.item()}")
