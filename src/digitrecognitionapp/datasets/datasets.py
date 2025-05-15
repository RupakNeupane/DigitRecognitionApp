import torchvision
import torchvision.transforms as transforms

train_dataset = torchvision.datasets.MNIST(
    root="./data", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = torchvision.datasets.MNIST(
    root="./data", train=False, transform=transforms.ToTensor(), download=True
)

if __name__ == "__main__":
    # Check if the datasets are loaded correctly
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    print(f"Sample data shape: {train_dataset[0][0].shape}")
    print(f"Sample label: {train_dataset[0][0]}")
