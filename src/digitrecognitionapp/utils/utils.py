import torch
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt


def train_batch(imgs, label, model, criterion, optimizer):
    model.train()
    optimizer.zero_grad()

    # forward Pass
    pred = model(imgs)
    loss = criterion(pred, label)

    # backward Pass
    loss.backward()
    optimizer.step()

    return loss


@torch.no_grad()
def validation_batch(imgs, label, model, criterion):
    model.eval()

    pred = model(imgs)
    loss = criterion(pred, label)
    return loss


def train(n_epoch, train_dataloader, test_dataloader, model, criterion, optimizer):
    train_loss = []
    test_loss = []

    for epoch in range(1, n_epoch + 1):
        epoch_train_loss, epoch_test_loss = 0, 0

        # train
        for images, label in tqdm(
            train_dataloader, desc=f"Training {epoch} of {n_epoch}"
        ):
            images = images.view(-1, 28 * 28)
            loss = train_batch(images, label, model, criterion, optimizer)
            epoch_train_loss += loss.item()
        epoch_train_loss /= len(train_dataloader)
        train_loss.append(epoch_train_loss)

        # validation
        for images, label in tqdm(test_dataloader, desc="validation"):
            images = images.view(-1, 28 * 28)
            loss = validation_batch(images, label, model, criterion)
            epoch_test_loss += loss.item()
        epoch_test_loss /= len(test_dataloader)
        test_loss.append(epoch_test_loss)

        print(
            f"Epoch {epoch} of {n_epoch}: Training Loss: {epoch_train_loss}, Test Loss: {epoch_test_loss}"
        )
    return train_loss, test_loss


def plotcurve(train_loss, test_loss, train_curve_path):
    epochs = np.arange(len(train_loss))
    plt.figure()
    plt.plot(epochs, train_loss, "b", label="Training Loss")
    plt.plot(epochs, test_loss, "r", label="Test Loss")
    plt.title("Training and Test Loss Curve Over Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(train_curve_path)


def load_model(model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=device, weights_only=False)
    return model


def preprocess_image(image: Image.Image):
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    image = transform(image)
    return image


def predict_digit(model, image_tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.view(-1, 28 * 28).to(device)
        output = model(image_tensor)
        _, predicted = torch.max(output.data, 1)
    return predicted.item()


# def visualization(img_path,model_input_size,viz_result_path,device):
#     img_disp = load_img(img_path,model_input_size,device)
#     plt.figure(figsize=(10, 10))
#     plt.title('Original Image')
#     plt.imshow(img_disp)
#     plt.savefig(viz_result_path)
