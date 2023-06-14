"""https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html"""
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from ImageDataset import ImageDataset
from interface import blur_tensor, polarize_output
import matplotlib.pyplot as plt

SHOULD_TRAIN = True
LOAD_MODEL = False
EPOCHS = 100000
LEARNING_RATE = 5e-3
BATCH_SIZE = 3
SEED = 0
BATCH_DISPLAY_INTERVAL = None
LOSS_FUNCTION = nn.MSELoss()  # nn.CrossEntropyLoss()
DIR_TRAINING_DATA = "input/train"
DIR_TEST_DATA = "input/train"
SAVE_MODEL_NAME = f"model2_e{EPOCHS}_l{LEARNING_RATE}_b{BATCH_SIZE}_s{SEED}_lf{LOSS_FUNCTION._get_name()}.pth"
LOAD_MODEL_NAME = SAVE_MODEL_NAME.replace(f"_e{EPOCHS}", "_e11526") + ".interrupted"


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(512 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Hardsigmoid(),
            nn.Linear(512, 512 * 512),
        )

    def forward(self, x) -> Tensor:
        """Forward pass of the model.

        :param x: The input tensor.
        :return: The output tensor.
        """
        x = self.flatten(x)
        return self.linear_stack(x)


def train(dataloader, network_model: NeuralNetwork, loss_fn, network_optimizer):
    """Trains the model on the given dataloader.

    :param dataloader: The dataloader to train on.
    :param network_model: The model to train.
    :param loss_fn: The loss function to use.
    :param network_optimizer: The optimizer to use.
    """
    size = len(dataloader.dataset)
    network_model.train()
    for batch, sample in enumerate(dataloader):
        inputs = sample["image"].to(device)

        # Compute prediction error
        predictions = network_model(inputs)
        loss = loss_fn.forward(blur_tensor(predictions), blur_tensor(inputs))

        # Backpropagation
        loss.backward()
        network_optimizer.step()
        network_optimizer.zero_grad()

        if BATCH_DISPLAY_INTERVAL is not None:
            if batch % BATCH_DISPLAY_INTERVAL == 0:
                current = (batch + 1) * len(blur_tensor(inputs))
                print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, network_model: NeuralNetwork, loss_fn, show_img: bool = False):
    """Tests the model on the given dataloader.

    :param dataloader: The dataloader to test on.
    :param network_model: The model to test.
    :param loss_fn: The loss function to use.
    :param show_img: Whether to show the image or not.
    """
    num_batches = len(dataloader)
    network_model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            inputs = sample["image"].to(device)
            predictions = network_model.forward(inputs)
            if show_img:
                show_images(sample, predictions)
            test_loss += loss_fn.forward(
                blur_tensor(predictions), blur_tensor(inputs)
            ).item()
    test_loss /= num_batches
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


def show_images(sample, predictions):
    """Shows the images in the given sample.

    :param sample: The sample to show.
    :param predictions: The predictions to show.
    """
    for filename, sample, predictions in zip(
        sample["filename"], sample["image"], predictions
    ):
        show_image_single(filename, sample, predictions)


def show_image_single(
    filename, original, prediction, image_shape: tuple[int, int] = (512, 512)
):
    """Shows the image with the given filename, original and prediction.

    :param filename: The filename of the image.
    :param original: The original image.
    :param prediction: The prediction of the image.
    :param image_shape: The shape of the image.
    """
    fig = plt.figure(figsize=(5, 12))
    fig.add_subplot(3, 1, 1)
    plt.imshow(
        polarize_output(prediction.cpu()).view(image_shape),
        cmap="gray",
        interpolation="nearest",
        vmin=0,
        vmax=1,
    )
    plt.title(f"{filename}: prediction")
    fig.add_subplot(3, 1, 2)
    plt.imshow(
        blur_tensor(prediction.cpu()).view(image_shape), cmap="gray", vmin=0, vmax=1
    )
    plt.title(f"{filename}: prediction blurred")
    fig.add_subplot(3, 1, 3)
    plt.imshow(
        blur_tensor(original.cpu()).view(image_shape), cmap="gray", vmin=0, vmax=1
    )
    plt.title(f"{filename}: original blurred")
    plt.tight_layout(h_pad=1.0)
    plt.show()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    # Download training data from open datasets.
    training_data = ImageDataset(DIR_TRAINING_DATA, transform=ToTensor())

    # Download test data from open datasets.
    test_data = ImageDataset(DIR_TEST_DATA, transform=ToTensor())

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    for test_sample in test_dataloader:
        print(f"Shape of image [N, C, H, W]: {test_sample['image'].shape}")
        print(f"filename of image: {test_sample['filename']}")
        break

    if not SHOULD_TRAIN:
        device = "cpu"
        model = NeuralNetwork().to(device)
        model.load_state_dict(torch.load(LOAD_MODEL_NAME))
        test(test_dataloader, model, LOSS_FUNCTION, show_img=True)
        exit(0)

    # Get cpu, gpu or mps device for training.
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    if LOAD_MODEL:
        model.load_state_dict(torch.load(LOAD_MODEL_NAME))
        print(f"Loaded PyTorch Model State from {LOAD_MODEL_NAME}")
    print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    epochs_completed = 0
    if LOAD_MODEL:
        epochs_completed = int(LOAD_MODEL_NAME.split("_e")[1].split("_")[0])
    try:
        for t in range(epochs_completed, EPOCHS):
            print(f"Epoch {t + 1}\n-------------------------------")
            train(train_dataloader, model, LOSS_FUNCTION, optimizer)
            test(test_dataloader, model, LOSS_FUNCTION)
            epochs_completed += 1
        print("Done!")
    except KeyboardInterrupt:
        print("You pressed CTRL + C.")
        print("Program interrupted.")
        interrupted_model_name = (
            SAVE_MODEL_NAME.replace(f"_e{EPOCHS}", f"_e{epochs_completed}")
            + ".interrupted"
        )
        torch.save(model.state_dict(), interrupted_model_name)
        print(f"Saved PyTorch Model State to {interrupted_model_name}")
        exit(0)

    torch.save(model.state_dict(), SAVE_MODEL_NAME)
    print(f"Saved PyTorch Model State to {SAVE_MODEL_NAME}")

    test(test_dataloader, model, LOSS_FUNCTION, show_img=True)
