"""https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html"""
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from ImageDataset import ImageDataset
from interface import blur_image

# Download training data from open datasets.
training_data = ImageDataset('input/train', transform=ToTensor())

# Download test data from open datasets.
test_data = ImageDataset('input/train', transform=ToTensor())

batch_size = 1

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for test_sample in test_dataloader:
    print(f"Shape of image [N, C, H, W]: {test_sample['image'].shape}")
    print(f"Shape of blurred image: {test_sample['blurred image'].shape} {test_sample['blurred image'].dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(512 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512 * 512)
        )

    def forward(self, x):
        x = self.flatten(x)
        print(
            f"type of x: {x.dtype}, shape of x: {x.shape}. type of bias: {self.linear_relu_stack[0].bias.dtype}, shape of bias: {self.linear_relu_stack[0].bias.shape}")
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork()  # .to(device)
print(model)


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, sample in enumerate(dataloader):
        print(f"batch: {batch}, X: {sample['image'].dtype}, y: {sample['blurred image'].dtype}")
        # X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(sample['image'])
        loss = loss_fn(blur_image(pred).flatten(), sample['blurred image'].flatten())

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(sample['image'])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            # X, y = X.to(device), y.to(device)
            pred = model(sample['image'])
            test_loss += loss_fn(blur_image(pred).flatten(), sample['blurred image'].flatten()).item()
            correct += (pred.argmax(1) == sample['blurred image']).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
epochs = 5
for t in range(epochs):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Problem found: detaching tensors should NOT be done if you want to backpropagate through them.
# https://discuss.pytorch.org/t/element-0-of-tensors-does-not-require-grad-and-does-not-have-a-grad-fn/32908/28
# https://discuss.pytorch.org/t/why-do-we-need-to-detach-variable-which-contains-hidden-representation/1426/2
# fix: implement the functions in python using Tensor operations.
