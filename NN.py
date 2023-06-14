"""https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html"""
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from ImageDataset import ImageDataset
from interface import blur_tensor, polarize_output
import matplotlib.pyplot as plt


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(512 * 512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.Sigmoid(),
            nn.Linear(512, 512 * 512)
        )

    def forward(self, x) -> Tensor:
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


def train(dataloader, model, loss_fn: nn.MSELoss, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, sample in enumerate(dataloader):
        inp = sample['image'].to(device)

        # Compute prediction error
        pred = model(inp)
        loss = loss_fn.forward(blur_tensor(pred).flatten(), blur_tensor(inp).flatten())

        # print(f"loss: {loss}")

        # Backpropagation
        loss.backward()
        # for i, param in enumerate(model.parameters()):
        #     print(f"grad {i}: {param.grad}")
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(blur_tensor(inp))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model: NeuralNetwork, loss_fn, show_img: bool = False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:
            inp = sample['image'].to(device)
            pred = model.forward(inp)
            if show_img:
                print(polarize_output(pred.cpu()).view(512, 512))
                plt.imshow(polarize_output(pred.cpu()).view(512, 512), cmap='gray', interpolation='nearest', vmin=0, vmax=1)
                plt.show()
                plt.imshow(blur_tensor(polarize_output(pred.cpu())).view(512, 512), cmap='gray', vmin=0, vmax=1)
                plt.show()
                plt.imshow(blur_tensor(inp.cpu()).view(512, 512), cmap='gray', vmin=0, vmax=1)
                plt.show()
            test_loss += loss_fn(blur_tensor(pred).flatten(),
                                 blur_tensor(inp).flatten()).item()
            # correct += (blur_tensor(pred).argmax(1) == blur_tensor(inp)).type(torch.float).sum().item()
    test_loss /= num_batches
    # correct /= size
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
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
    # device = "cpu"
    print(f"Using {device} device")

    model = NeuralNetwork().to(device)
    for param in model.parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
    print(model)
    loss_fn = nn.MSELoss()  # nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    epochs = 2000
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    test(test_dataloader, model, loss_fn, show_img=True)
    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
    #
    # device = "cpu"
    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth"))
    # loss_fn = nn.MSELoss()
    # test(test_dataloader, model, loss_fn, show_img=True)
