"""https://pytorch.org/tutorials/beginner/data_loading_tutorial.html"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from interface import \
    get_normalized_images_training_data_from_directory as get_training_data

__all__ = ["ImageDataset"]


class ImageDataset(Dataset):
    """Image dataset."""

    def __init__(
            self,
            directory: str,
            expected_size: tuple = (28, 28),
            radius: int = 1,
            transform=None,
    ):
        """The constructor of the ImageDataset class.

        :param directory: The directory to read the images from.
        :param expected_size: The expected size of the images.
        :param radius: The radius to blur the images with.
        :param transform:
        """
        self.training_data = get_training_data(directory, expected_size)
        self.transform = transform
        self.expected_size = expected_size

    def __len__(self) -> int:
        return len(self.training_data)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        image = self.training_data[index][1]
        filename = self.training_data[index][0]
        sample = {
            "image": np.reshape(image, self.expected_size).astype(np.float32),
            "filename": filename,
        }

        if self.transform:
            sample["image"] = self.transform(sample["image"])[0]

        return sample


if __name__ == "__main__":
    dataset = ImageDataset("input",
                           expected_size=(28, 28),
                           transform=ToTensor())
    for i, samp in enumerate(dataset):
        print(samp)
        # img = np.reshape(samp['image'], (28, 28))
        # blurred_img = np.reshape(samp['blurred image'], (28, 28))
        plt.imshow(samp["image"], cmap="gray", vmin=0, vmax=1)
        plt.show()
        plt.imshow(samp["blurred image"], cmap="gray", vmin=0, vmax=1)
        plt.show()
