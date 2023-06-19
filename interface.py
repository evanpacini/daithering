"""This module provides an interface to the c library of netpbm-c."""
import os
from ctypes import *

import numpy as np
from torch import Tensor
from torch import tensor
from torchvision.transforms import GaussianBlur

__all__ = [
    "get_normalized_images_training_data_from_directory",
    "polarize_output",
    "blur_image",
    "blur_tensor",
    "write_image",
]


class PgmImage(Structure):
    """The PgmImage struct used by the c library."""

    _fields_ = [
        ("width_", c_uint32),
        ("height_", c_uint32),
        ("max_gray_", c_uint16),
        ("data_", POINTER(c_uint8)),
    ]


# Load the shared library into c types.
ctest = CDLL("./libnetpbm.dll", winmode=0)

# read image
ctest.ReadPgm.argtypes = [c_char_p]
ctest.ReadPgm.restype = POINTER(PgmImage)

# normalize image
ctest.NormalizePgm.argtypes = [POINTER(PgmImage)]
ctest.NormalizePgm.restype = POINTER(c_double)

# blur image
ctest.KasperBlur.argtypes = [POINTER(PgmImage), c_int8]
ctest.KasperBlur.restype = POINTER(PgmImage)

# write pgm
ctest.WritePgm.argtypes = [c_char_p, POINTER(PgmImage)]
ctest.WritePgm.restype = c_bool


def get_normalized_images_training_data_from_directory(
    directory: str, expected_size: tuple = (512, 512)
) -> list[tuple[str, np.ndarray]]:
    """Reads the data of all images in the specified directory.

    The data is normalized to values between 0 and 1.
    It is assumed that the images are pgm images.
    :param directory: The directory to read the images from.
    :param expected_size: The expected size of the images.
    :return: A list containing a tuple with the image filename and the normalized image data.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".pgm"):
            img = ctest.ReadPgm(
                os.path.join(directory, filename).encode("utf-8"))
            if (img.contents.width_ != expected_size[0]
                    or img.contents.height_ != expected_size[1]):
                print(
                    f"An image that is not {expected_size[0]}x{expected_size[1]} was found: {filename}"
                    f" is {img.contents.width_}x{img.contents.height_}")
                continue
            normalized_img = ctest.NormalizePgm(img)
            input_data_image = np.ctypeslib.as_array(
                normalized_img, shape=(expected_size[0] *
                                       expected_size[1], )).astype(np.double)
            images.append((filename, input_data_image))
    return images


def polarize_output(output: Tensor) -> Tensor:
    """Polarizes the output of the neural network to a discrete 0 or 1.

    :param output: The output of the neural network.
    :return: The polarized output.
    """
    return tensor((output.detach().numpy() > 0.5).astype(np.float32))


def blur_image(image: Tensor,
               radius: int = 1,
               shape: tuple = (512, 512)) -> Tensor:
    """Blurs the given image with the given radius.

    :param image: The image to blur.
    :param radius: The radius with which to blur the image.
    :param shape: The shape of the image.
    :return: The blurred image.
    """
    img = PgmImage(
        width_=shape[0],
        height_=shape[1],
        max_gray_=255,
        data_=image.detach().numpy().astype(np.uint8).ctypes.data_as(
            POINTER(c_uint8)),
    )
    blurred_img = ctest.KasperBlur(img, radius)
    normalized_blurred_img = ctest.NormalizePgm(blurred_img)
    return tensor(
        np.ctypeslib.as_array(normalized_blurred_img,
                              shape=(shape[0] * shape[1], )).astype(
                                  np.float32))


def blur_tensor(
        output: Tensor,
        image_shape: tuple[int, int] = (512, 512),
        kernel_size: float = 5,
        sigma: tuple[float, float] = (5, 5),
) -> Tensor:
    """Blurs the given tensor with the given kernel size and sigma.

    :param output: The tensor to blur.
    :param image_shape: The shape of the image.
    :param kernel_size: The kernel size to use.
    :param sigma: The sigma to use.
    :return: The blurred tensor.
    """
    try:
        return GaussianBlur(kernel_size,
                            sigma=sigma)(output.view(output.shape[0],
                                                     image_shape[0],
                                                     image_shape[1]))
    except RuntimeError:
        return GaussianBlur(kernel_size,
                            sigma=sigma)(output.view(1, image_shape[0],
                                                     image_shape[1]))


def write_image(image: Tensor, filename: str):
    """Writes the given image to the given filename.

    :param image: The image to write.
    :param filename: The filename to write to.
    """
    img = PgmImage(
        width_=image.shape[0],
        height_=image.shape[1],
        max_gray_=255,
        data_=(image.detach().numpy() * 255).astype(np.uint8).ctypes.data_as(
            POINTER(c_uint8)),
    )
    ctest.WritePgm(filename.encode("utf-8"), img)
