import os
from ctypes import *

import numpy as np
from torch import Tensor
from torch import tensor


# Define the data types used in the C struct.
class PgmImage(Structure):
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
        directory: str,
        expected_size: tuple = (512, 512),
        radius: int = 1) -> np.ndarray:
    """Reads the data of all images in the specified directory.

    The data is normalized to values between 0 and 1.
    It is assumed that the images are pgm images.
    training data is formed from the images by blurring them with a given radius.
    :param directory: The directory to read the images from.
    :param expected_size: The expected size of the images.
    :param radius: The radius to blur the images with.
    :return: A numpy array containing a tuple with the normalized image data and corresponding blurred image.
    """
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".pgm"):
            img = ctest.ReadPgm(
                os.path.join(directory, filename).encode("utf-8"))
            if (img.contents.width_ != expected_size[0]
                    or img.contents.height_ != expected_size[1]):
                print(
                    f"An image that is not {expected_size[0]}x{expected_size[1]} was found: {filename} is {img.contents.width_}x{img.contents.height_} instead of 512"
                )
                continue
            normalized_img = ctest.NormalizePgm(img)
            input_data_image = np.ctypeslib.as_array(
                normalized_img, shape=(expected_size[0] *
                                       expected_size[1], )).astype(np.double)
            target_img = ctest.KasperBlur(img, radius)
            normalized_target_img = ctest.NormalizePgm(target_img)
            target_data_image = np.ctypeslib.as_array(
                normalized_target_img,
                shape=(expected_size[0] * expected_size[1], )).astype(
                    np.double)
            images.append((input_data_image, target_data_image))
    return np.array(images)


def polorize_output(output: Tensor) -> Tensor:
    """Polorizes the output of the neural network to a discrete 0 or 1.

    :param output: The output of the neural network.
    :return: The polorized output.
    """
    return tensor(
        list(map(lambda x: 255 if x > 0.5 else 0,
                 output.detach().numpy())))


def blur_image(image: Tensor,
               radius: int = 1,
               shape: tuple = (512, 512)) -> Tensor:
    """Blurs the given image with the given radius.

    :param image: The image to blur.
    :param radius: The radius to blur the image with.
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
