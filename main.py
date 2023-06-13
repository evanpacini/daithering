from ctypes import *
import numpy as np


# Define the data types used in the C struct.
class PgmImage(Structure):
    _fields_ = [("width_", c_uint32), ("height_", c_uint32), ("max_gray_", c_uint16), ("data_", POINTER(c_uint8))]


# Load the shared library into c types.
ctest = CDLL("./libnetpbm.dll", winmode=0)

# Define the return type of the C function.
ctest.AllocatePgm.argtypes = [c_uint32, c_uint32]
ctest.AllocatePgm.restype = POINTER(PgmImage)

# Call the C function.
img = ctest.AllocatePgm(2, 3)

# Print the result.
print(img.contents.width_)
print(img.contents.height_)

ctest.WritePgm.argtypes = [c_char_p, POINTER(PgmImage)]
ctest.WritePgm.restype = c_bool

img.contents.data_[0] = 255
img.contents.data_[1] = 0
img.contents.data_[2] = 0
img.contents.data_[3] = 0
img.contents.data_[4] = 255
img.contents.data_[5] = 0

# convert to numpy array (1D)
np_img = np.ctypeslib.as_array(img.contents.data_, shape=(img.contents.width_ * img.contents.height_,))
print(np_img)

ctest.WritePgm("test.pgm".encode("utf-8"), img)

new_image = PgmImage(2, 3, 255, (c_uint8 * 6)(255, 0, 0, 0, 255, 0))

ctest.WritePgm("test2.pgm".encode("utf-8"), byref(new_image))

# read image
ctest.ReadPgm.argtypes = [c_char_p]
ctest.ReadPgm.restype = POINTER(PgmImage)

img2 = ctest.ReadPgm("input/grayscale.pgm".encode("utf-8"))

print(img2.contents.width_)
print(img2.contents.height_)
print(img2.contents.max_gray_)
print(np.ctypeslib.as_array(img2.contents.data_, shape=(img2.contents.width_ * img2.contents.height_,)))

# create image using numpy array

image = np.array([255, 0, 0, 0, 255, 0], dtype=np.uint8)

# convert to c array
c_image = (c_uint8 * len(image))(*image)

new_image = PgmImage(2, 3, 255, c_image)

ctest.WritePgm("test3.pgm".encode("utf-8"), byref(new_image))
