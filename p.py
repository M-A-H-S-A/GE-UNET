# define input data
import numpy as np
import tensorflow
from keras import Sequential
from keras.layers import Conv1D
from PIL import Image
import random

data = np.random.randint = [[random.random() for e in range(1)] for e in range(100)]

data = np.squeeze(np.asarray(data))
print(data)

#data[256,256] = [255,0,0]

img = Image.fromarray(data, 'RGB')
img.show()