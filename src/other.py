import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img

# File to handle other data noise

other = os.listdir('../backend/temp')

other = [filename for filename in other if '.png' in filename]
len(other)
X = []
y = []

for file in other:
    image = load_img(f'../data/temp/{file}', target_size=(200, 200, 3))
    image = np.array(image)
    image = image / 255
    X.append(image[:, :, 0])
    y.append(10)

np.save('../data/X_other', X)
np.save('../data/y_other', y)

X

x = np.array([[1], [2]])

r = np.array([[2], [5]])

np.concatenate((x, r))
