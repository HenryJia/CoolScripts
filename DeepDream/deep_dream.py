import numpy as np
from scipy.misc import imsave, imread
from scipy.optimize import fmin_l_bfgs_b
from cv2 import resize

from keras.applications import vgg16
from keras import backend as K
from keras.layers import Input

import matplotlib.pyplot as plt

directory = './ToddProfile.jpg'

# Read image and resize to 600 x 600 for VGG16
img = resize(imread('ToddProfile.jpg', mode = 'RGB'), (600, 600)).astype(np.float32)
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis = 0)

# Zero centre by mean pixels per channel (we'll undo this at the end)
# 'RGB'->'BGR'
img = img[:, ::-1, :, :]
img[:, 0, :, :] -= 103.939
img[:, 1, :, :] -= 116.779
img[:, 2, :, :] -= 123.68
print np.mean(img[:, 0]), np.mean(img[:, 1]), np.mean(img[:, 2])

input_img = Input(shape = (3, 600, 600))

print 'Load VGG16 Model (Cos we lazy and don\'t want to train our own)'

model = vgg16.VGG16(input_tensor = input_img, weights = 'imagenet', include_top = False)

print model.summary()

loss = K.sum(input_img ** 2) / (3 * 600 * 600)
#loss -= 1.1e-4 * K.sum(model.layers[9].output ** 2)
loss -= K.sum(model.layers[-1].output ** 2)

grads = K.gradients(loss, input_img)

print 'Make the functions'

loss_fn = K.function([input_img], loss)
grads_fn = K.function([input_img], grads)

img_flat = img.flatten()

loss_fmin_fn = lambda x: loss_fn([x.reshape(-1, 3, 600, 600)])
grads_fmin_fn = lambda x: grads_fn([x.reshape(-1, 3, 600, 600)]).flatten().astype('float64')

print 'Try Dreaming :D'

img_final = img_flat
for i in xrange(5):
    random_jitter = 2 * (np.random.random((3 * 600 * 600, )) - 0.5)
    img_final += random_jitter

    img_final, min_val, info = fmin_l_bfgs_b(loss_fmin_fn, img_final, fprime = grads_fmin_fn, maxfun = 7)
    img_final_save = np.transpose(np.reshape(np.copy(img_final - random_jitter), (3, 600, 600)), (1, 2, 0))
    img_final_save[:, :, 0] += 103.939
    img_final_save[:, :, 1] += 116.779
    img_final_save[:, :, 2] += 123.68

    #img_final_save[:, :, 0] = (img_final_save[:, :, 0] - np.min(img_final_save[:, :, 0])) / (np.max(img_final_save[:, :, 0]) - np.min(img_final_save[:, :, 0])) * 255.0
    #img_final_save[:, :, 1] = (img_final_save[:, :, 1] - np.min(img_final_save[:, :, 1])) / (np.max(img_final_save[:, :, 1]) - np.min(img_final_save[:, :, 1])) * 255.0
    #img_final_save[:, :, 2] = (img_final_save[:, :, 2] - np.min(img_final_save[:, :, 2])) / (np.max(img_final_save[:, :, 2]) - np.min(img_final_save[:, :, 2])) * 255.0
    # 'BGR' -> 'RGB'
    img_final_save = img_final_save[:, :, ::-1]
    img_final_save = np.clip(img_final_save, 0, 255).astype(np.uint8)
    plt.figure(i)
    plt.imshow(img_final_save)
    imsave('dream' + str(i) + '.jpg', img_final_save)

