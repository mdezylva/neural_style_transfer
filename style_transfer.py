import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import functools
import time
from PIL import Image
import numpy as np
import os
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras.preprocessing import image as kp_image

import transfer

mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


# Insert your prefered path here
root_dir = '/home/mitchell/Documents/projects/cnn_style_transfer/'
im_dir =  root_dir+ 'images/'
style_dir = im_dir + 'style/'
content_dir = im_dir + 'content/'
output_path = im_dir + 'output/'

if not os.path.exists(im_dir):
    os.makedirs(im_dir)
if not os.path.exists(style_dir):
    os.makedirs(style_dir)
if not os.path.exists(content_dir):
    os.makedirs(content_dir)
if not os.path.exists(output_path):
    os.makedirs(output_path)

tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))

# Set up some global values here
content = input("Enter name of content image to be converted: ")
content_path = content_dir + content + '.jpg'

style = input("Enter name of style image to be used: ")
style_path = style_dir + style + '.jpg'

# Show the base images
plt.figure(figsize=(10, 10))

content = transfer.load_img(content_path).astype('uint8')
style = transfer.load_img(style_path).astype('uint8')

plt.subplot(1, 2, 1)
transfer.imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
transfer.imshow(style, 'Style Image')
plt.show()

iter_num = int(input('Number of times to iterate: '))

best, best_loss = transfer.run_style_transfer(
    content_path, style_path, num_iterations=iter_num)

transfer.show_results(best, content_path, style_path)

output = Image.fromarray(best)
os.chdir(output_path)
filename = input('Enter filename to save as:')
output.save(filename,format='TIFF')
