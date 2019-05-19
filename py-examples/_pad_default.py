from fastai.vision import *
from fastai.vision.transform import _pad_default, _pad
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
sd = il.split_by_folder('train', 'valid')
ll = sd.label_from_folder()
# print out the original image
img = ll.train.x[0]
title = str(img.shape)
img.show(title=title, figsize=(3,3))
# add 10 pixels to each side, with default mode `reflection`
img = ll.train.x[0]
x_pad = _pad_default(img.px, 10)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
# print out images with 5 padding on each side
img = ll.train.x[0]
x_pad = _pad_default(img.px, 5)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
# print out images with padding mode 'zeros'
img = ll.train.x[0]
x_pad = _pad_default(img.px, 5, mode='zeros')
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
# print out images with padding mode 'zeros'
img = ll.train.x[0]
x_pad = _pad(img.px, 5, mode='zeros')
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
