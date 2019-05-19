from fastai.vision import *
from fastai.vision.transform import _crop_default, _crop
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
sd = il.split_by_folder('train', 'valid')
ll = sd.label_from_folder()
img = ll.train.x[0]
title = str(img.shape)
img.show(title=title, figsize=(3,3))
img = ll.train.x[0]
x_pad = _crop_default(img.px, 20)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop_default(img.px, 15)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop_default(img.px, 10)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop(img.px, 10)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop(img.px, 18, 0.1, 0.1)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop(img.px, 18, 0.8, 0.8)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
