from fastai.vision import *
from fastai.vision.transform import _crop_default, _crop, _crop_pad_default
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
sd = il.split_by_folder('train', 'valid')
ll = sd.label_from_folder()
img = ll.train.x[0]
help(_crop_pad_default)
x_pad = _crop_pad_default(img.px, 20, row_pct=0.5, col_pct=0.5)

x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop_pad_default(img.px, 38, row_pct=0.5, col_pct=0.5)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
img = ll.train.x[0]
x_pad = _crop_pad_default(img.px, 48, row_pct=0.5, col_pct=0.5)
x_pad.shape
img.px = x_pad
title = str(img.shape)
img.show(title=title, figsize=(5,5))
