from fastai.vision import *
from fastai.vision.transform import _pad_default
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
sd = il.split_by_folder('train', 'valid')
ll = sd.label_from_folder()
x = ll.train.x[0].data
x_pad = _pad_default(x, 1)
