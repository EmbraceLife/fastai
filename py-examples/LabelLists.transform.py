from fastai.vision import *
import fastai.vision as fv
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
sd = il.split_by_folder('train', 'valid')
ll = sd.label_from_folder()
x, y = ll.train[0]
x.show(y=y)
x.show(title='test')
x.show(y=y, title='test')
print(y,x.shape)
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
ll = ll.transform(tfms, size=30, tfm_y=True)
