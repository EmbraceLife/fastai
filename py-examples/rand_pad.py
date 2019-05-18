from fastai.vision import *
import fastai.vision as fv
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
sd = il.split_by_folder('train', 'valid')
ll = sd.label_from_folder()
ll.train.x[0].show() # plt.show()
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')],[])
ll = ll.transform(tfms)
ll.train.x[0].show() # plt.show()
