from fastai.vision import *
import fastai.vision as fv
path_data = untar_data(URLs.MNIST_TINY)
path_data.ls()
il = ImageList.from_folder(path_data, convert_mode='L')
print(il)
il[0]
