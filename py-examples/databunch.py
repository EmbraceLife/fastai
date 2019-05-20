from fastai.vision import *
path = untar_data(URLs.MNIST)
il = ImageList.from_folder(path, convert_mode='L')
defaults.cmap='binary'
il[0].show()
labell = il.split_by_folder(train='training', valid='testing').label_from_folder()

tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
labell = labell.transform(tfms)
bs = 128
# not using imagenet_stats because not using pretrained model
db = labell.databunch(bs=bs)
data = db.normalize()
data.show_batch()
