from fastai.vision import *
path = untar_data(URLs.MNIST_TINY)
il = ImageList.from_folder(path, convert_mode='L')
defaults.cmap='binary'
sd = il.split_by_folder(train='train', valid='valid')
labell = sd.label_from_folder()
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
labell = labell.transform(tfms)
bs = 128
# not using imagenet_stats because not using pretrained model
data = labell.databunch(bs=bs).normalize()
def _plot(i,j,ax): data.train_ds[0][0].show(ax, cmap='gray')
plot_multi(_plot, 3, 3, figsize=(8,8))
xb,yb = data.one_batch()
xb.shape,yb.shape
data.show_batch(rows=3, figsize=(5,5))
