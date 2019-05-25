from fastai.vision import *
path = untar_data(URLs.MNIST_TINY)
il = ImageList.from_folder(path, convert_mode='L')
defaults.cmap='binary'
sd = il.split_by_folder(train='train', valid='valid')
labell = sd.label_from_folder()
tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
labell = labell.transform(tfms)
bs = 1 # 128
db = labell.databunch(bs=bs)
data = db.normalize()
def conv(ni,nf):
    return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
model = nn.Sequential(
    conv(1, 8), # 14
    )
xb,yb = data.one_batch()
xb = xb.cpu() # is it really necessary given the previous step
output = model(xb)

