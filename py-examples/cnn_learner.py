from fastai.vision import *
path = untar_data(URLs.MNIST_TINY)
il = ImageList.from_folder(path, convert_mode='L')
defaults.cmap='binary'
il[0].show()
sd = il.split_by_folder(train='train', valid='valid')
labell = sd.label_from_folder()

tfms = ([*rand_pad(padding=3, size=28, mode='zeros')], [])
labell = labell.transform(tfms)
bs = 128
# not using imagenet_stats because not using pretrained model
db = labell.databunch(bs=bs)
data = db.normalize()
data.show_batch()
def conv(ni,nf):
    return nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
model = nn.Sequential(
    conv(1, 8), # 14
    )
xb,yb = data.one_batch()
xb = xb.cpu()
model(xb).shape

#########################
model = nn.Sequential(
    conv(1, 8), # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    )
xb,yb = data.one_batch()
xb = xb.cpu()
model(xb).shape

#########################
model = nn.Sequential(
    conv(1, 8), # 14
    nn.BatchNorm2d(8),
    nn.ReLU(),
    conv(8, 16), # 7
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 32), # 4
    nn.BatchNorm2d(32),
    nn.ReLU(),
    conv(32, 16), # 2
    nn.BatchNorm2d(16),
    nn.ReLU(),
    conv(16, 10), # 1
    nn.BatchNorm2d(10),
    Flatten()     # remove (1,1) grid
)

learn = Learner(data, model, 
        loss_func = nn.CrossEntropyLoss(), metrics=accuracy)
print(learn.summary())
xb,yb = data.one_batch()
xb = xb.cpu()
model(xb).shape
learn.lr_find(end_lr=100)
learn.recorder.plot()
learn.fit_one_cycle(3, max_lr=0.1)
