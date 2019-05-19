from fastai.vision import *
tfms = get_transforms(max_rotate=25)
len(tfms)
my_crop_pad = tfms[0][0]
# RandTransform(tfm=TfmCrop (crop_pad), kwargs={'row_pct': (0, 1), 'col_pct': (0, 1), 'padding_mode': 'reflection'}, p=1.0, resolved={}, do_run=True, is_random=True, use_on_y=True)

my_crop_pad.resolve()
my_crop_pad
my_crop_pad.resolve()
my_crop_pad
