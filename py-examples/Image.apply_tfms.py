from fastai.vision import *
tfms = get_transforms(max_rotate=25)
len(tfms)
def get_ex(): return open_image('imgs/cat_example.jpg')
# def plots_f(rows, cols, width, height, **kwargs):
#     [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(rows,cols,figsize=(width,height))[1].flatten())]
def plots_f_m(rows, cols, width, height, **kwargs):
    my_enum = plt.subplots(rows,cols,figsize=(width,height))[1].flatten()
    res=[]
    for i,ax in enumerate(my_enum):
        res.append(get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax))

# plots_f(2, 4, 12, 6, size=224)
plots_f_m(2, 4, 12, 6, size=224)
