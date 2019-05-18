# %% markdown
# ## Image transforms
# %%
%load_ext autoreload
%autoreload 2
# %%
from fastai.gen_doc.nbdoc import *
from fastai.vision import *
# %% markdown
# fastai provides a complete image transformation library written from scratch in PyTorch. Although the main purpose of the library is data augmentation for use when training computer vision models, you can also use it for more general image transformation purposes. Before we get in to the detail of the full API, we'll look at a quick overview of the data augmentation pieces that you'll almost certainly need to use.
# %% markdown
# ## Data augmentation
# %% markdown
# Data augmentation is perhaps the most important regularization technique when training a model for Computer Vision: instead of feeding the model with the same pictures every time, we do small random transformations (a bit of rotation, zoom, translation, etc...) that don't change what's inside the image (to the human eye) but do change its pixel values. Models trained with data augmentation will then generalize better.
#
# To get a set of transforms with default values that work pretty well in a wide range of tasks, it's often easiest to use [`get_transforms`](/vision.transform.html#get_transforms). Depending on the nature of the images in your data, you may want to adjust a few arguments, the most important being:
#
# - `do_flip`: if True the image is randomly flipped (default behavior)
# - `flip_vert`: limit the flips to horizontal flips (when False) or to horizontal and vertical flips as well as 90-degrees rotations (when True)
#
# [`get_transforms`](/vision.transform.html#get_transforms) returns a tuple of two lists of transforms: one for the training set and one for the validation set (we don't want to modify the pictures in the validation set, so the second list of transforms is limited to resizing the pictures). This can be passed directly to define a [`DataBunch`](/basic_data.html#DataBunch) object (see below) which is then associated with a model to begin training.
#
# Note that the defaults for [`get_transforms`](/vision.transform.html#get_transforms) are generally pretty good for regular photos - although here we'll add a bit of extra rotation so it's easier to see the differences.
# %%
tfms = get_transforms(max_rotate=25)
len(tfms)
# %% markdown
# We first define a function to return a new image, since transformation functions modify their inputs. We also define a little helper function `plots_f` to let us output a grid of transformed images based on a function - the details of this function aren't important here.
# %%
def get_ex(): return open_image('imgs/cat_example.jpg')

def plots_f(rows, cols, width, height, **kwargs):
    [get_ex().apply_tfms(tfms[0], **kwargs).show(ax=ax) for i,ax in enumerate(plt.subplots(
        rows,cols,figsize=(width,height))[1].flatten())]
# %% markdown
# If we want to have a look at what these transforms actually do, we need to use the [`apply_tfms`](/vision.image.html#Image.apply_tfms) function. It will be in charge of picking the values of the random parameters and doing the transformations to the [`Image`](/vision.image.html#Image) object. This function has multiple arguments you can customize (see its documentation for details); here we will highlight the most useful. The first one we'll need to set, especially if our images are of different shapes, is the target `size`. It will ensure that all the images are cropped or padded to the same size so we can then collate them into batches.
# %%
plots_f(2, 4, 12, 6, size=224)
# %% markdown
# Note that the target `size` can be a rectangle if you specify a tuple of int.
# %%
jekyll_note("""In fastai we follow the convention of numpy and pytorch for image dimensions: (height, width). It's different
            from PIL or matplolib so don't get confused.""")
# %%
plots_f(2, 4, 12, 8, size=(300,200))
# %% markdown
# The second argument that can be customized is how we treat missing pixels: when applying transforms (like a rotation), some of the pixels inside the square won't have values from the image. We can set missing pixels to one of the following:
# - black (`padding_mode`='zeros')
# - the value of the pixel at the nearest border (`padding_mode`='border')
# - the value of the pixel symmetric to the nearest border (`padding_mode`='reflection')
#
# `padding_mode`='reflection' is the default. Here is what `padding_mode`='zeros' looks like:
# %%
plots_f(2, 4, 12, 6, size=224, padding_mode='zeros')
# %% markdown
# And `padding_mode`='border' looks like this:
# %%
plots_f(2, 4, 12, 6, size=224, padding_mode='border')
# %% markdown
# The third argument that might be useful to change is [`resize_method`](vision.image.html#ResizeMethod). Images are often rectangles of different ratios, so to get them to the target `size`, we may need to crop, squish, or pad them to get the ratio right.
#
# By default, the library resizes the image while keeping its original ratio so that the smaller size corresponds to the given size, then takes a crop (<code>ResizeMethod.CROP</code>). You can choose to resize the image while keeping its original ratio so that the bigger size corresponds to the given size, then take a pad (<code>ResizeMethod.PAD</code>). Another way is to just squish the image to the given size (<code>ResizeMethod.SQUISH</code>).
# %%
_,axs = plt.subplots(1,3,figsize=(9,3))
for rsz,ax in zip([ResizeMethod.CROP, ResizeMethod.PAD, ResizeMethod.SQUISH], axs):
    get_ex().apply_tfms([crop_pad()], size=224, resize_method=rsz, padding_mode='zeros').show(ax=ax, title=rsz.name.lower())
# %% markdown
# ## Data augmentation details
# %% markdown
# If you want to quickly get a set of random transforms that have worked well in a wide range of tasks, you should use the [`get_transforms`](/vision.transform.html#get_transforms) function. The most important parameters to adjust are *do\_flip* and *flip\_vert*, depending on the type of images you have.
# %%
show_doc(get_transforms,  arg_comments={
    'do_flip': 'if True, a random flip is applied with probability 0.5',
    'flip_vert': 'requires do_flip=True. If True, the image can be flipped vertically or rotated by 90 degrees, otherwise only an horizontal flip is applied',
    'max_rotate': 'if not None, a random rotation between -max\_rotate and max\_rotate degrees is applied with probability p\_affine',
    'max_zoom': 'if not 1. or less, a random zoom betweem 1. and max\_zoom is applied with probability p\_affine',
    'max_lighting': 'if not None, a random lightning and contrast change controlled by max\_lighting is applied with probability p\_lighting',
    'max_warp': 'if not None, a random symmetric warp of magnitude between -max\_warp and maw\_warp is applied with probability p\_affine',
    'p_affine': 'the probability that each affine transform and symmetric warp is applied',
    'p_lighting': 'the probability that each lighting transform is applied',
    'xtra_tfms': 'a list of additional transforms you would like to be applied'
})
# %% markdown
# This function returns a tuple of two lists of transforms, one for the training set and the other for the validation set (which is limited to a center crop by default).
# %%
tfms = get_transforms(max_rotate=25); len(tfms)
# %% markdown
# Let's see how [`get_transforms`](/vision.transform.html#get_transforms) changes this little kitten now.
# %%
plots_f(2, 4, 12, 6, size=224)
# %% markdown
# Another useful function that gives basic transforms is [`zoom_crop`](/vision.transform.html#zoom_crop):
# %%
show_doc(zoom_crop, arg_comments={
    'scale': 'Decimal or range of decimals to zoom the image',
    'do_rand': "If true, transform is randomized, otherwise it's a `zoom` of `scale` and a center crop",
    'p': 'Probability to apply the zoom'
})
# %% markdown
# `scale` should be a given float if `do_rand` is False, otherwise it can be a range of floats (and the zoom will have a random value in between). Again, here is a sense of what this can give us:
# %%
tfms = zoom_crop(scale=(0.75,2), do_rand=True)
plots_f(2, 4, 12, 6, size=224)
# %%
show_doc(rand_resize_crop, ignore_warn=True, arg_comments={
    'size': 'Final size of the image',
    'max_scale': 'Zooms the image to a random scale up to this',
    'ratios': 'Range of ratios in which a new one will be randomly picked'
})
# %% markdown
# This transform is an implementation of the main approach used for nearly all winning Imagenet entries since 2013, based on Andrew Howard's [Some Improvements on Deep Convolutional Neural Network Based Image Classification](https://arxiv.org/abs/1312.5402). It determines a new width and height of the image after the random scale and squish to the new ratio are applied. Those are switched with probability 0.5. Then we return the part of the image with the width and height computed, centered in `row_pct`, `col_pct` if width and height are both less than the corresponding size of the image. Otherwise we try again with new random parameters.
# %%
tfms = [rand_resize_crop(224)]
plots_f(2, 4, 12, 6, size=224)
# %% markdown
# ## Randomness
# %% markdown
# The functions that define each transform, such as [`rotate`](/vision.transform.html#_rotate)or [`flip_lr`](/vision.transform.html#_flip_lr) are deterministic. The fastai library will then randomize them in two different ways:
# - each transform can be defined with an argument named `p` representing the probability for it to be applied
# - each argument that is type-annotated with a random function (like [`uniform`](/torch_core.html#uniform) or [<code>rand_bool</code>](http://docs.fast.ai/vision.image.html#rand_bool)) can be replaced by a tuple of arguments accepted by this function, and on each call of the transform, the argument that is passed inside the function will be picked randomly using that random function.
#
# If we look at the function [`rotate`](/vision.transform.html#_rotate) for instance, we see it has an argument `degrees` that is type-annotated as uniform.
#
# **First level of randomness:** We can define a transform using [`rotate`](/vision.transform.html#_rotate) with `degrees` fixed to a value, but by passing an argument `p`. The rotation will then be executed with a probability of `p` but always with the same value of `degrees`.
# %%
tfm = [rotate(degrees=30, p=0.5)]
fig, axs = plt.subplots(1,5,figsize=(12,4))
for ax in axs:
    img = get_ex().apply_tfms(tfm)
    title = 'Done' if tfm[0].do_run else 'Not done'
    img.show(ax=ax, title=title)
# %% markdown
# **Second level of randomness**: We can define a transform using [`rotate`](/vision.transform.html#_rotate) with `degrees` defined as a range, without an argument `p`. The rotation will then always be executed with a random value picked uniformly between the two floats we put in `degrees`.
# %%
tfm = [rotate(degrees=(-30,30))]
fig, axs = plt.subplots(1,5,figsize=(12,4))
for ax in axs:
    img = get_ex().apply_tfms(tfm)
    title = f"deg={tfm[0].resolved['degrees']:.1f}"
    img.show(ax=ax, title=title)
# %% markdown
# **All combined**: We can define a transform using [`rotate`](/vision.transform.html#_rotate) with `degrees` defined as a range, and an argument `p`. The rotation will then be executed with a probability `p` and with a random value picked uniformly between the two floats we put in `degrees`.
# %%
tfm = [rotate(degrees=(-30,30), p=0.75)]
fig, axs = plt.subplots(1,5,figsize=(12,4))
for ax in axs:
    img = get_ex().apply_tfms(tfm)
    title = f"Done, deg={tfm[0].resolved['degrees']:.1f}" if tfm[0].do_run else f'Not done'
    img.show(ax=ax, title=title)
# %% markdown
# ## List of transforms
# %% markdown
# Here is the list of all the deterministic functions on which the transforms are built. As explained before, each of them can have a probability `p` of being executed, and any time an argument is type-annotated with a random function, it's possible to randomize it via that function.
# %%
show_doc(brightness)
# %% markdown
# This transform adjusts the brightness of the image depending on the value of `change`. A `change` of 0 will transform the image to black, and a `change` of 1 will transform the image to white. `change`=0.5 doesn't adjust the brightness.
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for change, ax in zip(np.linspace(0.1,0.9,5), axs):
    brightness(get_ex(), change).show(ax=ax, title=f'change={change:.1f}')
# %%
show_doc(contrast)
# %% markdown
# `scale` adjusts the contrast. A `scale` of 0 will transform the image to grey, and a `scale` over 1 will transform the picture to super-contrast. `scale` = 1. doesn't adjust the contrast.
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for scale, ax in zip(np.exp(np.linspace(log(0.5),log(2),5)), axs):
    contrast(get_ex(), scale).show(ax=ax, title=f'scale={scale:.2f}')
# %%
show_doc(crop)
# %% markdown
# This transform takes a crop of the image to return one of the given size. The position is given by `(col_pct, row_pct)`, with `col_pct` and `row_pct` being normalized between 0. and 1.
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for center, ax in zip([[0.,0.], [0.,1.],[0.5,0.5],[1.,0.], [1.,1.]], axs):
    crop(get_ex(), 300, *center).show(ax=ax, title=f'center=({center[0]}, {center[1]})')
# %%
show_doc(crop_pad, ignore_warn=True, arg_comments={
    'x': 'Image to transform',
    'size': "Size of the crop, if it's an int, the crop will be square",
    'padding_mode': "How to pad the output image ('zeros', 'border' or 'reflection')",
    'row_pct': 'Between 0. and 1., position of the center on the y axis (0. is top, 1. is bottom, 0.5 is center)',
    'col_pct': 'Between 0. and 1., position of the center on the x axis (0. is left, 1. is right, 0.5 is center)'
})
# %% markdown
# This works like [`crop`](/vision.transform.html#_crop) but if the target size is bigger than the size of the image (on either dimension), padding is applied according to `padding_mode` (see [`pad`](/vision.transform.html#_pad) for an example of all the options) and the position of center is ignored on that dimension.
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for size, ax in zip(np.linspace(200,600,5), axs):
    crop_pad(get_ex(), int(size), 'zeros', 0.,0.).show(ax=ax, title=f'size = {int(size)}')
# %%
show_doc(dihedral)
# %% markdown
# This transform applies combines a flip (horizontal or vertical) and a rotation of a multiple of 90 degrees.
# %%
fig, axs = plt.subplots(2,4,figsize=(12,8))
for k, ax in enumerate(axs.flatten()):
    dihedral(get_ex(), k).show(ax=ax, title=f'k={k}')
plt.tight_layout()
# %%
show_doc(dihedral_affine)
# %% markdown
# This is an affine implementation of [`dihedral`](/vision.transform.html#_dihedral) that should be used if the target is an [`ImagePoints`](/vision.image.html#ImagePoints) or an [`ImageBBox`](/vision.image.html#ImageBBox).
# %%
show_doc(flip_lr)
# %% markdown
# This transform horizontally flips the image. [`flip_lr`](/vision.transform.html#_flip_lr) mirrors the image.
# %%
fig, axs = plt.subplots(1,2,figsize=(6,4))
get_ex().show(ax=axs[0], title=f'no flip')
flip_lr(get_ex()).show(ax=axs[1], title=f'flip')
# %%
show_doc(flip_affine)
# %% markdown
# This is an affine implementation of [`flip_lr`](/vision.transform.html#_flip_lr) that should be used if the target is an [`ImagePoints`](/vision.image.html#ImagePoints) or an [`ImageBBox`](/vision.image.html#ImageBBox).
# %%
show_doc(jitter, doc_string=False)
# %% markdown
# This transform changes the pixels of the image by randomly replacing them with pixels from the neighborhood (how far the neighborhood extends is controlled by the value of `magnitude`).
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for magnitude, ax in zip(np.linspace(-0.05,0.05,5), axs):
    tfm = jitter(magnitude=magnitude)
    get_ex().jitter(magnitude).show(ax=ax, title=f'magnitude={magnitude:.2f}')
# %%
show_doc(pad)
# %% markdown
# Pad the image by adding `padding` pixel on each side of the picture according to `mode`:
# - `mode='zeros'`:  pads with zeros,
# - `mode='border'`: repeats the pixels at the border.
# - `mode='reflection'`: pads by taking the pixels symmetric to the border.
# %%
fig, axs = plt.subplots(1,3,figsize=(12,4))
for mode, ax in zip(['zeros', 'border', 'reflection'], axs):
    pad(get_ex(), 50, mode).show(ax=ax, title=f'mode={mode}')
# %%
show_doc(perspective_warp)
# %% markdown
# Perspective warping is a deformation of the image as seen in a different plane of the 3D-plane. The new plane is determined by telling where we want each of the four corners of the image (from -1 to 1, -1 being left/top, 1 being right/bottom).
# %%
fig, axs = plt.subplots(2,4,figsize=(12,8))
for i, ax in enumerate(axs.flatten()):
    magnitudes = torch.tensor(np.zeros(8))
    magnitudes[i] = 0.5
    perspective_warp(get_ex(), magnitudes).show(ax=ax, title=f'coord {i}')
# %% markdown
# #### resize
#
# Pytorch's `transforms.Resize(size)` equivalent is implemented without an explicit transform function in fastai. It's done via the arguments `size` and `resize_method`.
#
# The `size` argument can be either a single `int` `224`, or a tuple of `int`s `(224,400)`. The default behavior is to crop the image to a square when a single `int` is passed and to squish it in the case of a `tuple`, so that:
# * if size=224 is passed, it will resize and then crop to (224,224)
# * if size=(224,400) is passed, it will squish it to (224,400)
# * if size=(224,224) is passed, it will squish (not crop!) it to (224,224)
#
# You can override the default [`resize_method`](vision.image.html#ResizeMethod).
#
# Note:
#
# If you receive an error similar to the one below:
#
# ```
# RuntimeError: Argument #4: Padding size should be less than the corresponding input dimension, but got: padding (46, 46) at dimension 3 of input [1, 3, 128, 36]
# ```
#
# this is caused by an issue with PyTorch's reflection padding, which the library uses by default. Adding an extra keyword argument `padding_mode = 'zeros'` should be able to serve as a workaround for now.
#
# The resize is performed slightly differently depending on how [`ImageDataBunch`](/vision.data.html#ImageDataBunch) is created:
#
# 1. When the shortcut [`ImageDataBunch`](/vision.data.html#ImageDataBunch) `from_*` methods are used, the `size` and `resize_method` arguments are passed with the rest of the arguments. For example, to resize images on the fly to `224x224` with `from_name_re` method, do:
#
#    ```
#    data = ImageDataBunch.from_name_re(path_img, fnames, pat, size=224, bs=bs)
#    ```
#     and to override the `resize_method`:
#    ```
#    data = ImageDataBunch.from_name_re(path_img, fnames, pat, size=224, resize_method=ResizeMethod.SQUISH, bs=bs)
#    ```
#
# 2. When [data block API](/data_block.html) is used, the `size` and `resize_method` are passed via the [`transform`](/vision.transform.html#vision.transform) method. For example:
#    ```
#    src = ImageList.from_folder(path).split_none().label_from_folder()
#    tfms = get_transforms() # or tfms=None if none are needed
#    size=224 # size=(224,224) or (400,224)
#    data = src.transform(tfms=tfms, size=size, resize_method=ResizeMethod.SQUISH).databunch(bs=bs, num_workers=4).normalize()
#    ```
#
#
#
#
# **Resizing before training**
#
# Do note that if you just want to resize the input images, doing it on the fly via transform is inefficient, since it will have to be done on every notebook re-run. Chances are that you will want to resize the images on the filesystem and use the resized datasets as needed.
#
# For example, you could use fastai code to do that, to create low-resolution images under 'small-96', and mid-resolution images under `small-256`:
# ```
# from fastai.vision import *
#
# path = untar_data(URLs.PETS)
# path_hr = path/'images'
# path_lr = path/'small-96'
# path_mr = path/'small-256'
#
# il = ImageList.from_folder(path_hr)
#
# def resize_one(fn, i, path, size):
#     dest = path/fn.relative_to(path_hr)
#     dest.parent.mkdir(parents=True, exist_ok=True)
#     img = PIL.Image.open(fn)
#     targ_sz = resize_to(img, size, use_min=True)
#     img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
#     img.save(dest, quality=75)
#
# # create smaller image sets the first time this nb is run
# sets = [(path_lr, 96), (path_mr, 256)]
# for p,size in sets:
#     if not p.exists():
#         print(f"resizing to {size} into {p}")
#         parallel(partial(resize_one, path=p, size=size), il.items)
# ```
# Of course, adjust `quality`, `resample`, and other arguments to suit your needs. You will also need to tweak it for custom directories ([`train`](/train.html#train), `test`, etc.)
#
# [imagemagick](https://www.imagemagick.org/)'s `mogrify` and `convert` are commonly-used tools to resize images via your shell. For example, if you're in a [`data`](/vision.data.html#vision.data) directory, containing a `test` directory:
#
# ```
# ls -1 *
# test
# ```
# and you want to create a new directory `300x224/test` with images resized to `300x224`:
#
# ```
# SRC=train; DEST=300x224; mkdir -p $DEST/$SRC; find $SRC -name "*.jpg" -exec convert -resize 300x224 -gravity center -extent 300x224 {} $DEST/{} \;
# ```
# Check the imagemagick documentation for the many various options.
#
# If you already have a directory which is a copy of original images, `mogrify` is usually applied directly to the files with the same result:
#
# ```
# mkdir 300x224
# cp -r train 300x224
# cd 300x224/train
# mogrify -resize 300x224 -gravity center -extent 300x224 *jpg
# ```
# %%
jekyll_note("""In fastai we follow the convention of numpy and pytorch for image dimensions: (height, width). It's different
            from PIL or matplotlib, so don't get confused. Passing size=(300,200) for instance will give you a height of 300
            and a width of 200.""")
# %%
show_doc(rotate)
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for deg, ax in zip(np.linspace(-60,60,5), axs):
    get_ex().rotate(degrees=deg).show(ax=ax, title=f'degrees={deg}')
# %%
show_doc(rgb_randomize)
# %% markdown
# - _channel_: Which channel (RGB) to randomise
# - _thresh_: After randomising, scale the values to not exceed the `thresh` value
#
# By randomizing one of the three channels, the [`learner`](/vision.learner.html#vision.learner) essentially sees the same image, but with different colors. Usually, every RGB image has one channel that is dominant, and randomizing this channel is the riskiest; thus, a low `thresh` (threshold) value must be applied. In this example, the _Green_ `channel` is the dominant one.
# %%
fig, axs = plt.subplots(3,3,figsize=(12,12))
channels = ['Red', 'Green', 'Blue']

for i in np.arange(0, 3):
    for thresh, ax in zip(np.linspace(0.2, 0.99, 3), axs[:, i]):
        get_ex().rgb_randomize(channel = i, thresh = thresh).show(
            ax=ax, title = f'{channels[i]}, thresh={thresh}')
# %%
show_doc(skew)
# %%
fig, axs = plt.subplots(2,4,figsize=(12,8))
for i, ax in enumerate(axs.flatten()):
    get_ex().skew(i, 0.2).show(ax=ax, title=f'direction={i}')
# %%
show_doc(squish)
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for scale, ax in zip(np.linspace(0.66,1.33,5), axs):
    get_ex().squish(scale=scale).show(ax=ax, title=f'scale={scale:.2f}')
# %%
show_doc(symmetric_warp)
# %% markdown
# Apply the four tilts at the same time, each with a strength given in the vector `magnitude`. See [`tilt`](/vision.transform.html#_tilt) just below for the effect of each individual tilt.
# %%
tfm = symmetric_warp(magnitude=(-0.2,0.2))
_, axs = plt.subplots(2,4,figsize=(12,6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm, padding_mode='zeros')
    img.show(ax=ax)
# %%
show_doc(tilt)
# %% markdown
# `direction` is a number (0: left, 1: right, 2: top, 3: bottom). A positive `magnitude` is a tilt forward (toward the person looking at the picture), a negative `magnitude` a tilt backward.
# %%
fig, axs = plt.subplots(2,4,figsize=(12,8))
for i in range(4):
    get_ex().tilt(i, 0.4).show(ax=axs[0,i], title=f'direction={i}, fwd')
    get_ex().tilt(i, -0.4).show(ax=axs[1,i], title=f'direction={i}, bwd')
# %%
show_doc(zoom)
# %%
fig, axs = plt.subplots(1,5,figsize=(12,4))
for scale, ax in zip(np.linspace(1., 1.5,5), axs):
    get_ex().zoom(scale=scale).show(ax=ax, title=f'scale={scale:.2f}')
# %%
show_doc(cutout)
# %% markdown
# The normalization technique described in this paper: [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
#
# By default, it will apply a single cutout (`n_holes=1`) of `length=40`) with probability `p=1`. The cutout position is always random. If you choose to do multiple cutouts, they may overlap.
#
# The paper above used cutouts of size 16x16 for CIFAR-10 (10 categiries classification) and cutouts of size 8x8 for CIFAR-100 (100 categories). Generally, the more categories, the less cutout you want.
# %%
tfms = [cutout()]
fig, axs = plt.subplots(1,5,figsize=(12,4))
for ax in axs:
    get_ex().apply_tfms(tfms).show(ax=ax)
# %% markdown
# You can add some randomness to the cutouts like this:
# %%
tfms = [cutout(n_holes=(1,4), length=(10, 160), p=.5)]
fig, axs = plt.subplots(1,5,figsize=(12,4))
for ax in axs:
    get_ex().apply_tfms(tfms).show(ax=ax)
# %% markdown
# ## Convenience functions
# %% markdown
# These functions simplify creating random versions of [`crop_pad`](/vision.transform.html#_crop_pad) and [`zoom`](/vision.transform.html#_zoom).
# %%
show_doc(rand_crop)
# %% markdown
# The `args` are for internal purposes and shouldn't be touched.
# %%
tfm = rand_crop()
_, axs = plt.subplots(2,4,figsize=(12,6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm, size=224)
    img.show(ax=ax)
# %%
show_doc(rand_pad)
# %%
tfm = rand_pad(4, 224)
_, axs = plt.subplots(2,4,figsize=(12,6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm, size=224)
    img.show(ax=ax)
# %%
show_doc(rand_zoom)
# %%
tfm = rand_zoom(scale=(1.,1.5))
_, axs = plt.subplots(2,4,figsize=(12,6))
for ax in axs.flatten():
    img = get_ex().apply_tfms(tfm)
    img.show(ax=ax)
# %% markdown
# ## Undocumented Methods - Methods moved below this line will intentionally be hidden
# %% markdown
# ## New Methods - Please document or move to the undocumented section
