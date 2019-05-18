# Data augmentation is perhaps the most important regularization technique when training a model for Computer Vision: instead of feeding the model with the same pictures every time, we do small random transformations (a bit of rotation, zoom, translation, etc...) that don't change what's inside the image (to the human eye) but do change its pixel values. Models trained with data augmentation will then generalize better.

# To get a set of transforms with default values that work pretty well in a wide range of tasks, it's often easiest to use [`get_transforms`](/vision.transform.html#get_transforms). Depending on the nature of the images in your data, you may want to adjust a few arguments, the most important being:
#
# - `do_flip`: if True the image is randomly flipped (default behavior)
# - `flip_vert`: limit the flips to horizontal flips (when False) or to horizontal and vertical flips as well as 90-degrees rotations (when True)
#
# [`get_transforms`](/vision.transform.html#get_transforms) returns a tuple of two lists of transforms: one for the training set and one for the validation set (we don't want to modify the pictures in the validation set, so the second list of transforms is limited to resizing the pictures). This can be passed directly to define a [`DataBunch`](/basic_data.html#DataBunch) object (see below) which is then associated with a model to begin training.
#
# Note that the defaults for [`get_transforms`](/vision.transform.html#get_transforms) are generally pretty good for regular photos - although here we'll add a bit of extra rotation so it's easier to see the differences.
from fastai.vision import *
tfms = get_transforms(max_rotate=25)
len(tfms)
