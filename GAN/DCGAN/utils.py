import scipy.misc
import numpy as np
from types import MethodType


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(
        x[j:j+crop_h, i:i+crop_w],
        [resize_h, resize_w])


def resize_img(dataset, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
    def _resize_img(image):
        image = image.transpose((1, 2, 0))
        if crop:
            cropped_image = center_crop(
                image, input_height, input_width, 
                resize_height, resize_width)
        else:
            cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
#        output = np.array(cropped_image)/127.5 - 1.
        output = cropped_image
        output = output.transpose((2, 0, 1)).astype(np.float32)
        return output

    dataset._get_example = dataset.get_example
    dataset.get_example = MethodType(lambda self, i: _resize_img(self._get_example(i)), dataset)
    return dataset
