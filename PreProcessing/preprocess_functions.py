import numpy as np
from PIL import Image, ImageStat
from albumentations import ChannelShuffle, Blur, RGBShift, RandomBrightness


def read_image(instance, size_x, size_y):
    image = Image.open(instance[0])
    image = image.resize((size_x, size_y), Image.ANTIALIAS)
    image = image.convert('RGB')
    return np.array(image)


def read_mask(instance, size_x, size_y):
    mask = Image.open(instance[1])
    mask = mask.resize((size_x, size_y), Image.ANTIALIAS)
    mask = np.array(mask, dtype=np.float32)
    mask[mask == 255.] = 0.
    return mask


def one_hot_encoder(mask, size_x, size_y, num_classes):
    one_hot_mask = np.zeros((size_x, size_y, num_classes))
    for i in range(size_x):
        for j in range(size_y):
            pix_val = mask[i, j]
            one_hot_mask[i, j, int(pix_val)] = 1
    return one_hot_mask


def augment_me(image, mask):
    x = image
    y = mask
    
    # Executing with Default Parameters
    aug = ChannelShuffle(p=0.5)
    augmented = aug(image=x, mask=y)
    x = augmented["image"]
    y = augmented["mask"]
    
    aug = Blur(p=0.5)
    augmented = aug(image=x,mask=y)
    x = augmented["image"]
    y = augmented["mask"]
    
    aug = RGBShift(p=0.5)
    augmented = aug(image=x,mask=y)
    x = augmented["image"]
    y = augmented["mask"]
    
    aug = RandomBrightness(p=0.5)
    augmented = aug(image=x,mask=y)
    x = augmented["image"]
    y = augmented["mask"]
    
    return x, y


