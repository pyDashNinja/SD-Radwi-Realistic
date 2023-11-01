from PIL import Image


def ImageResize(img, target_size, constant):
    # img is a Pillow image object, get height (h) and width (w)
    h, w = img.size

    if h * w <= target_size * target_size:
        pass
    elif h * w <= target_size * target_size * constant * constant:
        # make h*w product equal to target_size*target_size
        scale = (target_size * target_size * constant) / (h * w)
        print(scale)
        h = int(h * scale)
        w = int(w * scale)
    elif h * w <= target_size * target_size * constant*2 * constant*2:
        # make h*w product equal to target_size*target_size
        scale = (target_size * target_size * constant*2) / (h * w)
        h = int((h * scale))
        w = int((w * scale))
    elif h * w <= target_size * target_size * constant*4 * constant*4:
        # make h*w product equal to target_size*target_size
        scale = (target_size * target_size * constant*4) / (h * w)
        h = int((h * scale))
        w = int((w * scale))

    # resize the image
    img = img.resize((w, h), Image.ANTIALIAS)
    return img


# h, w = 375*8, 375*8

# print(h, w, "h*w", h*w)
