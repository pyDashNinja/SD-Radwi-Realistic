import cv2
from PIL import Image


def inference(face_enhancer, img):
    try:
        if len(img.shape) == 3 and img.shape[2] == 4:
            img_mode = 'RGBA'
        else:
            img_mode = None

        h, w = img.shape[0:2]
        if h < 300:
            img = cv2.resize(img, (w * 2, h * 2),
                             interpolation=cv2.INTER_LANCZOS4)

        if img_mode == 'RGBA':
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            _, _, output = face_enhancer.enhance(
                img, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as error:
            print('Error', error)

        if img_mode == 'RGBA':
            output = cv2.cvtColor(output, cv2.COLOR_BGRA2RGBA)
            output = Image.fromarray(output, mode=img_mode)
        else:
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            output = Image.fromarray(output)

        return output
    except Exception as error:
        print('global exception', error)
        return None, None
