from random import randint

from PIL import Image


class PairedRandomCrop:
    def __init__(self, size:int | tuple[int, int]):
        """
        :param size:  crop size (height, width)
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError("size should be a tuple (h, w) or an integer")
            if size[0] != size[1]:
                print("Please provide a square size for cropping")

            self.size = size

        self.h = self.size[0]
        self.w = self.size[1]

    def __call__(self, image: Image.Image, mask: Image.Image):

        # i dont see this happening, but just to be safe
        if image.size != mask.size:
            raise ValueError("Image and mask should be the same size")#

        image_w, image_h = image.size
        if image_w < self.w or image_h < self.h:
            raise ValueError("Crop size is too large for the image")

        left = randint(0, image_w - self.w)
        top = randint(0, image_h - self.h)
        box = (left, top, left + self.w, top + self.h)

        return image.crop(box), mask.crop(box)