# Código de Cayetano Guerra
# coding=utf-8
from __future__ import absolute_import, division, print_function, unicode_literals
from PIL import Image
import glob
import os


# sizes = {}


def process(img):
    # width, height = img.size

    # img = img.resize((759, 759 * height/width), Image.ANTIALIAS)
    img = img.resize((380, 172), Image.ANTIALIAS)

    # width, height = img.size
    # upper = int(height/2. - 171)
    # lower = int(height/2. + 172)

    # img = img.crop((0, upper, 759, lower))

    return img

    # width, height = img.size
    # if (width, height) in sizes:
    #     sizes[(width, height)] += 1
    # else:
    #     sizes[(width, height)] = 1
    # return


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("You should specify the path to the images.\n")
    else:
        path_to_the_images = sys.argv[1]
        if not os.access(path_to_the_images, os.F_OK):
            print("The specified path is not found!!")
        else:
            if len(sys.argv) < 3:
                print("You should specify the destination path for the images.\n")
            else:
                dst_path = sys.argv[2]

                number_of_files = len(
                    [f for f in os.listdir(path_to_the_images) if os.path.isfile(os.path.join(path_to_the_images, f))])
                digits = len(str(number_of_files))

                index = 0
                for infile in glob.glob(path_to_the_images + "/*.jpg"):
                    im = Image.open(infile)
                    im = process(im)
                    dst = dst_path + "/p_img" + str(index).zfill(digits) + ".jpg"
                    im.save(dst, "JPEG")
                    index += 1

                    # print sizes
