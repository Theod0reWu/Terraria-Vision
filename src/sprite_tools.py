import matplotlib.pyplot as plt
import cv2

def display(img, path = None):
    '''
        Given a cv2 image, displays the image using pyplot if path is None, otherwise saves to path
    '''
    if (len(img.shape) > 2 and img.shape[2] == 3):
        img = img[:, :, ::-1]
    else:
        plt.gray()
    plt.axis("off")
    plt.imshow(img)
    if (not path):
        plt.show()
    else:
        plt.savefig(path)

def get_sprite(img, x, y, incr = 16, offset = 2):
    '''
        If each sprite is <incr>x<incr> with a offset of <offset> between them, get the [x,y] sprite
    '''
    start_x = x * 16 + offset * x
    start_y = y * 16 + offset * y
    return img[start_x: start_x + incr, start_y : start_y + 16]

def downsample(img):
    '''
        Halves the dimensions of the given sprite 

        Used to go from 16x16 to 8x8 sprites
    '''
    return cv2.resize(img, (0,0), fx=0.5, fy=0.5) 

def is_blank(img):
    '''
        Returns true if the sprite is blank
    '''
    return (img == 0).all()

def get_dimensions(img, incr = 16, offset = 2):
    '''
        Given the increment and offset returns the number of rows and columns in a sprite sheet
    '''
    return img.shape[0] / (incr + offset), img.shape[1] / (incr + offset)