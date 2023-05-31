import cv2
import numpy as np

import tkinter as tk
from PIL import ImageTk, Image

from model import Model

def convert_to_bw(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to obtain a binary image
    _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Invert the binary image to have black letter on white background
    inverted_image = cv2.bitwise_not(binary_image)

    # Convert the image to 3-channel format
    inverted_image = cv2.cvtColor(inverted_image, cv2.COLOR_GRAY2BGR)

    # Invert the color of the image
    inverted_image = cv2.bitwise_not(inverted_image)

    # Display the original and processed images
    return inverted_image


class Drawer:
    def __init__(self):
        self.mouse_pressed = False
        self.img = np.zeros(shape=(1024, 1024, 3), dtype=np.uint8)
        self.char_color = (255, 255, 255)

    def get_contours(self):
        """
        Method to find contours in an image and crop them and return a list with cropped contours
        """

        images = []
        main_image = self.img
        orig_image = main_image.copy()

        # convert to grayscale and apply Gaussian filtering
        main_image = cv2.cvtColor(src=main_image, code=cv2.COLOR_BGR2GRAY)
        main_image = cv2.GaussianBlur(src=main_image, ksize=(5, 5), sigmaX=0)

        # threshold the image
        _, main_image = cv2.threshold(src=main_image, thresh=127, maxval=255, type=cv2.THRESH_BINARY)

        # find contours in the image
        contours, _ = cv2.findContours(image=main_image.copy(), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

        # get rectangles containing each contour
        bboxes = [cv2.boundingRect(array=contour) for contour in contours]

        for bbox in bboxes:
            x, y, width, height = bbox[:4]
            images.append(orig_image[y:y + height, x:x + width])

        return images

    def load_image(self, image_path):
        self.img = cv2.imread(image_path)

    def get_images(self, image_path):
        images = []

        self.load_image(image_path)

        char_images = self.get_contours()

        for cimg in char_images:
            images.append(Drawer.convert_to_emnist(img=cimg))

        return images

    @staticmethod
    def convert_to_emnist(img):
        """
        Method to make an image EMNIST format compatible. img is a cropped version of the character image.

        Conversion process available in section II-A of the EMNIST paper available at https://arxiv.org/abs/1702.05373v1
        """

        # Resize the image to 28x28 pixels
        img = cv2.resize(src=img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

        return img


image_path = "./images/T.png"

if __name__ == '__main__':
    images = Drawer().get_images(image_path)

    for image in images:
        image = convert_to_bw(image)
        label = Model().predict(img=image)
        cv2.imshow(winname=label, mat=image)
        cv2.waitKey(delay=0)
        cv2.destroyAllWindows()