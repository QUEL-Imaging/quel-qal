import numpy as np
import matplotlib.pyplot as plt


class PhantomCropper:

    def __init__(self):
        self.img = None
        self.log_img = None
        self.borders = None

    def crop_image(self, img):
        """
        Prompt user to crop region of image where the phantom is.

        :param img:         Input image
        :return:
        """

        self.img = img
        self.cvt_to_log_scale()

        y_ext, x_ext = img.shape

        fig, ax = plt.subplots(figsize=[11, 7.5])
        ax.imshow(self.log_img, cmap='gray')
        ax.set_axis_off()
        use_word = ['top', 'bottom', 'left', 'right']
        pts = []
        self.borders = {}
        for i in range(4):
            ax.set_title(f"Select a point on the {use_word[i]} edge of the phantom")
            plt.draw()
            pt = plt.ginput(timeout=-1)[0]
            pts.append(pt)
            if i < 2:
                ax.plot([0, x_ext - 1], [pt[1], pt[1]], color='r')
                self.borders[use_word[i]] = pt[1]
            else:
                ax.plot([pt[0], pt[0]], [0, y_ext - 1], color='r')
                self.borders[use_word[i]] = pt[0]
        ax.set_title("Close figure to continue")
        plt.show()

        cropped = self.log_img[int(pts[0][1]):int(pts[1][1]), int(pts[2][0]):int(pts[3][0])]
        fig, ax = plt.subplots()
        ax.imshow(cropped, cmap='gray')
        ax.set_title("Cropped region\n(Close figure to continue)")
        ax.set_axis_off()
        plt.show()

    def cvt_to_log_scale(self):
        """
        Convert input image to log scale.
        """

        uint16_max = 65535
        image = self.img.astype(float) - np.nanmin(self.img)
        image = uint16_max * (image.astype(float) / np.max(image))

        c = uint16_max / np.log(1 + np.max(image))
        log_image = c * (np.log(image + 1.0))
        self.log_img = np.array(log_image, dtype=np.uint16)
