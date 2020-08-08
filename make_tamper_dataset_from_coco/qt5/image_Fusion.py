"""
created by haoran
time : 2020-7-15

this file is to fusion the foreground and the background
"""
import cv2 as cv
class Possion:
    def __init__(self):
        pass

    def test_one_image(self, foreground, background, foreground_mask):
        height, width ,channel = foreground.shape
        center = (height / 2, width / 2)
        flags = 'cv2.MIXED_CLONE'
        output = cv.seamlessClone(foreground, background, foreground_mask, center, flags)
        return output