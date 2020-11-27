"""
created by haoran
time：11/11
description:

"""
from PIL import Image
import os,sys
import numpy as np
import matplotlib.pylab as plt
import cv2 as cv
import random
import traceback
class Splicing:
    def __init__(self):
        pass
    def gen_two_image_line_splicing(self):
        pass

    def __line_splicing(self, a_image_path, b_image_path):
        """
        input two image path, using line splicing method to splicing two image
        :return: numpy,[tamper image,gt]
        """
        if Splicing.__input_path_check(self,path=a_image_path) and Splicing.__input_path_check(self,b_image_path):
            a_image = Image.open(a_image_path)
            b_image = Image.open(b_image_path)

            # choose a random line

            pass
        else:
            traceback.print_exc('The path is un correct!')


    def __curve_splicing(self):
        """
        input two image path, using curve splicing method to splicing two image
        :return: numpy,[tamper image,gt]
        """


    def __input_path_check(self, path):
        """
        robust method, before input,check it
        :return: BOOL
        """
        if os.path.exists(path):
            return True
        else:
            return False

    def __image_requirement(self):
        pass

    def __gen_random_line_or_curve(self, image_size=320):
        """
        gen random line or curve
        :return:two point(row,col)
        """
        # setting range
        start = 80
        end = image_size - 80

        # choose one side,left,bottom,right top 1,2,3,4
        sides = [1,2,3,4]
        side_1 = random.randint(1, 4)
        another_sides = sides.remove(side_1)
        # random choose another side
        side_2 = random.sample(another_sides, 1)

        # using one side to random choose a point
        random_number_1 = random.randint(start,end)
        point_a = (160, 160)
        point_b = (160, 160)

        if side_1 == 1:
            point_a = (random_number_1, 0)
        elif side_1==2:
            point_a = (image_size, random_number_1)
        elif side_1 ==3:
            point_a = (random_number_1,image_size)
        elif side_1 ==4:
            point_a = (0, random_number_1)
        else:
            traceback.print_exc('when choose sides, an error occur!')

        # base on the point, choose another point
        random_number_2 = random.randint(start,end)
        if side_2 == 1:
            point_b = (random_number_2, 0)
        elif side_2 == 2:
            point_b = (image_size, random_number_2)
        elif side_2 == 3:
            point_b = (random_number_2,image_size)
        elif side_2 == 4:
            point_b = (0, random_number_2)
        else:
            traceback.print_exc('when choose sides, an error occur!')
        return point_a, point_b


if __name__ == '__main__':
    pass