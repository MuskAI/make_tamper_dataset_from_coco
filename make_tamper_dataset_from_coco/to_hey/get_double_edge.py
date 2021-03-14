import numpy as np
import skimage.morphology as dilation
def __mask_to_double_edge(self, orignal_mask):
    """
    :param orignal_mask: 输入的是 01 mask图
    :return: 255 100 50 mask 图
    """
    # print('We are in mask_to_outeedge function:')
    try:
        mask = orignal_mask
        # print('the shape of mask is :', mask.shape)
        selem = np.ones((3, 3))
        dst_8 = dilation.binary_dilation(mask, selem=selem)
        dst_8 = np.where(dst_8 == True, 1, 0)
        difference_8 = dst_8 - orignal_mask

        difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
        difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
        double_edge_candidate = difference_8_dilation + mask
        double_edge = np.where(double_edge_candidate == 2, 1, 0)
        ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(
            mask == 1, 50, 0)  # 所以内侧边缘就是100的灰度值
        ground_truth = np.where(ground_truth == 305, 255, ground_truth)
        ground_truth = np.array(ground_truth, dtype='uint8')
        return ground_truth

    except Exception as e:
        print(e)