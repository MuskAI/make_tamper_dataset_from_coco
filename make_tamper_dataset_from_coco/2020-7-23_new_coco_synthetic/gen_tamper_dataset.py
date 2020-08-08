from pycocotools.coco import COCO
import numpy as np
import cv2
import skimage.io as io
import matplotlib.pyplot as plt
import matplotlib
import pylab
import os
import sys
import time
from PIL import Image
from PIL import ImageFilter
import argparse
import sys
import pdb
from get_double_edge import mask_to_outeedge
from image_Fusion import Possion
import poisson_image_editing
import skimage.morphology as dilation
import traceback
import myCalImage

matplotlib.use('Qt5Agg')
def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='input begin and end category')
  parser.add_argument('--begin', dest='begin',
            help='begin type of cat', default=None, type=int)
  parser.add_argument('--end', dest='end',
            help='begin type of cat',
            default=None, type=int)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args




def splicing_tamper_one_image(input_foreground, input_background,mask, similar_judge = True):
	"""
	在输入这个函数之前，输入的两个参数需要以及经过挑选了的,需要import poisson融合的代码
	:param foreground:
	:param background:
	:return: 返回两个参数：直接篡改图, poisson融合篡改图, GT
	"""

	I = input_foreground
	I1 = I
	# mask 是 01 蒙版
	I1[:, :, 0] = np.array(I[:, :, 0] * mask)
	I1[:, :, 1] = np.array(I[:, :, 1] * mask)
	I1[:, :, 2] = np.array(I[:, :, 2] * mask)
	# differece_8是background的edge
	difference_8 = mask_to_outeedge(mask)


	difference_8_dilation = dilation.binary_dilation(difference_8, np.ones((3, 3)))
	difference_8_dilation = np.where(difference_8_dilation == True, 1, 0)
	double_edge_candidate = difference_8_dilation + mask
	double_edge = np.where(double_edge_candidate == 2, 1, 0)
	ground_truth = np.where(double_edge == 1, 255, 0) + np.where(difference_8 == 1, 100, 0) + np.where(mask == 1, 50,0)  # 所以内侧边缘就是100的灰度值
	b1 = input_background

	background = Image.fromarray(b1, 'RGB')
	foreground = Image.fromarray(I1, 'RGB').convert('RGBA')
	datas = foreground.getdata()

	newData = []
	for item in datas:
		if item[0] == 0 and item[1] == 0 and item[2] == 0:
			newData.append((0, 0, 0, 0))
		else:
			newData.append(item)
	foreground.putdata(newData)
	background = background.resize((foreground.size[0], foreground.size[1]),Image.ANTIALIAS)
	# input_background = np.resize(input_background,(foreground.size[1], foreground.size[0],3))
	background_area = np.array(background)

	try:

		if similar_judge:
			foreground_area = I1
			background_area[:, :, 0] = np.array(background_area[:, :, 0] * mask)
			background_area[:, :, 1] = np.array(background_area[:, :, 1] * mask)
			background_area[:, :, 2] = np.array(background_area[:, :, 2] * mask)

			foreground_area = Image.fromarray(foreground_area)
			background_area = Image.fromarray(background_area)

			if myCalImage.calc_similar(foreground_area,background_area)>0.4:
				pass
			else:
				return False,False,False
	except Exception as e:
		print(e)
		traceback.print_exc()


	try:
		mask = Image.fromarray(mask)
	except Exception as e:
		print('mask to Image error', e)
	# 在这里的时候，mask foreground background 尺寸都是一致的了，poisson融合时，offset置为0
	try:
		poisson_foreground = cv2.cvtColor(np.asarray(foreground.convert('RGB')), cv2.COLOR_RGB2BGR)
		poisson_background = cv2.cvtColor(np.asarray(background), cv2.COLOR_RGB2BGR)
		poisson_mask = np.asarray(mask)
		poisson_mask = np.where(poisson_mask == 1, 255, 0)
		poisson_fusion_image = poisson_image_editing.poisson_fusion(poisson_foreground, poisson_background,poisson_mask)
		poisson_fusion_image = Image.fromarray(cv2.cvtColor(poisson_fusion_image, cv2.COLOR_BGR2RGB))
		background.paste(foreground, (0, 0), mask=foreground.split()[3])
		return background, poisson_fusion_image, ground_truth
	except Exception as e:
		traceback.print_exc()


def judge_required_image(area= None, f_size = None,b_size =None, min_area = 0, max_area = 99999,size_threshold = 0.5 ):

	try:
		if area == None or f_size == None or b_size == None:
			return True
		else:
			pass

		if area >=min_area and area <= max_area:
			return True
		else:
			return False

		if b_size[0] > f_size[0] * (1 - size_threshold) and b_size[0] < f_size[0] * (1 + size_threshold) \
				and b_size[1] > f_size[1] * (1 - size_threshold) and b_size[1] < f_size[1] * (1 + size_threshold):
			return True
		else:
			return False

	except Exception as e:
		traceback.print_exc()



def judge_area_similar(foreground_area, background_area, similar_threshold = 0.5):
	"""
	判断两个区域是否相似，这有利于poisson编辑的效果，这只是固定位置判断，输入的都是两个mask区域

	1.判断直方图
	2. 判断
	:param foreground_area:
	:param background_area:
	:param similar_threshold:有百分之多少的相似 默认 0.5
	:return: 返回一个bool数,表示该选择的预期ok否
	"""

	# 直方图
	similar_score = myCalImage.calc_similar(foreground_area,background_area)

	print('直方图相似程度:', similar_score)
	if similar_score > 0.5 * 100:
		return True
	else:
		return False

# def paste_object_to_background(foreground, background, mask, bbox, tamper_num = 1, optimize = False ):
# 	"""
# 	拿一张前景图 和mask 图 将mask区域paste到background上
# 	:param foreground:
# 	:param background:
# 	:param mask:
# 	:param tamper_num: return 多少张篡改图——位置不一样 ，默认为1
# 	:param optimize: 是否使用优化算法，寻找最佳区域
# 	:return:
# 	"""
# 	try:
# 		# 把特定区域的object给弄出来
# 		object_area = foreground
# 		object_area[:, :, 0] = np.array(object_area[:, :, 0] * mask)
# 		object_area[:, :, 1] = np.array(object_area[:, :, 1] * mask)
# 		object_area[:, :, 2] = np.array(object_area[:, :, 2] * mask)
#
# 		cut_mask = mask[bbox[1]:bbox[3],bbox[0]:bbox[2]]
# 		cut_object = object_area[bbox[1]:bbox[3],bbox[0]:bbox[2]]
#
#
# 		# 以左上角的点作为参考点，计算可以paste的区域
# 		background_size = background.size
# 		paste_area = []
#
# 		# 如果没有optimize则随机的选取一块区域paste上去
# 		if len(paste_area) < 100:
# 			print('无可供选择区域')
# 			return
#
#
# 		tamper_image = []
# 		if optimize == False :
# 			for times in tamper_num:
# 				random_x = np.random.randint(paste_area[])
# 				random_y = np.random.randint(paste_area[])
# 				padding_mask = np.pad()
# 				padding_object_area = np.pad()
#
# 				background.paste(padding_object_area, (0, 0), mask=padding_mask)
# 				tamper_image.append(background)
#
# 			return tamper_image
#
# 		else:
# 			# 暂时先空着
# 			pass
#
#
# 	except Exception as e:
# 		traceback.print_exc()
#

def random_area_to_background(foreground, background, mask,):
	"""
	similar to up
	:param foreground:
	:param background:
	:param mask:
	:return:
	"""


def image_save_method(src, tamper_image, img = None, img1 = None, tamper_poisson = None, ground_truth=None, save_path=None, foreground_name=None, background_name=None, cat = None):
	"""

	:param src:
	:param tamper_image:
	:param img:
	:param img1:
	:param tamper_poisson:
	:param ground_truth:
	:param save_path:
	:param foreground_name:
	:param background_name:
	:param cat:
	:return:
	"""
	try:
		if save_path ==None:
			print('请输入保存root路径')
		if os.path.exists(save_path) == False:
			print('请手动创建数据集root目录')

		src_path = 'src'
		tamper_path = 'tamper'
		tamper_poisson_path = 'tamper_poisson'
		ground_truth_path = 'ground_truth'
		os.makedirs(os.path.join(save_path,src))
		os.makedirs(os.path.join(save_path,tamper_path))
		os.makedirs(os.path.join(save_path,tamper_poisson_path))
		os.makedirs(os.path.join(save_path,ground_truth_path))


		image_format = ['.jpg', '.png', '.bmp']
		tptype = ['Default', 'poisson']
		save_name_part2 = tptype[0] + str(img['id']) + '_' + str(img1['id']) + '_' + cat + image_format[0]
		save_name_part1 = {'src':os.path.join(save_path, src_path),'tamper':os.path.join(save_path, tamper_path),
						   'tamper_poisson':os.path.join(save_path, tamper_poisson_path),'ground_truth':os.path.join(save_path, ground_truth_path)}
		save_name = {'src':os.path.join(save_name_part1['src'],save_name_part2),'tamper':os.path.join(save_name_part1['tamper'],save_name_part2),
					 'tamper_poisson':os.path.join(save_name_part1['tamper_poisson'],save_name_part2), 'ground_truth':os.path.join(save_name_part1['ground_truth'],save_name_part2)}
		if not os.path.isfile(save_name['tamper']):
			print(save_name['tamper'])
			cv2.imwrite(save_name['tamper_poisson'], ground_truth)
		if not os.path.isfile(save_name['tamper_poisson']):
			print(save_name['tamper_poisson'])
			cv2.imwrite(save_name['tamper_poisson'], ground_truth)
		if not os.path.isfile(save_name['ground_truth']):
			print(save_name['ground_truth'])
			cv2.imwrite(save_name['ground_truth'], ground_truth)
		if not os.path.isfile(save_name['tamper']):
			print(save_name['tamper'])
			cv2.imwrite(save_name['tamper'], ground_truth)

	except Exception as e:
		traceback.print_exc()
		sys.exit(0)
	return True

def main():
	cycle_flag = 0
	pylab.rcParams['figure.figsize'] = (10.0, 8.0)
	dataDir = 'D:\\实验室\\图像篡改检测\\数据集\\COCO\\'
	# dataDir = '/media/musk/File/实验室/图像篡改检测/数据集/COCO'
	dataType = 'val2017'
	annFile = '%s/annotations/instances_%s.json' % (dataDir, dataType)
	coco = COCO(annFile)
	cats = coco.loadCats(coco.getCatIds())
	for cat in cats[5:80]:
		for num in range(20):
			try:

				catIds = coco.getCatIds(catNms=[cat['name']])
				imgIds = coco.getImgIds(catIds=catIds)
				img = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]

				annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
				anns = coco.loadAnns(annIds)
				img1 = coco.loadImgs(imgIds[np.random.randint(0, len(imgIds))])[0]
				bbx = anns[0]['bbox']


				# 判断随机出来的两幅图像符不符合要求
				if judge_required_image(anns[0]['area'], (img['height'],img['width']),(img1['height'],img1['width']),1000, size_threshold=0.2):
					cycle_flag += 1
					print('循环的次数为:', cycle_flag)
					if cycle_flag >=  50:
						print('50 times 循环')
					elif cycle_flag >= 30:
						print('30 times 循环')
					elif cycle_flag >10:
						print('10 times 循环')
					elif cycle_flag >3:
						print('循环的次数为：%d', cycle_flag)
					else:
						pass
					pass
				else:
					continue



				I = io.imread(os.path.join(dataDir, dataType, '{:012d}.jpg'.format(img['id'])))
				b1 = io.imread(os.path.join(dataDir, dataType, '{:012d}.jpg'.format(img1['id'])))
				mask = np.array(coco.annToMask(anns[0]))
				#
				# plt.figure('看读取的第一张')
				# plt.imshow(I)
				# plt.show()
				#
				# plt.figure('看读取的第二张')
				# plt.imshow(b1)
				# plt.show()



				tamper_raw_image, tamper_poisson_image, ground_truth = splicing_tamper_one_image(I,b1,mask)
				if tamper_raw_image ==False:
					continue
				cycle_flag =0








				cv2.imwrite(
					'../ground_truth/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(bbx[0]) + '_' + str(
						bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' + cat['name'] + '.bmp',
					ground_truth)

				if not os.path.isfile('../filter_tamper2/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(
						bbx[0]) + '_' + str(bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' +
									  cat['name'] + '.bmp'):
					print('../filter_tamper2/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(
						bbx[0]) + '_' + str(bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' +
						  cat['name'] + '.png')

					tamper_raw_image.save('../filter_tamper2/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(
						bbx[0]) + '_' + str(bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' +
									cat['name'] + '.bmp')
					tamper_poisson_image.save(
						'../filter_tamper2_poisson_fusion/Tp_' + str(img['id']) + '_' + str(img1['id']) + '_' + str(
							bbx[0]) + '_' + str(
							bbx[1]) + '_' + str(bbx[0] + bbx[2]) + '_' + str(bbx[1] + bbx[3]) + '_' + cat[
							'name'] + '.bmp')
			except Exception as e:
				print(e)
	print('finished')


if __name__ == '__main__':
	main()