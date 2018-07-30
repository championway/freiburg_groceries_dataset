# PeterHuang@NCTU ARG lab

import numpy as np
import os
from PIL import Image, ImageDraw
import cv2
import xmltodict
import sys
# This function help you to parse MIT LabelMe annotation into polygon list.
# Input: xml_path(This should give xml absolute file path), product(This should give the object labeled name in LabelMe)
# Output: file_exist(0 for xml file is not exists or label be deleted, 1 for xml file exists and has labeled), xpts(list for polygon x locations in order), ypts(list for polygon y locations in order)

def xml2dict(xml_path, product):
	print xml_path
	file_exist = os.path.isfile(xml_path)    # True
	# print xml_path
	pts = []
	xpts = []
	ypts = []

	if file_exist:

		with open(xml_path) as fd:
			label_dict = xmltodict.parse(fd.read())
	else: 
		return file_exist, xpts, ypts


	
	try:
		for object_ in label_dict['annotation']['object']:	
			if object_['name'] == product  and object_['deleted'] == '0':
				pts.append(object_['polygon']['pt'])

				for pts_idx in range(0,len(object_['polygon']['pt'])):
					xpts.append(int(object_['polygon']['pt'][pts_idx]['x']))
					ypts.append(int(object_['polygon']['pt'][pts_idx]['y']))

				return file_exist, xpts,ypts

	except:

		try: ## xml exist and have label
			object_ = label_dict['annotation']['object']
			#print object_
			if object_['name'] == product  and object_['deleted'] == '0':
				
				#print object_['polygon']['pt']			
				pts.append(object_['polygon']['pt'])
				# print pts
				for pts_idx in range(0,len(object_['polygon']['pt'])):
					xpts.append(int(object_['polygon']['pt'][pts_idx]['x']))
					ypts.append(int(object_['polygon']['pt'][pts_idx]['y']))
				print "2"

				return file_exist, xpts,ypts


			else: #xml exist, labeld has been deleted
				print "xml exist, labeld has been deleted"
				xpts = []
				ypts = []
				file_exist = False
				return file_exist, xpts, ypts
			



		except: ## xml exist, but no label polygon
			xpts = []
			ypts = []
			file_exist = False
			print "no xml"
			return file_exist, xpts, ypts
	

	else:
		file_exist = False
		return file_exist, xpts, ypts

# This function help you to prepare polygon format needed by PIL.
def merge_x_y_pts(file_exist, xpts, ypts):
	mixed_x_y = []
	if file_exist:
		for pt_idx in range(0,len(xpts)):
			mixed_x_y.append(xpts[pt_idx])
			mixed_x_y.append(ypts[pt_idx])
			
	return mixed_x_y



# This function help you to return a mask, In this mask, only the pixels in bounding box area are set to 255, other pixels are set to 0.
def poly2mask(mixed_x_y, fill_value=255):
	width = 640
	height = 480
	img = Image.new('L', (width, height), 0)
	ImageDraw.Draw(img).polygon(mixed_x_y, outline=1, fill=fill_value)
	mask = np.array(img)
	return mask


# This function help you to crop mask area from original iamge.
def mask_image_crop(mask, img):
	H = 480
	W = 640
	min_x = 0
	min_y = 0
	max_x = 0
	max_y = 0

	isarea = False
	area = np.nonzero(mask)
	
	if len(area[0]) is 0 and len(area[1]) is 0:
		return isarea, area

	else:
		isarea = True
		min_x = min(area[1])
		min_y = min(area[0])
		max_x = max(area[1])
		max_y = max(area[0])
		print min_x,min_y
		print max_x,max_y
		
		crop_h = max_y - min_y
		crop_w = max_x - min_x

		crop_img = img[min_y:min_y + crop_h, min_x:min_x + crop_w]   
		return isarea, crop_img	



def main():
	xml_path = sys.argv[2]
	img_path = sys.argv[1]
	class_name = ['blue_circle','green_triangle','red_cross']
	img = cv2.imread(img_path)
	for i in range(0,3):
		cropped_path = class_name[i] + '_' + img_path
		file_exist, xpts, ypts = xml2dict(xml_path, class_name[i])
		print file_exist
		print xpts, ypts
		mixed_x_y = merge_x_y_pts(file_exist, xpts, ypts)
		mask = poly2mask(mixed_x_y, fill_value=255)
		isarea, cropped = mask_image_crop(mask, img)
		
		if isarea is True:
			print "write to %s" %(cropped_path)
			cv2.imwrite(cropped_path, cropped)


if __name__ == "__main__":
	main()
