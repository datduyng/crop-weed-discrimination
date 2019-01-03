import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import os
import math


def overlap_area(box1, box2):
	'''
	box1: coordinate of box1(x,y,w,h) 
	box2: coor of box 2 
	return the overlaping area
	'''
	#unpack variable 
	x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
	x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
	
	w =  min(x1+w1, x2+w2)- max(x1, x2)
	h = min(y1+h1, y2+h2) - max(y1, y2)
	if(w < 0 or h < 0):
		return -1 # not overlap 
	return w * h # Overlap area


def is_enclosed(box1, box2): 
	'''
	assume rect is sort in x, box1.x < box2.x(only box2 can be enclosed in box1)
	if box1 > box2(box2 enclosed in box1) -> return 1 
	else: 0

	check if 2 box is 
	'''
	#unpack variable 
	x1, y1, w1, h1 = box1[0], box1[1], box1[2], box1[3]
	x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
	#in other 2 box enclosed it need to be overlap as well
	if(x1+w1 >= x2+w2 and y1+h1 >= y2+h2 and overlap_area(box1, box2) > -1):# check box x and y 
		return True
	else: return False
	


def remove_overlap(boxes, axis=0, overlap_ratio=0.01): 
	'''
	boxes: list of tuple coordinate boxes[i] = (xi, yi, wi, hi) 
	overlap_ratio: if 2 boxes overlap area > one of the boxes area * overlap_ratio then remove that box
	This algorithm asumme that boxes is sorted by X correspond with boxes[0]
	'''
	i, j = 0, 0 
	if(len(boxes) == 1): return None
	while(j < len(boxes)-1 and i < len(boxes)): 
		if(i == j): i += 1 
		start = boxes[i][axis] 
		end = boxes[j][axis] + boxes[j][axis+2] #x + w 
		if(start < end): #check for overlap in x-dir
			#Unpack variable 
			xj, yj, wj, hj = boxes[j][0], boxes[j][1], boxes[j][2], boxes[j][3]
			xi, yi, wi, hi = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
			
			if(is_enclosed(boxes[j], boxes[i])):#assume it only possible if boxi enclosed in boxj
				boxes.remove(boxes[i]) 
				i = j
				continue 
			#check for overlap in y-dir  as well
			if(overlap_area(boxes[i], boxes[j]) > overlap_ratio * wi * hi or 
			   overlap_area(boxes[i], boxes[j]) > overlap_ratio * wj * hj):
				#update new bouned box on boxj 
				#tuple is immutable therefore, cast it to list
				#reassign value then cast back to tuple
				box_join = list(boxes[j])
				box_join[0] = min(xi, xj)#X component 
				box_join[1] = min(yi, yj)#y component 
				box_join[2] = max(xi + wi, xj + wj) - box_join[0] #width
				box_join[3] = max(yi + hi, yj + hj) - box_join[1] # height
				boxes[j] = tuple(box_join)
				
				boxes.remove(boxes[i])
				i = j
				continue # we don't want to update i since removing element. index i will hold new element
			i += 1
		elif(start >= end): 
			i = j #reset counter so start checking down from j
			j += 1
def mask_plant(frame, plant_thresh=200): 
	'''
	Remove red and green component of an image 
	then get a mask of the image 
	'''
	# frame = cv2.resize(frame, (400, 400))
	p_blue, p_green, p_red= cv2.split(frame) # For BGR image # For RGB image

	#Subtract out red and green channel out of image 
	p_excess_green = 128 + np.int16(p_green) - np.int16(p_blue) + np.int16(p_green) - np.int16(p_red)
	p_excess_green = np.uint8(np.clip(p_excess_green, 0, 255))

	mask = np.uint8((p_excess_green > plant_thresh)*1)
	return mask

def getBoundingBox(frame, overlap_ratio=0.5): 
	'''
	Return Min bounding box of plants in images given mask image 
	boxes: contain boxes with no enclosed box and overlap box with overlap_ratio threshhold
	'''
	#Find contour
	mask = mask_plant(frame) 
	_, contours, hierachy = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	#Get all bounding boxes 
	boxes = []
	for c in contours: 
		x,y,w,h = cv2.boundingRect(c)
		if(w > 20 and h > 20):#don't box item if it is too small 
			boxes.append((x,y,w,h))#(x,y,w,h)
	#sort boxes by x 
	boxes.sort(key=lambda tup: tup[0])  # sorts in place
	remove_overlap(boxes, axis=0, overlap_ratio=overlap_ratio)

	boxes.sort(key=lambda tup: tup[1])  # sorts in place by y
	remove_overlap(boxes, axis=1, overlap_ratio=overlap_ratio)

	return boxes, mask

def translate(box, origin_h_w=(400,400), scale_h_w=(800,600)):
    '''
    box: tuple of shape (x,y,w,h) 
    origin_h_w: tuple shape (h, w)  
    scale_h_w: tupple shape (h, w)to scale
    return new box with given(height, width) scale
    '''
    h_scale, w_scale = scale_h_w[0]/origin_h_w[0], scale_h_w[1]/origin_h_w[1]
    return (int(box[0]*w_scale), int(box[1]*h_scale), int(box[2]*w_scale), int(box[3]*h_scale))

def find_centroid(mask):#crop by h, w
	h, w = mask.shape
	#find centroid 
	x_axis = np.arange(0,w,1)

	y_axis = np.arange(0,h,1) 
	x_v, y_v = np.meshgrid(x_axis, y_axis)

	Mx = x_v[mask.astype('bool')].mean()
	My = y_v[mask.astype('bool')].mean()
	return Mx, My
	
	
def crop_box(box, mask, x_fit = 200, y_fit = 200, y_drop_ratio=0.5, x_drop_ratio=0.5):# process image is 400,400
	'''
	mask: mask image
	'''
	x, y, w, h = box[0],box[1], box[2], box[3]
	x_fit = min(x_fit, w) 
	y_fit = min(y_fit, h)
	
	n_w = w // x_fit if (w // x_fit) is not 0 else 1 
	n_h = h // y_fit if (h // y_fit) is not 0 else 1 
	
	
	#take care of last box indepedently 
	last_region = (x+x_fit*n_w, y+y_fit*n_h, w%x_fit, h%y_fit)
	last_w_margin = max(w - x_fit*n_w, 0)
	last_h_margin = max(h - y_fit*n_h, 0)
	
	cropw = x_fit
	croph = y_fit 
	
	crop_region = []
	for xi in range(n_w + (1 if (last_w_margin > x_fit*x_drop_ratio) else 0)):
		for yi in range(n_h + (1 if(last_h_margin > y_fit*y_drop_ratio) else 0)):
			if(xi==n_w and (last_w_margin > x_fit*x_drop_ratio)):cropw = last_w_margin
			else: cropw = x_fit
				
			if(yi==n_h and (last_h_margin > y_fit*y_drop_ratio)):croph = last_h_margin 
			else: croph = y_fit
				
			crop_region.append([int(x+x_fit*xi), int(y+y_fit*yi), cropw, croph])
	
	#     find centroid and shift the box to the centroid 
	for i in range(len(crop_region)):
		x_, y_, w_, h_ = crop_region[i]
		origin_comx, origin_comy = (w_/2)+x_, (h_/2)+y_
		new_comx, new_comy = find_centroid(mask[y_:y_+h_, x_:x_+w_])
		new_comx = origin_comx if(math.isnan(new_comx)) else new_comx
		new_comy = origin_comy if(math.isnan(new_comy)) else new_comy

		shift_x, shift_y = (x_+new_comx - origin_comx), (y_+new_comy - origin_comy)
		crop_region[i][0] +=  int(shift_x)
		crop_region[i][1] +=  int(shift_y)

	return crop_region
	




if __name__ == "__main__":
	if(len(sys.argv) < 2):
		print('NO Video path define') 
		quit() 

	video = cv2.VideoCapture(sys.argv[1].replace('\'',''))
	video_name = os.path.basename(sys.argv[1]).split('.')[0]

	print('current fps', video.get(cv2.CAP_PROP_FRAME_COUNT))
	WIDTH = 480 
	HEIGHT = 640
	video.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
	video.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
	video.set(cv2.CAP_PROP_FPS, 1)
	print('now fps', video.get(cv2.CAP_PROP_FRAME_COUNT))

	font                   = cv2.FONT_HERSHEY_DUPLEX
	bottomLeftCornerOfText = (10,500)
	fontScale              = 0.4
	fontColor              = (255,255,0)
	lineType               = 2

	##Option for saving video
	out = None
	if(len(sys.argv) == 3):
		if(sys.argv[2] == 'save'):
			out = cv2.VideoWriter('../crop-image/saved-video/'+video_name+'.avi',
				cv2.VideoWriter_fourcc('M','J','P','G'), 30, (WIDTH,HEIGHT))

	while(video.isOpened()):
		ret, frame = video.read() #ret: bool
		frame = cv2.transpose(frame)
		frame = cv2.resize(frame,(WIDTH, HEIGHT))
		boxes, mask = getBoundingBox(frame, overlap_ratio=0.5)
	#     #Draw bounding box on frame
		for box in boxes:
			box = crop_box(box, mask, x_fit = 100, y_fit = 100, y_drop_ratio= 0.1, x_drop_ratio = 0.1)
			for region in box: 
				region = translate(region,origin_h_w=(400,400), scale_h_w=(HEIGHT,WIDTH))
				cv2.rectangle(frame,pt1=(region[0],region[1]), 
									pt2=(region[0]+region[2],region[1]+region[3]), 
									color=(0,0,255), thickness = 1,lineType=4)# 4: 4 edges
				cv2.putText(frame,'Plant/Weed:..%', 
								(region[0]+30,region[1]+30), # bottom left corner
								font, 
								fontScale,
								fontColor,
								lineType)
		if(len(sys.argv) == 3):
			if(sys.argv[2] == 'save'):
				out.write(frame)
		# for box in boxes:
		# cv2.rectangle(frame,(box[0],box[1]), (box[0]+box[2], box[1]+box[3]), (0,0,255), 4)# 4: 4 edges

		cv2.imshow('Field',frame)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			break
# When everything done, release the video capture and video write objects
cap.release()
out.release() if out is not None else print()
 
# Closes all the frames
cv2.destroyAllWindows() 