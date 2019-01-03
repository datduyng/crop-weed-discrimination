import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import sys
import re
import os

if(len(sys.argv) != 2):
	print("def video path") 
	quit()

video = cv2.VideoCapture(sys.argv[1])

print('current fps', video.get(cv2.CAP_PROP_FRAME_COUNT))
video.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
video.set(cv2.CAP_PROP_FPS, 1)

video_name = sys.argv[1].split('/')[-1].partition('.')[0]
print(video_name)
os.makedirs('../crop-image/%s/'%(video_name),exist_ok=True)
count = 0
while(video.isOpened()):
	ret, frame = video.read()
	frame=cv2.transpose(frame)
	frame = cv2.resize(frame, (600, 800))
	cv2.imshow('Playing-%s'%(video_name),frame)
	cv2.imwrite("../crop-image/%s/frame%d.jpg"%(video_name, count) , frame)
	count += 1 
	if cv2.waitKey(10) & 0xFF == ord('q'):
		break


cap.release()
cv2.destroyAllWindows()  # destroy all the opened windows