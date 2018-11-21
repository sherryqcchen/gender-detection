import cv2
import os
import matplotlib.pyplot as plt

img_path = '../data/trainset/male'
# img_path = 'all/'
model_path = 'trained-models/'
models = [
# 'haarcascades/haarcascade_frontalface_alt_tree.xml',
# 'haarcascades/haarcascade_frontalface_alt.xml',
# 'haarcascades/haarcascade_frontalface_alt2.xml',
'haarcascades/haarcascade_frontalface_default.xml',
# 'haarcascades/lbpcascade_profileface.xml'
]

whole_img_names = [x for x in filter(lambda filename: filename.endswith('jpeg'),os.listdir(img_path))]
face_cascades = [cv2.CascadeClassifier(model_path+x) for x in models]

print(len(whole_img_names))

def detect(length):
	img_names = whole_img_names[:length]
	total = len(img_names)
	count = 0
	for img_name in img_names:
		img = cv2.imread(img_path+img_name)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		max_grades = 0
		for face_cascade in face_cascades:
			faces = face_cascade.detectMultiScale(
			    gray,
			    # 1.3,
			    # 5
			    scaleFactor=1.1,
			    minNeighbors=4,
			    minSize=(20, 20),
			    flags = cv2.CASCADE_SCALE_IMAGE
			)
			if len(faces) > max_grades:
				max_grades = len(faces)
		if max_grades > 0:
			count+=1

	print('tatal:{0},faces:{1},percentage:{2}'.format(total,count,count/total))
# detect(len(whole_img_names))
detect(10)
