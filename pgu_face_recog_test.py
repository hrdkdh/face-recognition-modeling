import os
import cv2
import sys 
import numpy as np
from tensorflow.keras.models import model_from_json

#얼굴판별 모델 로드
model_pgu = model_from_json(open("pgu_face_model_json.json").read())
model_pgu.load_weights("pgu_face_model_weights.h5")

img_file_dir = "face_recorded_files"
img_file_list = os.listdir(img_file_dir)

img = img_file_list[10]

this_img_org = cv2.imread(img_file_dir+"/"+img, cv2.IMREAD_COLOR)
this_img = this_img_org/255.0 #normalize
this_img = np.expand_dims(this_img, axis=0)
print(this_img.shape)
predicted_vector = model_pgu.predict(this_img)
predicted_class = model_pgu.predict_classes(this_img)
print('predicted vector', predicted_vector)
print('predicted class', predicted_class)
cv2.imshow("image", this_img_org)
cv2.waitKey(0)