import os
import cv2
import sys
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

def histo_normalize(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   # YCrCb 변환
    ycrcb_ch = cv2.split(img_ycrcb)  # 색상 채널 분리
    ycrcb_ch[0] = cv2.equalizeHist(ycrcb_ch[0])   # Y 채널만 히스토그램 평활화
    dst_ycrcb = cv2.merge(ycrcb_ch)   # 색상채널 결합
    dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)    # BGR로 변환
    return dst

# GPU 메모리 문제 발생 예방
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# 얼굴판별 모델 경로와 버전 설정
pgu_face_model_ver = "210122_0959"
pgu_face_model_dir = "models/pgu_face_model"

#캡처할 영역 사이즈 설정 (4:3)
face_width = 150
face_height = 200
face_area_ratio_by_width = face_height/face_width #(1.33)
face_area_ratio_by_height = face_width/face_height #(0.75)

# 얼굴인식 모델로드
model = "models/face_rec/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = "models/face_rec/deploy.prototxt"

files = [model, config]

#설정파일 있는지 체크
for f in files:
    if os.path.isfile(f) is False:
        print("얼굴인식 모델 혹은 설정파일이 없습니다 : "+f)
        sys.exit()

#얼굴인식 모델 불러오기
net = cv2.dnn.readNet(model, config)  
if net.empty():
    print("Net open failed!")
    sys.exit()
    
#openCV에서 CUDA를 사용토록 설정
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

#얼굴판별 모델 로드
model_pgu = tf.keras.models.model_from_json(open(pgu_face_model_dir + "/pgu_face_model_json_" + pgu_face_model_ver + ".json").read())
model_pgu.load_weights(pgu_face_model_dir + "/pgu_face_model_weights_" + pgu_face_model_ver + ".h5")
pgu_face_class_names_file = pgu_face_model_dir + "/pgu_face_class_names_" + pgu_face_model_ver + ".json"
with open(pgu_face_class_names_file, "r") as json_file:
    pgu_face_class_names = json.load(json_file)

cap = cv2.VideoCapture(0)   #카메라 오픈
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 카메라 프레임 원본 크기 출력
org_frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
org_frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("카메라 원본 너비 :", org_frame_width)   #카메라 가로 픽셀 출력
print("카메라 원본 높이 :", org_frame_height) # 카메라 세로 픽셀 출력

# 카메라 프레임 리사이즈
res_frame_width = 1280
res_frame_height = 720
print("리사이즈된 너비 :", res_frame_width)   #카메라 가로 픽셀 출력
print("리사이즈된 높이 :", res_frame_height) # 카메라 세로 픽셀 출력
cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_frame_height)

#창조절 가능하도록 설정
cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()   #카메라 프레임 읽어오기

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1, (300, 300), (104, 177, 123)) #이미지 크기, 컬러 평균값 지정
    net.setInput(blob) #입력 설정
    out = net.forward() #추론

    detect = out[0, 0, :, :]   # 출력 배열에서 뒷쪽 2개 차원만 활용  [200,7]
    (h, w) = frame.shape[:2]  #카메라 프레임의 크기 읽어오기 

    frame_copied = cv2.copyTo(frame, None)
    face_img = np.zeros((face_height, face_width, 3), dtype=np.uint8)
    face_img_org = face_img
    predicted_vector = None
    predicted_class = None
    label = None
    for i in range(detect.shape[0]):     #200행을 차례로 불러오기
        confidence = detect[i, 2]          # c 값을 읽어 confidence에 저장
        if confidence < 0.5:  #confidence 가 0.5보다 작을때는 skip 
            break
        x1 = int(detect[i, 3] * w)  #현재 프레임에 맞춰서 좌표 계산
        y1 = int(detect[i, 4] * h)
        x2 = int(detect[i, 5] * w)
        y2 = int(detect[i, 6] * h)

        face_height_org = y2 - y1
        face_width_org = x2 - x1

        if face_height_org < 100 or face_width_org < 75:
            continue
        
        face_height_center, face_width_center = int(y2 - ((face_height_org)/2)), int(x2 - ((face_width_org)/2))
        this_ratio = face_height_org / face_width_org
    
        face_x1 = x1
        face_x2 = x2
        face_y1 = y1
        face_y2 = y2

        if this_ratio >= face_area_ratio_by_width:
            face_width_new = int(face_height_org/face_area_ratio_by_width)
            face_width_ext_size = int((face_width_new - face_width_org)/2)
            face_x1 = x1-face_width_ext_size
            face_x2 = x2+face_width_ext_size
            face_img_org_prev = frame_copied[face_y1:face_y1+face_y2, face_x1:face_x2]
        else:
            face_height_new = int(face_width_org/face_area_ratio_by_height)
            face_height_ext_size = int((face_height_new - face_height_org)/2)
            face_y1 = y1-face_height_ext_size
            face_y2 = y2+face_height_ext_size
            face_img_org_prev = frame_copied[face_y1:face_y2, face_x1:face_x2]

        if face_y1 < 0 or face_y2 > res_frame_height or face_x1 < 0 or face_x2 > res_frame_width:
            pass
        else:
            face_img_org_prev = cv2.resize(face_img_org_prev, dsize=(face_width, face_height), interpolation=cv2.INTER_LINEAR)
            if face_img_org_prev.shape[0]/face_img_org_prev.shape[1] == face_area_ratio_by_width:
                face_img_org = histo_normalize(face_img_org_prev)
            else:
                print(face_img_org_prev.shape[0], face_img_org_prev.shape[1])

        face_img = face_img_org/255.0 #normalize
        face_img = np.expand_dims(face_img, axis=0)
        predicted_vector = model_pgu.predict(face_img)
        
        if max(predicted_vector[0]) > 0.95:
            predicted_class = model_pgu.predict_classes(face_img)
            class_no = str(predicted_class).replace("[", "").replace("]", "")
            label = pgu_face_class_names[class_no] + " : " + str(round(max(predicted_vector[0]), 2))
            cv2.putText(frame, label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)
            # cv2.imshow("face", face_img_org)  #출력
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))  # 얼굴인식 박스 그리기
    
    cv2.imshow("frame", frame)  #출력

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()