import os
import cv2
import sys 
import numpy as np
from datetime import datetime

def histo_normalize(img):
    img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   # YCrCb 변환
    ycrcb_ch = cv2.split(img_ycrcb)  # 색상 채널 분리
    ycrcb_ch[0] = cv2.equalizeHist(ycrcb_ch[0])   # Y 채널만 히스토그램 평활화
    dst_ycrcb = cv2.merge(ycrcb_ch)   # 색상채널 결합
    dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)    # BGR로 변환
    return dst
    
def onMouse(event, x, y, flags, param):
    # print(event,x,y)   #마우스 이벤트 출력
    if event == cv2.EVENT_LBUTTONDOWN and face_img is not None:   #마우스 왼쪽 버튼을 클릭하면 현재 프레임 저장
        this_datetime = str(datetime.now().timestamp()).replace(".", "")
        cv2.imwrite("face_recorded_files/" + face_name + "_" + this_datetime + ".png", face_img)

face_name = input("이름을 입력하세요 :")

#캡처할 영역 사이즈 설정 (4:3)
face_width = 150
face_height = 200
face_area_ratio_by_width = face_height/face_width #(1.33)
face_area_ratio_by_height = face_width/face_height #(0.75)

# caffe, tensorflow 모델을 차례로 테스트 해볼것
model = "models/face_rec/res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = "models/face_rec/deploy.prototxt"

files = [model, config]

#설정파일 있는지 체크
for f in files:
    if os.path.isfile(f) is False:
        print("모델 혹은 설정파일이 없습니다 : "+f)
        sys.exit()

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
res_frame_width = 640
res_frame_height = 480
print("리사이즈된 너비 :", res_frame_width)   #카메라 가로 픽셀 출력
print("리사이즈된 높이 :", res_frame_height) # 카메라 세로 픽셀 출력
cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_frame_height)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)

net = cv2.dnn.readNet(model, config)  #모델 불러오기
if net.empty():
    print("Net open failed!")
    sys.exit()

#CUDA를 사용토록 설정
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

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
            face_img_prev = frame_copied[face_y1:face_y1+face_y2, face_x1:face_x2]
        else:
            face_height_new = int(face_width_org/face_area_ratio_by_height)
            face_height_ext_size = int((face_height_new - face_height_org)/2)
            face_y1 = y1-face_height_ext_size
            face_y2 = y2+face_height_ext_size
            face_img_prev = frame_copied[face_y1:face_y2, face_x1:face_x2]

        if face_y1 < 0 or face_y2 > res_frame_height or face_x1 < 0 or face_x2 > res_frame_width:
            pass
        else:
            face_img_prev = cv2.resize(face_img_prev, dsize=(face_width, face_height), interpolation=cv2.INTER_LINEAR)
            if face_img_prev.shape[0]/face_img_prev.shape[1] == face_area_ratio_by_width:
                face_img = histo_normalize(face_img_prev)
            else:
                print(face_img_prev.shape[0], face_img_prev.shape[1])

        label = f"Face: {confidence:4.2f}"    #확률값 출력 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0))  # 박스 그리기        
        cv2.putText(frame, label, (x1, y1 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, "left click to shot", (10, 20 - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
    
    cv2.imshow("face", face_img)  #출력
    cv2.imshow("frame", frame)  #출력
    #마우스 이벤트 함수 호출
    cv2.setMouseCallback("frame", onMouse)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()