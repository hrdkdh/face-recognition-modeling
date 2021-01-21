import os
import cv2

img_file_dir = "face_recorded_files"
img_file_save_dir = img_file_dir + "/" + "cleaning"

img_file_list = os.listdir(img_file_dir)
for img_file in img_file_list:
    if len(img_file.split(".")) > 1:
        img = cv2.imread(img_file_dir + "/" + img_file, cv2.IMREAD_COLOR)
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)   # YCrCb 변환
        ycrcb_ch = cv2.split(img_ycrcb)  # 색상 채널 분리
        ycrcb_ch[0] = cv2.equalizeHist(ycrcb_ch[0])   # Y 채널만 히스토그램 평활화
        dst_ycrcb = cv2.merge(ycrcb_ch)   # 색상채널 결합
        dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)    # BGR로 변환
        cv2.imwrite(img_file_save_dir + "/" + img_file.split(".")[0] + "_cleaned." + img_file.split(".")[1], dst)