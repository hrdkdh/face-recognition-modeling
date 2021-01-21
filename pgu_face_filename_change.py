import os

img_file_dir = "face_recorded_files"
img_file_list = os.listdir(img_file_dir)
for img in img_file_list:
    os.rename(img_file_dir+"/"+img, img_file_dir+"/"+img.replace("baekhoon", "baek hoon"))