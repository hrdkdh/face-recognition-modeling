import os
import cv2
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow import keras
from tensorflow.python.client import device_lib
from sklearn.model_selection import train_test_split

# GPU 메모리 문제 발생 예방
config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.5
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
tf.compat.v1.disable_eager_execution() #GPU 병렬사용을 하기 위해서는 설정해 주어야 함

#libcusolver.so.10 파일이 없어 임의로 복사해 옴
#sudo cp /home/baekh/anaconda3/lib/libcusolver.so.10 /usr/local/cuda-10.0/targets/x86_64-linux/lib/libcusolver.so.10

# physical_devices = tf.config.experimental.list_physical_devices('GPU')

#캡처된 영역 사이즈 설정 (4:3)
face_width = 150
face_height = 200

img_file_dir = "face_recorded_files"
img_file_list = os.listdir(img_file_dir)
x_list = []
y_list = []
class_names = []
for img in img_file_list:
    class_names.append(img.split("_")[0])

# 클래스 이름에 번호를 부여하고 딕셔너리로 저장
class_name_set = set(class_names)
class_name_dic_by_name = {}
class_name_dic_by_no = {}

for i, class_name in enumerate(class_name_set):
    class_name_dic_by_name[class_name] = i
    class_name_dic_by_no[i] = class_name

class_counts = len(class_name_dic_by_no)

with open("pgu_face_class_names.json", "w") as outfile:
    json.dump(class_name_dic_by_no, outfile)

for img in img_file_list:
    this_img = cv2.imread(img_file_dir+"/"+img, cv2.IMREAD_COLOR)
    this_img = this_img/255.0 #normalize
    x_list.append(this_img)
    y_list.append(class_name_dic_by_name[img.split("_")[0]])

x_list, y_list = np.array(x_list), np.array(y_list)

# print(x_list, y_list)
# print(x_list.shape, y_list.shape)

x_train, x_test, y_train, y_test = train_test_split(x_list, y_list, test_size=0.2, shuffle=True, stratify=y_list, random_state=34)

# print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) #(82, 200, 150, 3) (21, 200, 150, 3) (82,) (21,)

# 데이터와 라벨이 제대로 붙었는지 확인
# for i in range(25):
#     plt.subplot(5,5, i+1) #5x5 subplot에서 i+1번째 subplot을 의미함
#     plt.xticks([]) #x축에 눈금을 없애는 역할
#     plt.yticks([]) #y축에 눈금을 없애는 역할
#     plt.grid(False)
#     plt.imshow(x_train[i], cmap = plt.cm.binary) #cmap = plt.cm.binary 는 흑백으로 보이게
#     plt.xlabel(class_name_dic_by_no[y_train[i]])
# plt.show()

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():

    # 네트워크 구조 정의. VGG를 차용하여 조금 수정함
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(face_height, face_width, 3), kernel_size=(3,3), filters=32, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=124, padding="same", activation="relu"),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=256, padding="same", activation="relu"),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Conv2D(kernel_size=(3,3), filters=512, padding="valid", activation="relu"),
        tf.keras.layers.MaxPool2D(strides=(2,2)),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=512, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(units=256, activation="relu"),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(units=10, activation="relu"),
        tf.keras.layers.Dense(units=class_counts, activation="softmax")
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), 
                loss="sparse_categorical_crossentropy", 
                metrics=["accuracy"])
              
#계층, 차원, 파라미터 수 요약 확인
model.summary()

print("클래스 갯수 : " + str(class_counts))

#과적합 방지를 위한 조기종료 콜백 함수 등록 https://wikidocs.net/28147
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10, verbose=0, mode="auto")
# history = model.fit(x_train, y_train, epochs=1000, validation_split=0.25, callbacks=[early_stopping])
history = model.fit(
    x_train,
    y_train,
    epochs=500
    # validation_split=0.25
)

#train loss, validation loss 그래프 출력
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["loss"], "b-", label="loss")
plt.plot(history.history["val_loss"], "r--", label="val_loss")
plt.xlabel("Epoch")
plt.legend()

#train accutacy, validation accuracy 그래프 출력
plt.subplot(1, 2, 2)
plt.plot(history.history["accuracy"], "g-", label="accuracy")
plt.plot(history.history["val_accuracy"], "k--", label="val_accuracy")
plt.xlabel("Epoch")
plt.ylim(0.7, 1)
plt.legend()
plt.show()

print("최종 결과 : ", model.evaluate(x_test, y_test))
print("클래스 갯수 : ", str(class_counts))

#모델 저장
model.save("pgu_face_model.h5")
json_string = model.to_json()
open("pgu_face_model_json.json", "w").write(json_string)
model.save_weights("pgu_face_model_weights.h5", overwrite=True)