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

def get_class_info(img_file_dir):
    img_file_list = os.listdir(img_file_dir)

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
    
    return class_name_dic_by_name, class_name_dic_by_no, class_counts
    
def train_test_split_by_class(img_file_dir, test_ratio, batch_size):
    class_name_dic_by_name, _, _ = get_class_info(img_file_dir)
    img_file_list = os.listdir(img_file_dir)

    #클래스별 균등한 비율로 train, test 데이터를 나눠야 함
    for idx, c_name in enumerate(class_name_dic_by_name):
        this_x_list, this_y_list = [], []
        for img in img_file_list:
            if img.split("_")[0] == c_name:
                this_img = cv2.imread(img_file_dir+"/"+img, cv2.IMREAD_COLOR)
                this_img = this_img/255.0 #normalize
                this_x_list.append(this_img)
                this_y_list.append(class_name_dic_by_name[img.split("_")[0]])
        this_x_list, this_y_list = np.array(this_x_list), np.array(this_y_list)
        this_x_train, this_x_test, this_y_train, this_y_test = train_test_split(this_x_list, this_y_list, test_size=test_ratio, shuffle=True, stratify=this_y_list, random_state=34)
        if idx==0:
            x_train, x_test, y_train, y_test = this_x_train, this_x_test, this_y_train, this_y_test
        else:
            x_train = np.concatenate([x_train, this_x_train], axis=0)
            x_test = np.concatenate([x_test, this_x_test], axis=0)
            y_train = np.concatenate([y_train, this_y_train], axis=0)
            y_test = np.concatenate([y_test, this_y_test], axis=0)
    
    if len(y_test)%batch_size > 0:
        print("------------------------------------------------------------------------------------------------------------------------------------------------")
        print("테스트 데이터 수를 배치 사이즈로 divide할 수 없어 오류가 발생할 수 있습니다. test_ratio 변수를 조정한 후 다시 시도해 주세요.")
        print("테스트 데이터 수 : ", len(y_test))
        print("배치 사이즈 수 : ", batch_size)
        print("test_ratio : ", test_ratio)
        print("------------------------------------------------------------------------------------------------------------------------------------------------")
        exit()

    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    #에포크 횟수 설정
    epoch_counts = 1000

    # GPU 메모리 문제 발생 예방
    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    #GPU 병렬사용을 위한 설정
    tf.compat.v1.disable_eager_execution() 
    mirrored_strategy = tf.distribute.MirroredStrategy()

    #캡처된 영역 사이즈 설정 (4:3)
    face_width = 150
    face_height = 200

    # 이미지 데이터가 저장된 경로 지정
    img_file_dir = "face_recorded_files"

    class_name_dic_by_name, class_name_dic_by_no, class_counts = get_class_info(img_file_dir)
    x_train, x_test, y_train, y_test = train_test_split_by_class(img_file_dir, test_ratio =  0.196, batch_size = 32)

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
    print("에포크 횟수 : " + str(epoch_counts))

    #과적합 방지를 위한 조기종료 콜백 함수 등록 https://wikidocs.net/28147
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.05, patience=10, verbose=0, mode="auto")
    # history = model.fit(x_train, y_train, epochs=1000, validation_split=0.25, callbacks=[early_stopping])
    history = model.fit(
        x_train,
        y_train,
        epochs=epoch_counts
        # validation_split=0.25
    )

    print("최종 결과 : ", model.evaluate(x_test, y_test))
    print("클래스 갯수 : ", str(class_counts))

    #모델 저장
    try:
        pgu_face_model_dir = "models/pgu_face_model"
        json_string = model.to_json()
        open(pgu_face_model_dir + "/pgu_face_model_json.json", "w").write(json_string)
        model.save_weights(pgu_face_model_dir + "/pgu_face_model_weights.h5", overwrite=True)
        with open(pgu_face_model_dir + "/pgu_face_class_names.json", "w") as outfile:
            json.dump(class_name_dic_by_no, outfile)
    except:
        print("=================================")
        print("=================================")
        print("모델 저장 중 오류가 발생하였습니다!")
        print("=================================")
        print("=================================")
    finally:
        #train loss, validation loss 그래프 출력
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], "b-", label="loss")
        # plt.plot(history.history["val_loss"], "r--", label="val_loss")
        plt.xlabel("Epoch")
        plt.legend()

        #train accutacy, validation accuracy 그래프 출력
        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], "g-", label="accuracy")
        # plt.plot(history.history["val_accuracy"], "k--", label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylim(0.7, 1)
        plt.legend()
        plt.show()