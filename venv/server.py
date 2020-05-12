import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm
import os.path
from socket import *
import sys
import numpy as np
import cv2
from torch.autograd import Variable
import face_cropper
import time
from PIL import Image
import matplotlib.pyplot as plt
from train_helper import TrainHelper
from model import PersonalModel
from data_process import DataProc

import torchvision.datasets
from torchvision import transforms


# 학습한 후 불러오기
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 트레인 데이터셋 경로
train_path = 'D:\\personal\\train\\'
# 테스트 데이터셋 경로
test_path = 'D:\\personal\\test\\'
# 배치 사이즈
batch_size = 50

# 트레인, 테스트 데이터셋 불러오기
data_proc = DataProc()
data_proc.dataset_loader(train_path, test_path, batch_size)

# 모델 생성
net = PersonalModel()

if os.path.exists("model.prm"):
    # 모델 불러오기
    net.load_state_dict(torch.load("model.prm"))

    net.to(device)

    # 검증모드
    net.eval()
else:
    net.to(device)
    # 훈련 실행
    helper = TrainHelper()
    helper.train_net(net, data_proc.train_loader, data_proc.test_loader, n_iter=50, device=device)
    # 모델 이름 지정
    savePath = "model.prm"
    # CPU 로 변경
    net.cpu()
    # 모델 저장
    torch.save(net.state_dict(), savePath)


net.to(device)

# 서버 연동
port = 4799

server_Socket = socket()  # 기본이 AF_INET, SOCK_STREAM
server_Socket.bind(('', port))

#  동시에 수신받을 수 있는 개수
server_Socket.listen(5)

#  이미지를 받기 위한 코드
while True:
    connectionSocket, addr = server_Socket.accept()
    print(addr)
    data = None
    fileSize = None

    fileSize = connectionSocket.recv(1024).decode()
    data = connectionSocket.recv(int(fileSize))

    while sys.getsizeof(data)<int(fileSize):
        data += connectionSocket.recv(1024)

    #  이미지로 디코딩하는 과정
    numpy_array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
    image = image[:, :, ::-1].copy()


    # 넘어온 이미지에서 얼굴 검출
    fc = face_cropper.FaceCropper()
    predict = fc.generate(image)


    predictedResult = -3

    # 변수가 int인 경우
    if str(type(predict)) == "<class 'int'>":
        predictedResult = predict

    # 얼굴이 검출된 이미지를 학습된 모델에 전달
    elif str(type(predict)) != "<class 'NoneType'>":

        data = Image.fromarray(predict, 'RGB')
        data.save('test.png')
        print(str(type(predict)))
        predict = data_proc.image_process(predict)
        print(net(predict))
        _, predic_result = net(predict).max(1)
        print(_)
        predic_result = predic_result.to('cpu')
        predic_result = predic_result.numpy()
        predictedResult = predic_result[0]
        print("Finish ")



    # 데이터 넘기는 코드
    result = str(predictedResult)
    print(result.encode())
    connectionSocket.sendall(result.encode())

    connectionSocket.close()