import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm
import Training
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

import torchvision.datasets
from torchvision import transforms


# 학습한 후 불러오기
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Training.CNN()

trans = transforms.Compose(
        [transforms.Resize([56, 56]),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


train_data = torchvision.datasets.ImageFolder(root='D:\\personal\\train\\', transform=trans)
test_data = torchvision.datasets.ImageFolder(root='D:\\personal\\test\\', transform=trans)

# 배치 크기가 50인 DataLoader 를 각각 작성
train_loader = DataLoader(dataset=train_data, batch_size=50, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=len(test_data))

# 신경망의 모든 파라미터를 GPU로 전송
model.to(device)

# 훈련 실행
model.train_net(model, train_loader, test_loader, n_iter=40, device=device)


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
    #  connectionSocket.close()

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

        predict = torch.from_numpy(predict)
        predict = np.transpose(predict, (2, 0, 1))
        predict = predict.unsqueeze(0)
        predict = predict.to(device, dtype=torch.float32)
        print(model(predict))
        _, predic_result = model(predict).max(1)
        print(_)
        print(predic_result)
        predic_result = predic_result.to('cpu')
        predic_result = predic_result.numpy()
        predictedResult = predic_result[0]
        print("Finish ")



    # 데이터 넘기는 코드
    if data:
        result = str(predictedResult)
        print(result.encode())
        connectionSocket.sendall(result.encode())

    connectionSocket.close()