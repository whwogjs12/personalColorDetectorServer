from face_cropper import FaceCropper
from PIL import Image
import numpy as np
import cv2
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
from torchvision import transforms
import torchvision.datasets
import torch


class DataProc:


	def __init__(self):
		self.trans = transforms.Compose(
			[transforms.Resize([56, 56]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
																					std=[0.229, 0.224, 0.225])])

		self.train_loader = None
		self.test_loader = None


	def image_process(self, image):
		# image의 타입은 ndarray
		fc = FaceCropper()  # 얼굴 자르기
		image = fc.generate(image)
		image = Image.fromarray(image)  # ndarray를 PIL로 변환
		image = self.trans(image).float()  # 이미지룰 신경망에 맞게 가공
		image = torch.unsqueeze(image, 0)  # 3차원을 4차원으로 변경
		return image.cuda()


	def dataset_loader(self, train_path, test_path, batch_size):
		# 데이터셋 불러오기
		train_data = torchvision.datasets.ImageFolder(root=train_path, transform=self.trans)
		test_data = torchvision.datasets.ImageFolder(root=test_path, transform=self.trans)

		# 배치 크기가 50인 DataLoader를 각각 작성
		self.train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle = True)
		self.test_loader = DataLoader(dataset=test_data, batch_size=len(test_data))