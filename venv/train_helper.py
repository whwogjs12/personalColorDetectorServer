import torch
from torch import nn, optim
import tqdm
import tqdm


class TrainHelper(object):
    
    def eval_net(self, net, data_loader, device="cpu"):
        # Dropout 및 BatchNorm을 무효화
        net.eval()
        ys = []
        ypreds = []
        for x, y in data_loader:
            # to 메서드로 계산을 실행할 디바이스로 전송
            x = x.to(device)
            y = y.to(device)
            # 확률이 가장 큰 클래스를 예측(리스트 2.1 참조)
            # 여기선 forward（추론） 계산이 전부이므로 자동 미분에
            # 필요한 처리는 off로 설정해서 불필요한 계산을 제한다
            with torch.no_grad():
                _, y_pred = net(x).max(1)
            ys.append(y)
            ypreds.append(y_pred)

        # 미니 배치 단위의 예측 결과 등을 하나로 묶는다
        ys = torch.cat(ys)
        ypreds = torch.cat(ypreds)
        # 예측 정확도 계산
        acc = (ys == ypreds).float().sum() / len(ys)
        return acc.item()



    # 훈련용 헬퍼 함수
    def train_net(self, net, train_loader, test_loader, optimizer_cls=optim.Adam,
                  loss_fn=nn.CrossEntropyLoss(),
                  n_iter=10, device="cpu"):
        train_losses = []
        train_acc = []
        val_acc = []
        optimizer = optimizer_cls(net.parameters())
        for epoch in range(n_iter):
            running_loss = 0.0
            # 신경망을 훈련 모드로 설정
            net.train()
            n = 0
            n_acc = 0
            # 시간이 많이 걸리므로 tqdm을 사용해서 진행바를 표시
            for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
                xx = xx.to(device)
                yy = yy.to(device)
                h = net(xx)
                loss = loss_fn(h, yy)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                n += len(xx)
                _, y_pred = h.max(1)
                n_acc += (yy == y_pred).float().sum().item()
            train_losses.append(running_loss / i)
            # 훈련 데이터의 예측 정확도
            train_acc.append(n_acc / n)

            # 검증 데이터의 예측 정확도
            val_acc.append(self.eval_net(net, test_loader, device))
            # epoch의 결과 표시
            print(epoch, train_losses[-1], train_acc[-1],
                  val_acc[-1], flush=True)
