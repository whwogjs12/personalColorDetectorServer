import cv2
import sys



class FaceCropper(object):

    def __init__(self):
        #얼굴인식을 위한 학습된 xml 불러오기
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.src = "D:\\personal\\predict_data\\"


    def generate(self, image):
        # 얼굴 검출
        faces = self.face_cascade.detectMultiScale(image, 1.1, 3, minSize=(56, 56))

        # 얼굴 검출이 되지 않았을 경우
        if (faces is None):
            print('Failed to detect face')
            return -1

        # 얼굴 개수
        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)

        # 사진에서 얼굴 자르기
        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)    

            faceimg = image[ny:ny+nr, nx:nx+nr]

            lastimg = cv2.resize(faceimg, (56, 56))

            # 얼굴이 1개일 경우
            if facecnt == 1:
                return lastimg

            # 얼굴이 1개보다 많은 경우
            elif facecnt > 1:
                return -2

            # 얼굴아 없는 경우
            else:
                print("zero faces")
                return -1

