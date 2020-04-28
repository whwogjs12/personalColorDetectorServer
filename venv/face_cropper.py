import face_recognition
import cv2


class FaceCropper(object):

    # def __init__(self):

    def generate(self, image_path):
        # 이미지 가져오기
        img = image_path

        # 불러오지 못한경우
        if img is None:
            print("Can't open image file")
            return 0

        # 얼굴 검출
        face_locations = face_recognition.face_locations(img)

        # 얼굴 검출을 실패한 경우
        if face_locations is None:
            print('Failed to detect face')
            return -1

        facecnt = len(face_locations)
        print("Detected faces: %d" % facecnt)

        # 검출된 얼굴이 0개일 경우
        if facecnt == 0:
            print("no face")
            return -1

        # 사진에서 얼굴 자르기
        for (top, right, bottom, left) in face_locations:
            face_image = img[top:bottom, left:right]

            lastimg = cv2.resize(face_image, (56, 56))

            # 얼굴이 1개일 경우
            if facecnt == 1:
                return lastimg
            # 얼굴아 2개 이상일 경우
            else:
                print("Two or many faces")
                return -2