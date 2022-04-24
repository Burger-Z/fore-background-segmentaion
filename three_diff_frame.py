import cv2
import numpy as np

video_path = "./test.mp4"
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 初始化第1.2.3帧
one_frame = np.zeros((height, width), dtype=np.uint8)
two_frame = np.zeros((height, width), dtype=np.uint8)
three_frame = np.zeros((height, width), dtype=np.uint8)

area_threh = 100  # 物体bbox面积阈值

while cap.isOpened():
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not ret:
        break
    one_frame, two_frame, three_frame = two_frame, three_frame, frame_gray

    # 1.2帧做差
    abs1 = cv2.absdiff(one_frame, two_frame)  # 相减
    _, thresh1 = cv2.threshold(abs1, 15, 255, cv2.THRESH_BINARY)  # 二值，大于40的为255，小于0

    # 2.3帧做差
    abs2 = cv2.absdiff(two_frame, three_frame)
    _, thresh2 = cv2.threshold(abs2, 15, 255, cv2.THRESH_BINARY)

    binary = cv2.bitwise_and(thresh1, thresh2)  # 与运算

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    #     erode = cv2.erode(binary,kernel)#腐蚀
    #     dilate =cv2.dilate(binary,kernel)#膨胀
    #     dilate =cv2.dilate(dilate,kernel)#膨胀

    # 轮廓提取
    _, contours, hierarchy = cv2.findContours(binary.copy(), mode=cv2.RETR_EXTERNAL,
                                           method=cv2.CHAIN_APPROX_SIMPLE)  # 寻找轮廓
    for contour in contours:
        if cv2.contourArea(contour) > area_threh:
            x, y, w, h = cv2.boundingRect(contour)  # 找方框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    img_show = np.hstack((frame, cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('three frame diff', img_show)

    if cv2.waitKey(50) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()