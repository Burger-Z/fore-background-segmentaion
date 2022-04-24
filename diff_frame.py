import cv2
import pandas as pd
import numpy as np

video_path = "./test.mp4"

cam = cv2.VideoCapture(video_path)  # 打开一个视频
input_fps = cam.get(cv2.CAP_PROP_FPS)  # 获取视频帧率
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)) # 获取视频宽度
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 获取视频高度

ret_val, input_image = cam.read()  # 读取视频第一帧
input_image = cv2.resize(input_image, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)  # 窗口大小
gray_lwpCV = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)  # 将第一帧转为灰度
gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)  # 对转换后的灰度图进行高斯模糊
background = gray_lwpCV  # 将高斯模糊后的第一帧作为初始化背景

area_threh = 100  # 物体bbox面积阈值

while (cam.isOpened()) and ret_val == True:
    ret_val, input_image = cam.read()  # 继续读取视频帧
    input_image = cv2.resize(input_image, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)  # 窗口大小
    gray_lwpCV = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)  # 对读取到的视频帧进行灰度处理+高斯模糊
    diff = cv2.absdiff(background, gray_lwpCV)  # 将最新读取的视频帧和背景做差

    # 跟着图像变换背景，如果背景变化区域小于20%或者大于75%，则将当前帧作为新得背景区域
    tem_diff = diff.flatten()

    tem_ds = pd.Series(tem_diff)
    tem_per = 1 - len(tem_ds[tem_ds == 0]) / len(tem_ds)
    if (tem_per < 0.2) | (tem_per > 0.75):
        background = gray_lwpCV
    else:
        ret, diff_binary = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)  # 对差值diff进行二值化
        _, contours, hierarchy = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)  # 对二值化之后得结果进行轮廓提取
        for c in contours:
            if (cv2.contourArea(c) < area_threh):  # 对于矩形区域，只显示大于给定阈值的轮廓（去除微小的变化等噪点）
                continue
            (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
            cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        rst = np.hstack((input_image, cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
        cv2.imshow('frame diff', rst)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('diff.png', diff)
            cv2.imwrite('df_rst.png', rst)

cam.release()
cv2.destroyAllWindows()