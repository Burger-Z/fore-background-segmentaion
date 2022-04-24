import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

video_path = "./test.mp4"

cap = cv2.VideoCapture(video_path)  # 打开一个视频

input_fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频帧率

# 获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret_val, input_image = cap.read()  # 读取视频第一帧
src = cv2.resize(input_image, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)  # 窗口大小
background = src  # 将高斯模糊后的第一帧作为初始化背景

#history =   # 建模帧数
area_threh = 100  # 物体bbox面积阈值
bs = cv2.createBackgroundSubtractorMOG2()   # 建立背景模型

while (cap.isOpened()) and ret_val == True:

    ret_val, input_image = cap.read()  # 继续读取视频帧
    src = cv2.resize(input_image, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)  # 窗口大小
    fg_mask = bs.apply(src)     # 获取前景掩码

   # cv2.imshow('foreground_mask', fg_mask)
    _, contours, hierarchy = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)  # 对二值化之后得结果进行轮廓提取
    for c in contours:
        if (cv2.contourArea(c) < area_threh):  # 对于矩形区域，只显示大于给定阈值的轮廓（去除微小的变化等噪点）
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    rst = np.hstack((src, cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)))
    cv2.imshow('result',rst)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('fg_mask.png',fg_mask)
        cv2.imwrite('bs_rst.png',rst)
cap.release()
cv2.destroyAllWindows()