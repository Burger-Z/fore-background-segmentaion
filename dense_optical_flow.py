import cv2
import numpy as np
from matplotlib import pyplot as plt
cap = cv2.VideoCapture("test.mp4")
# 获取视频宽度
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# 获取视频高度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
ret, input_image = cap.read()
src = cv2.resize(input_image, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)  # 窗口大小
prvs = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(src)
hsv[...,1] = 255
area_threh = 100
while(1):
    ret, input_image = cap.read()
    src = cv2.resize(input_image, (frame_width // 2, frame_height // 2), interpolation=cv2.INTER_AREA)  # 窗口大小
    next = cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # 前一帧图像 当前帧图像 金字塔尺度关系 图像金字塔层数 均值窗口大小 迭代次数
    # prevImg： 前一帧8-bit单通道图像
    #
    # nextImg： 当前帧图像，与前一帧保持同样的格式、尺寸
    #
    # pyr_scale： 金字塔上下两层之间的尺度关系，该参数一般设置为pyrScale=0.5，表示图像金字塔上一层是下一层的2倍降采样
    #
    # levels：图像金字塔的层数
    #
    # winsize：均值窗口大小，winsize越大，算法对图像噪声越鲁棒，并且能提升对快速运动目标的检测效果，但也会引起运动区域模糊。
    #
    # iterations：算法在图像金字塔每层的迭代次数
    #
    # poly_n：用于在每个像素点处计算多项式展开的相邻像素点的个数。poly_n越大，图像的近似逼近越光滑，算法鲁棒性更好，也会带来更多的运动区域模糊。通常，poly_n=5 or 7
    #
    # poly_sigma：标准差，poly_n=5时，poly_sigma = 1.1；poly_n=7时，poly_sigma = 1.5
    # flags：Operation flags that can be a combination of the following:
    # OPTFLOW_USE_INITIAL_FLOW：Use the input flow as an initial flow approximation.
    # OPTFLOW_FARNEBACK_GAUSSIAN：Use the Gaussian
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    # 色调H：用角度度量，取值范围为0°～360°，从红色开始按逆时针方向计算，红色为0°，绿色为120°,蓝色为240°
    # 饱和度S：取值范围为0.0～1.0
    # 亮度V：取值范围为0.0(黑色)～1.0(白色)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    input = cv2.cvtColor(rgb,cv2.COLOR_BGR2GRAY)

    #cv2.imshow('rgb',rgb)
    #ret, diff_binary = cv2.adaptiveThreshold(input, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)  # 对差值diff进行二值化
    ret, diff_binary = cv2.threshold(input, 10, 255, cv2.THRESH_BINARY)  # 对差值diff进行二值化
    _, contours, hierarchy = cv2.findContours(diff_binary, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)  # 对二值化之后得结果进行轮廓提取
    for c in contours:
        if (cv2.contourArea(c) < area_threh):  # 对于矩形区域，只显示大于给定阈值的轮廓（去除微小的变化等噪点）
            continue
        (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
        cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('optical_flow_rst', np.hstack((src, cv2.cvtColor(input, cv2.COLOR_GRAY2BGR))))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',src)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

cap.release()
cv2.destroyAllWindows()