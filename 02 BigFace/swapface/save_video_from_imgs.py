import cv2 as cv
import os
import pprint

# 设置帧率
fps = 25
# 获取窗口大小
size = (880, 390)
# 调用VideoWrite（）函数
# videoWrite = cv.VideoWriter('MySaveVideo.avi', cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
# videoWrite = cv.VideoWriter('MySaveVideo.flv', cv.VideoWriter_fourcc(*'FLV1'), fps, size)
videoWrite = cv.VideoWriter('MySaveVideo.mp4', cv.VideoWriter_fourcc(*'mp4v'), fps, size)
# 先获取一帧，用来判断是否成功调用摄像头
p = 'video/pinjie'
tmp_paths = [os.path.join(p, i) for i in os.listdir(p)]
tmp_paths = sorted(tmp_paths)
pprint.pprint(tmp_paths)

# 通过循环保存帧
for path in tmp_paths:
    frame = cv.imread(path)
    videoWrite.write(frame)

videoWrite.release()

# # 调用摄像头
# videoCapture = cv.VideoCapture(0)
# # 设置帧率
# fps = 30
# # 获取窗口大小
# size = (int(videoCapture.get(cv.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv.CAP_PROP_FRAME_HEIGHT)))
#
# # 调用VideoWrite（）函数
# videoWrite = cv.VideoWriter('MySaveVideo.avi', cv.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
#
# # 先获取一帧，用来判断是否成功调用摄像头
# success, frame = videoCapture.read()
#
# # 通过设置帧数来设置时间,减一是因为上面已经获取过一帧了
# numFrameRemainling = fps * 5 - 1
#
# # 通过循环保存帧
# while success and numFrameRemainling > 0:
#     videoWrite.write(frame)
#     success, frame = videoCapture.read()
#     numFrameRemainling -= 1
#
# # 释放摄像头
# videoCapture.release()
# videoWrite.release()
