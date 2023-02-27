import os

import cv2

# TODO 截取视频的部分区域


def videoJDK(cy, cx, vw, vh, url_1, url_2):
    videoCapture = cv2.VideoCapture(url_1)  # 从文件读取视频
    # 判断视频是否打开
    if videoCapture.isOpened():
        print('Open')
    else:
        print('Fail to open!')

    fps = videoCapture.get(cv2.CAP_PROP_FPS)  # 获取原视频的帧率
    size = (int(vw), int(vh))  # 自定义需要截取的画面的大小
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videoWriter = cv2.VideoWriter(url_2, fourcc, fps, size)
    success, frame = videoCapture.read()  # 读取第一帧
    count = 1
    # ints = int(fps)
    while success:
        print(count)
        frame = frame[cy:cy + vh, cx:cx + vw]  # 截取画面
        # cv2.imshow("Oto Video", frame) #显示
        # cv2.waitKey(int(1000/ints)) #延迟
        videoWriter.write(frame)  # 写视频帧
        success, frame = videoCapture.read()  # 获取下一帧
        count += 1
    videoCapture.release()


# 使用cv2获得视频的分辨率width和height。
def get_vedio_height_width(filename):
    cap = cv2.VideoCapture(filename)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width,height)
    return width,height


def divide_video(m, n, src, des):
    width, height = get_vedio_height_width(src)
    w = int(width/m)
    h = int(height/n)
    count = 1
    # for i in range(m):
    #     for j in range(n):
    #         cy = j*h
    #         cx = i*w
    #         url = des + '/' + "%s" % count + ".mp4"
    #         count += 1
    #         videoJDK(cy, cx, w, h, src, url)
    cy = 2*h
    cx = 0
    url = des + '/' + "%s" % count + ".mp4"
    count += 1
    videoJDK(cy, cx, w, h, src, url)


if __name__ == '__main__':

    url_1 = r"D:\Study\experiment\Relaxing_highway_traffic.mp4"  # 源视频
    url_2 = "D:/python/DATA"  # 转换后视频
    if not os.path.exists(url_1):
        message = "Cannot find " + url_1
        print(message)
        exit()

    if not os.path.exists(url_2):
        print("%s don't exist" % url_2)
        exit()
    divide_video(5, 5, url_1, url_2)
