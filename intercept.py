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
    ints = int(fps)
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
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width,height)
    return width,height


if __name__ == '__main__':
    url_1 = "D:/python/opencv_study/video/video/ten_second.mp4"  # 源视频
    url_2 = "D:/python/opencv_study/video/video/test.mp4"  # 转换后视频

    cx = 640  # 起点x
    cy = 360  # 起点y
    width, height = get_vedio_height_width(url_1)
    vw = int(width/2)  # 宽
    vh = int(height/2)  # 高

    videoJDK(cy, cx, vw, vh, url_1, url_2)