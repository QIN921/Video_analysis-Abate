from moviepy.video.VideoClip import ColorClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.io.VideoFileClip import VideoFileClip
import cv2

# TODO 视频添加黑色mask

def hide(src, dst, cy, cx, vw, vh):
    video = VideoFileClip(src)
    mask = (ColorClip((vw, vh), (0, 0, 0))
            .set_position((cx, cy))
            .set_duration(video.duration)
            )
    CompositeVideoClip([video, mask]).write_videofile(dst)


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
    width, height = get_vedio_height_width(url_1)
    cx = 0  # 起点x
    cy = 0  # 起点y
    vw = int(width/2)  # 宽
    vh = int(height/2)  # 高
    hide(url_1,url_2, cy, cx, vw, vh)