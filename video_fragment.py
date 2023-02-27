from moviepy.editor import *
import cv2


# TODO 长视频变短视频
def seconds():
    source = r"D:\Study\experiment\Relaxing_highway_traffic.mp4"
    video = CompositeVideoClip([VideoFileClip(source).subclip(10, 130)])
    des = "D:/python/DATA/10_130.mp4"
    video.write_videofile(des)


def frames(videos_path, video_dir, time_interval, fps):
    '''
    :param videos_path: 视频的存放路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    counter = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    dir = video_dir + '%d' % counter + '.mp4'
    videoWriter = cv2.VideoWriter(dir, fourcc, fps, (1280, 720))
    while success:
        success, image = vidcap.read()
        count += 1
        videoWriter.write(image)
        if count % time_interval == 0:
            videoWriter.release()
            counter += 1
            if counter < 20:
                dir = video_dir + '%d' % counter + '.mp4'
                videoWriter = cv2.VideoWriter(dir, fourcc, fps, (1280, 720))
    print(counter)


if __name__ == '__main__':
    videos_path = r"D:\Python\DATA\0_10.mp4"
    frames_save_path = './video/'
    time_interval = 15  # 隔一帧保存一次
    fps = 30
    frames(videos_path, frames_save_path, time_interval, fps)
