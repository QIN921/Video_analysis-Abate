from moviepy.editor import VideoFileClip

a = VideoFileClip('/experiment/video/ten_second.mp4')
a.write_videofile('D:/python/video_process/experiment/video/ten_second_50k.mp4', bitrate='50k')
