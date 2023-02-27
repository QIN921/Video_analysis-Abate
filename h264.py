import os
from F1_score import F1_score


def groundTruth():
    count = 0
    while count < 20:
        command = 'python D:/Python/yolov5-master/detect.py --source ./video/%d.mp4 --name gt_%d' % (count, count)
        os.system(command)
        count += 1


def h264():
    count = 0
    sizes = []
    scores = []
    while count < 20:
        command = 'ffmpeg -i ./video/%d.mp4 -b:v 600k ./output/output_%d.mp4 -y' % (count, count)
        os.system(command)
        video_size = os.path.getsize('D:/Python/Video_analysis/output/output_%d.mp4' % count)
        sizes.append(video_size/1024)
        command_1 = 'python D:/Python/yolov5-master/detect.py --source ./output/output_%d.mp4 --name output_%d' % (count, count)
        os.system(command_1)
        stdtxt = 'D:/Python/yolov5-master/runs/detect/gt_%d/labels/' % count  # 标注txt路径,高质量,需要最后一个斜杠
        testtxt = 'D:/Python/yolov5-master/runs/detect/output_%d/labels/' % count # 测试txt路径,低质量,需要最后一个斜杠
        stdpath = '%d' % count
        testpath = 'output_%d' % count
        label = 2
        threshold = 0.5
        score = F1_score(stdtxt, testtxt, stdpath, testpath, label, threshold)
        scores.append(score)
        print("F1 score is ", score)
        count += 1
    print(sizes)
    print(scores)


if __name__ == '__main__':
    # groundTruth()
    h264()