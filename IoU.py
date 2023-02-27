# 读取txt-标准txt为基准-分类别求阈值-阈值为0. 0.3 0.5 0.7的统计
import glob
import os
import numpy as np


def compute_IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    # 两矩形无相交区域的情况
    # if left_column_max >= right_column_min or down_row_min <= up_row_max:
    #     return 0
    # # 两矩形有相交区域的情况
    # else:
    S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
    S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
    S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
    return S_cross/(S1+S2-S_cross)


threshold1 = 0.3
threshold2 = 0.5
threshold3 = 0.7

stdtxt = 'D:/python/yolov5-6.0/runs/detect/exp2/labels/'  # 标注txt路径
testtxt = 'D:/python/yolov5-6.0/runs/detect/exp4/labels/'  # 测试txt路径

txtlist = glob.glob(r'%s\*.txt' % stdtxt)  # 获取所有txt文件
count = 1
result = 0

for path in txtlist:  # 对每个txt操作
    print(count)
    counter0 = 0
    counter1 = 0
    counter2 = 0
    counter3 = 0
    txtname = os.path.basename(path)[:-4]  # 获取txt文件名
    label = 2
    eachtxt = np.loadtxt(path)  # 读取文件
    for line in eachtxt:
        if line[0] == label:
            line[1] = line[1] * 1920
            line[2] = line[2] * 1080
            line[3] = line[3] * 1920
            line[4] = line[4] * 1080
            testfile = np.loadtxt(testtxt + "ten_second_100k_%d" % count + '.txt')
            iou_list = []  # 用来存储所有iou的集合
            for tline in testfile:  # 对测试txt的每行进行操作
                if tline[0] == label:
                    tline[1] = tline[1] * 1920
                    tline[2] = tline[2] * 1080
                    tline[3] = tline[3] * 1920
                    tline[4] = tline[4] * 1080
                    iou = compute_IOU(line[1:5],tline[1:5])
                    # print(iou)
                    iou_list.append(iou)  # 添加到集合尾部

            threshold = max(iou_list)  # 阈值取最大的
            # 阈值统计
            if threshold >= threshold3:
                counter3 = counter3 + 1
            elif threshold >= threshold2:
                counter2 = counter2 + 1
            elif threshold >= threshold1:
                counter1 = counter1 + 1
            elif threshold < threshold1:  # 漏检
                counter0 = counter0 + 1

    print(counter3, counter2, counter1, counter0)
    result += counter3/(counter0 + counter1 + counter2 + counter3)
    print("IOU超过百分之70的比例%.3f" % (counter3/(counter0 + counter1 + counter2 + counter3)))
    count += 1

    res = result/(count - 1)
    print(res)