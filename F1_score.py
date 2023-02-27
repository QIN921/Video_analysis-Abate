import os
import numpy as np


def compute_IOU(rec1, rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    ans1 = [(rec1[0] - rec1[2]/2), (rec1[1] - rec1[3]/2), (rec1[0] + rec1[2]/2), (rec1[1] + rec1[3]/2)]
    ans2 = [(rec2[0] - rec2[2]/2), (rec2[1] - rec2[3]/2), (rec2[0] + rec2[2]/2), (rec2[1] + rec2[3]/2)]

    left_column_max  = max(ans1[0], ans2[0])
    right_column_min = min(ans1[2], ans2[2])
    up_row_max       = max(ans1[1], ans2[1])
    down_row_min     = min(ans1[3], ans2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (ans1[2]-ans1[0])*(ans1[3]-ans1[1])
        S2 = (ans2[2]-ans2[0])*(ans2[3]-ans2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
    return S_cross/(S1+S2-S_cross)


def high2low(stdtxt, testtxt, stdpath, testpath, label, threshold):
    # 计算TP与FN
    count = 1  # 用于输出运算到哪一帧
    TP, FN = 0, 0
    # TP true positive最高分辨率检测出来，并在低分辨率也检测出来
    # FN false negative最高分辨率检测出来，但低分辨率没有检测出来
    while count <= 300:  # 对每个txt操作
        # print(count)
        path1 = stdtxt + "stdpath_%d" % (count) + '.txt'  # 需要修改
        path1 = path1.replace('stdpath', stdpath)
        # print(path1)
        if not os.path.exists(path1):  # 高质量视频此帧不存在物体
            count += 1
            continue
        eachtxt = np.loadtxt(path1)
        if eachtxt.ndim == 1:
            eachtxt = eachtxt.reshape((1, 6))

        for line in eachtxt:
            if line[0] == label:
                line[1] = line[1] * 1920
                line[2] = line[2] * 1080
                line[3] = line[3] * 1920
                line[4] = line[4] * 1080
                path2 = testtxt + "testpath_%d" % count + '.txt'
                path2 = path2.replace('testpath', testpath)
                if not os.path.exists(path2):  # 低质量视频此帧不存在物体，则全为FN
                    FN += 1
                    continue

                testfile = np.loadtxt(path2)  # 此处需要修改
                if testfile.ndim == 1:
                    testfile = testfile.reshape((1, 6))
                iou_list = []  # 用来存储所有iou的集合
                for tline in testfile:  # 对测试txt的每行进行操作
                    if tline[0] == label:
                        tline[1] = tline[1] * 1920
                        tline[2] = tline[2] * 1080
                        tline[3] = tline[3] * 1920
                        tline[4] = tline[4] * 1080
                        iou = compute_IOU(line[1:5], tline[1:5])
                        iou_list.append(iou)  # 添加到集合尾部

                if not iou_list:
                    result = 0
                else:
                    result = max(iou_list)  # 阈值取最大的
                # 阈值统计
                if result >= threshold:
                    TP += 1
                else:
                    FN += 1
        count += 1
    return TP, FN


def low2high(stdtxt, testtxt, stdpath, testpath, label, threshold):
    # 计算FP
    count = 1  # 用于输出运算到哪一帧
    FP = 0
    # FP false positive最高分辨率检测没有，但在低分辨率被认为是
    while count <= 300:  # 对每个txt操作
        # print(count)
        path1 = testtxt + "testpath_%d" % count + '.txt'
        path1 = path1.replace('testpath', testpath)
        if not os.path.exists(path1):  # 低质量视频这帧不存在物体
            count += 1
            continue
        eachtxt = np.loadtxt(path1)  # 读取文件
        if eachtxt.ndim == 1:
            eachtxt = eachtxt.reshape((1, 6))

        for line in eachtxt:
            if line[0] == label:
                line[1] = line[1] * 1920
                line[2] = line[2] * 1080
                line[3] = line[3] * 1920
                line[4] = line[4] * 1080

                path2 = stdtxt + "stdpath_%d" % (count) + '.txt'
                path2 = path2.replace('stdpath', stdpath)

                if not os.path.exists(path2):  # 高质量视频此帧不存在物体，则所有均为FP
                    FP += 1
                    continue

                testfile = np.loadtxt(path2)  # 此处需要修改
                if testfile.ndim == 1:
                    testfile = testfile.reshape((1, 6))
                iou_list = []  # 用来存储所有iou的集合
                for tline in testfile:  # 对测试txt的每行进行操作
                    if tline[0] == label:
                        tline[1] = tline[1] * 1920
                        tline[2] = tline[2] * 1080
                        tline[3] = tline[3] * 1920
                        tline[4] = tline[4] * 1080
                        iou = compute_IOU(line[1:5], tline[1:5])
                        # print(iou)
                        iou_list.append(iou)  # 添加到集合尾部
                if not iou_list:
                    result = 0
                else:
                    result = max(iou_list)  # 阈值取最大的
                # 阈值统计
                if result < threshold:
                    FP += 1
        count += 1
    return FP


def F1_score(stdtxt, testtxt, stdpath, testpath, label, threshold):
    TP, FN, FP = 0, 0, 0
    TP, FN = high2low(stdtxt, testtxt, stdpath, testpath, label, threshold)
    FP = low2high(stdtxt, testtxt, stdpath, testpath, label, threshold)

    # print("TP为", TP)
    # # TP true positive最高分辨率检测出来，并在低分辨率也检测出来
    # print("FN为", FN)
    # # FN false negetive最高分辨率检测出来，但低分辨率没有检测出来
    # print("FP为", FP)
    # # FP false positive最高分辨率检测没有，但在低分辨率被认为是

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    # print("precision为", precision)
    # print("recall为", recall)
    # print("F1 score为", F1)
    return F1


if __name__ == '__main__':
    threshold = 0.5

    stdtxt = 'D:/Python/yolov5-master/runs/detect/move/labels/'  # 标注txt路径,高质量,需要最后一个斜杠
    testtxt = 'D:/Python/yolov5-master/runs/detect/move_result/labels/'  # 测试txt路径,低质量,需要最后一个斜杠
    stdpath = 'move'
    testpath = 'output'

    label = 2
    F1 = F1_score(stdtxt, testtxt, stdpath, testpath, label, threshold)
    print(F1)



