import heapq
import os
import random

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


def fun(function, line):
    if function == 'reverseSize':
        return 1/(line[3]*line[4])
    if function == 'size':
        return line[3] * line[4]
    if function == 'num':
        return 1


def statistic(path, txt, m, n, function, label, begin, end):
    # TODO: 统计各个框内物体数目/bounding box的大小
    if m > 5 or n > 5:
        print("m or n out of range")
        return
    count = begin
    w = 1/m
    h = 1/n
    counter = [0 for _ in range(m*n)]
    num = [0 for _ in range(m*n)]
    while count <= end:
        position = path + txt + '_%d' % count + '.txt'
        eachtxt = np.loadtxt(position)
        for line in eachtxt:
            if line[0] == label:
                if line[1] < w:
                    if line[2] < h:
                        counter[0] += fun(function, line)
                        num[0] += 1
                    elif line[2] < 2 * h:
                        counter[1] += fun(function, line)
                        num[1] += 1
                    elif n >= 3 and line[2] < 3 * h:
                        counter[2] += fun(function, line)
                        num[2] += 1
                    elif n >= 4 and line[2] < 4 * h:
                        counter[3] += fun(function, line)
                        num[3] += 1
                    elif n >= 5 and line[2] < 5 * h:
                        counter[4] += fun(function, line)
                        num[4] += 1
                elif line[1] < 2*w:
                    if line[2] < h:
                        counter[n] += fun(function, line)
                        num[n] += 1
                    elif line[2] < 2 * h:
                        counter[n+1] += fun(function, line)
                        num[n+1] += 1
                    elif n >= 3 and line[2] < 3 * h:
                        counter[n+2] += fun(function, line)
                        num[n+2] += 1
                    elif n >= 4 and line[2] < 4 * h:
                        counter[n+3] += fun(function, line)
                        num[n+3] += 1
                    elif n >= 5 and line[2] < 5 * h:
                        counter[n+4] += fun(function, line)
                        num[n+4] += 1
                elif m >= 3 and line[1] < 3*w:
                    if line[2] < h:
                        counter[2*n] += fun(function, line)
                        num[2*n] += 1
                    elif line[2] < 2 * h:
                        counter[2*n+1] += fun(function, line)
                        num[2*n+1] += 1
                    elif n >= 3 and line[2] < 3 * h:
                        counter[2*n+2] += fun(function, line)
                        num[2*n+2] += 1
                    elif n >= 4 and line[2] < 4 * h:
                        counter[2*n+3] += fun(function, line)
                        num[2*n+3] += 1
                    elif n >= 5 and line[2] < 5 * h:
                        counter[2*n+4] += fun(function, line)
                        num[2*n+4] += 1
                elif m >= 4 and line[1] < 4*w:
                    if line[2] < h:
                        counter[3*n] += fun(function, line)
                        num[3*n] += 1
                    elif line[2] < 2 * h:
                        counter[3*n+1] += fun(function, line)
                        num[3*n+1] += 1
                    elif n >= 3 and line[2] < 3 * h:
                        counter[3*n+2] += fun(function, line)
                        num[3*n+2] += 1
                    elif n >= 4 and line[2] < 4 * h:
                        counter[3*n+3] += fun(function, line)
                        num[3*n+3] += 1
                    elif n >= 5 and line[2] < 5 * h:
                        counter[3*n+4] += fun(function, line)
                        num[3*n+4] += 1
                elif m >= 5 and line[1] < 5*w:
                    if line[2] < h:
                        counter[4*n] += fun(function, line)
                        num[4*n] += 1
                    elif line[2] < 2 * h:
                        counter[4*n+1] += fun(function, line)
                        num[4*n+1] += 1
                    elif n >= 3 and line[2] < 3 * h:
                        counter[4*n+2] += fun(function, line)
                        num[4*n+2] += 1
                    elif n >= 4 and line[2] < 4 * h:
                        counter[4*n+3] += fun(function, line)
                        num[4*n+3] += 1
                    elif n >= 5 and line[2] < 5 * h:
                        counter[4*n+4] += fun(function, line)
                        num[4*n+4] += 1
        count += 1
    return counter, num


def rank(array):
    result_arg = np.argsort(array)
    result_rank = np.zeros(len(result_arg))
    for i in range(len(result_arg)):
        result_rank[result_arg[i]] = i
    return result_rank


def time_slot():
    path = "D:/python/DATA/labels_14min/labels/"  # 需要最后一个斜杠
    txt = 'Relaxing_highway_traffic'
    m, n = 5, 5
    label = 2
    begin = 1

    dot = 100
    slot = [100*i for i in range(1, dot+1)]
    ans1 = []
    ans2 = []
    for end in slot:
        counter, num = statistic(path, txt, m, n, 'size', label, begin, end)
        res = [0 for i in range(len(counter))]
        for i in range(len(counter)):
            if num[i] != 0:
                res[i] = counter[i] / num[i]
        counter = res
        maximum1 = max(counter)
        maximum2 = max(num)

        for index, item in enumerate(counter):
            counter[index] = item/maximum1
        ans1.append(counter)
        for index, item in enumerate(num):
            num[index] = item/maximum2
        ans2.append(num)

    change1 = [[] for _ in range(m * n)]
    change2 = [[] for _ in range(m * n)]
    for i in range(m*n):
        for j in range(len(ans1)):
            change1[i].append(ans1[j][i])
            change2[i].append(ans2[j][i])

    zeros = [0 for _ in range(dot)]
    for index, item in enumerate(change1):
        if item != zeros:
            if item[-1] > 0.05:
                continue
            X = [100*i for i in range(1, dot+1)]
            plt.plot(X, item, linewidth=1.0, linestyle="-", label=str(index))
    plt.legend(loc="best")
    plt.xlabel('time slot(frame)', fontsize=10)
    plt.ylabel('box size', fontsize=10)
    plt.show()
    for index, item in enumerate(change2):
        if item != zeros:
            X = [100*i for i in range(1, dot+1)]
            plt.plot(X, item, linewidth=1.0, linestyle="-", label=str(index))
    plt.legend(loc="best")
    plt.xlabel('time slot(frame)', fontsize=10)
    plt.ylabel('box num', fontsize=10)
    plt.show()
    print(change1)
    print(change2)


def heatmap():
    path = "D:/Python/yolov5-master/runs/detect/exp/labels/"  # 文件夹目录不需要最后一个斜杠
    txt = 'gt'
    m, n = 5, 5
    label = 2
    begin, end = 1, 300
    counter, num = statistic(path, txt, m, n, 'size', label, begin, end)
    res = [0 for i in range(len(counter))]
    for i in range(len(counter)):
        if num[i] != 0:
            res[i] = counter[i]/num[i]
    counter = num
    maximum = max(counter)
    print(counter)
    ranking = rank(counter)
    print(ranking)

    array = [[] for _ in range(n)]
    for i in range(n):
        for j in range(m):
            array[i].append(counter[j*n+i]/maximum)
            # array[i].append(counter[j * n + i])
    print(array)

    ax = sns.heatmap(array, cmap="YlGnBu", annot=True, linewidths=.5)
    # ax.set_title('sum of 1/(bbox size)')  # 图标题
    # ax.set_xlabel('x label')  # x轴标题
    # ax.set_ylabel('y label')
    plt.show()


def chooseRegion(size: list, num: list) -> list:
    nummax = heapq.nlargest(5, range(len(num)), num.__getitem__)
    # print(nummax)
    for index, val in enumerate(size):
        if val == 0:
            size[index] = 1
    sizemin = heapq.nsmallest(5, range(len(size)), size.__getitem__)
    # print(sizemin)
    ans = [val for val in nummax if val in sizemin]
    return ans


def calcuate_choose():
    path = "D:/python/DATA/labels_14min/labels/"  # 需要最后一个斜杠
    txt = 'Relaxing_highway_traffic'
    m, n = 5, 5
    label = 2
    begin, end = 1, 2000

    counter, num = statistic(path, txt, m, n, 'size', label, begin, end)
    res = [0 for i in range(len(counter))]
    for i in range(len(counter)):
        if num[i] != 0:
            res[i] = counter[i] / num[i]
    counter = res
    maximum1 = max(counter)
    maximum2 = max(num)

    for index, item in enumerate(counter):
        counter[index] = item / maximum1

    for index, item in enumerate(num):
        num[index] = item / maximum2
    print("Box size:")
    print(counter)
    print("Box num:")
    print(num)
    print(chooseRegion(counter, num))


def observe():
    path = "D:/python/DATA/labels_14min/labels/"  # 需要最后一个斜杠
    txt = 'Relaxing_highway_traffic'
    m, n = 5, 5
    label = 2
    begin, end = 1, 2000
    list = [10*i for i in range(1, 1000)]
    pre = []
    for end in list:
        counter, num = statistic(path, txt, m, n, 'size', label, begin, end)
        res = [0 for i in range(len(counter))]
        for i in range(len(counter)):
            if num[i] != 0:
                res[i] = counter[i] / num[i]
        counter = res
        maximum1 = max(counter)
        maximum2 = max(num)

        for index, item in enumerate(counter):
            counter[index] = item / maximum1

        for index, item in enumerate(num):
            num[index] = item / maximum2

        if end == 10:
            pre = chooseRegion(counter, num)
            print("The region we choose:")
            print(pre)
        else:
            cur = chooseRegion(counter, num)
            if cur != pre:
                print('%d frame, the result change' % end)
                print("The region we choose:")
                print(cur)
                print('\n')
                pre = cur


if __name__ == '__main__':
    heatmap()
    # time_slot()
    # calcuate_choose()
    # observe()

