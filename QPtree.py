import os
import time
import cv2
import numpy as np
from F1_score import F1_score


def fun(function, line, m, n):
    if function == 'reverseSize':
        return 1/(line[3]*line[4])
    if function == 'size':
        return line[3] * line[4]
    if function == 'relativesize':
        return line[3] * line[4] * m * n
    if function == 'num':
        return 1


def normalization(m: int, n: int, pos: str, line: list[float]):
    temp = int(ord(pos[-1])-ord('a'))
    w = (temp // n) / m
    h = (temp % n) / n
    line[1] = (line[1] - w) * m
    line[2] = (line[2] - h) * n
    line[3] = line[3] * m
    line[4] = line[4] * n
    return line


def statistic(path, txt, m, n, function, label):
    # 只能读取一个txt
    # TODO: 统计各个框内物体数目/bounding box的大小
    if m > 5 or n > 5:
        print("m or n out of range")
        return
    w = 1/m
    h = 1/n
    counter = [0 for _ in range(m*n)]
    num = [0 for _ in range(m*n)]
    lists = [[] for _ in range(m*n)]
    position = path + txt
    eachtxt = np.loadtxt(position)
    for line in eachtxt:
        if line[0] == label:
            if line[1] < w:
                if line[2] < h:
                    counter[0] += fun(function, line, m, n)
                    num[0] += 1
                    line = normalization(m, n, 'a', line)
                    lists[0].append(line)
                elif line[2] < 2 * h:
                    counter[1] += fun(function, line, m, n)
                    num[1] += 1
                    line = normalization(m, n, 'b', line)
                    lists[1].append(line)
                elif n >= 3 and line[2] < 3 * h:
                    counter[2] += fun(function, line, m, n)
                    num[2] += 1
                    line = normalization(m, n, 'c', line)
                    lists[2].append(line)
                elif n >= 4 and line[2] < 4 * h:
                    counter[3] += fun(function, line, m, n)
                    num[3] += 1
                    line = normalization(m, n, 'd', line)
                    lists[3].append(line)
                elif n >= 5 and line[2] < 5 * h:
                    counter[4] += fun(function, line, m, n)
                    num[4] += 1
                    line = normalization(m, n, 'e', line)
                    lists[4].append(line)
            elif line[1] < 2*w:
                if line[2] < h:
                    counter[n] += fun(function, line, m, n)
                    num[n] += 1
                    line = normalization(m, n, chr(n + 97), line)
                    lists[n].append(line)
                elif line[2] < 2 * h:
                    counter[n+1] += fun(function, line, m, n)
                    num[n+1] += 1
                    line = normalization(m, n, chr(n+1 + 97), line)
                    lists[n+1].append(line)
                elif n >= 3 and line[2] < 3 * h:
                    counter[n+2] += fun(function, line, m, n)
                    num[n+2] += 1
                    line = normalization(m, n, chr(n+2 + 97), line)
                    lists[n+2].append(line)
                elif n >= 4 and line[2] < 4 * h:
                    counter[n+3] += fun(function, line, m, n)
                    num[n+3] += 1
                    line = normalization(m, n, chr(n+3 + 97), line)
                    lists[n+3].append(line)
                elif n >= 5 and line[2] < 5 * h:
                    counter[n+4] += fun(function, line, m, n)
                    num[n+4] += 1
                    line = normalization(m, n, chr(n+4 + 97), line)
                    lists[n+4].append(line)
            elif m >= 3 and line[1] < 3*w:
                if line[2] < h:
                    counter[2*n] += fun(function, line, m, n)
                    num[2*n] += 1
                    line = normalization(m, n, chr(2*n + 97), line)
                    lists[2*n].append(line)
                elif line[2] < 2 * h:
                    counter[2*n+1] += fun(function, line, m, n)
                    num[2*n+1] += 1
                    line = normalization(m, n, chr(2 * n+1 + 97), line)
                    lists[2*n+1].append(line)
                elif n >= 3 and line[2] < 3 * h:
                    counter[2*n+2] += fun(function, line, m, n)
                    num[2*n+2] += 1
                    line = normalization(m, n, chr(2 * n+2 + 97), line)
                    lists[2*n+2].append(line)
                elif n >= 4 and line[2] < 4 * h:
                    counter[2*n+3] += fun(function, line, m, n)
                    num[2*n+3] += 1
                    line = normalization(m, n, chr(2 * n+3 + 97), line)
                    lists[2*n + 3].append(line)
                elif n >= 5 and line[2] < 5 * h:
                    counter[2*n+4] += fun(function, line, m, n)
                    num[2*n+4] += 1
                    line = normalization(m, n, chr(2 * n+4 + 97), line)
                    lists[2*n + 4].append(line)
            elif m >= 4 and line[1] < 4*w:
                if line[2] < h:
                    counter[3*n] += fun(function, line, m, n)
                    num[3*n] += 1
                    line = normalization(m, n, chr(3 * n + 97), line)
                    lists[3*n].append(line)
                elif line[2] < 2 * h:
                    counter[3*n+1] += fun(function, line, m, n)
                    num[3*n+1] += 1
                    line = normalization(m, n, chr(3 * n+1 + 97), line)
                    lists[3*n + 1].append(line)
                elif n >= 3 and line[2] < 3 * h:
                    counter[3*n+2] += fun(function, line, m, n)
                    num[3*n+2] += 1
                    line = normalization(m, n, chr(3 * n+2 + 97), line)
                    lists[3*n + 2].append(line)
                elif n >= 4 and line[2] < 4 * h:
                    counter[3*n+3] += fun(function, line, m, n)
                    num[3*n+3] += 1
                    line = normalization(m, n, chr(3 * n+3 + 97), line)
                    lists[3*n + 3].append(line)
                elif n >= 5 and line[2] < 5 * h:
                    counter[3*n+4] += fun(function, line, m, n)
                    num[3*n+4] += 1
                    line = normalization(m, n, chr(3 * n+4 + 97), line)
                    lists[3*n + 4].append(line)
            elif m >= 5 and line[1] < 5*w:
                if line[2] < h:
                    counter[4*n] += fun(function, line, m, n)
                    num[4*n] += 1
                    line = normalization(m, n, chr(4 * n + 97), line)
                    lists[4*n].append(line)
                elif line[2] < 2 * h:
                    counter[4*n+1] += fun(function, line, m, n)
                    num[4*n+1] += 1
                    line = normalization(m, n, chr(4 * n+1 + 97), line)
                    lists[4*n + 1].append(line)
                elif n >= 3 and line[2] < 3 * h:
                    counter[4*n+2] += fun(function, line, m, n)
                    num[4*n+2] += 1
                    line = normalization(m, n, chr(4 * n+2 + 97), line)
                    lists[4*n + 2].append(line)
                elif n >= 4 and line[2] < 4 * h:
                    counter[4*n+3] += fun(function, line, m, n)
                    num[4*n+3] += 1
                    line = normalization(m, n, chr(4 * n+3 + 97), line)
                    lists[4*n + 3].append(line)
                elif n >= 5 and line[2] < 5 * h:
                    counter[4*n+4] += fun(function, line, m, n)
                    num[4*n+4] += 1
                    line = normalization(m, n, chr(4 * n+4 + 97), line)
                    lists[4*n + 4].append(line)
    # sum = 0  # sum是物体数目之和
    for i in range(len(counter)):
        if num[i] != 0:
            counter[i] = counter[i] / num[i]  # counter里是该区域物体的平均大小
            # sum += num[i]

    # for index in range(len(num)):
    #     num[index] = num[index] / sum  # num里是该区域物体数目占所有物体数目的比例
    return counter, num, lists


def lists_save(txt, lists):
    if txt != 'merge.txt':
        txt = txt[:-4]
    else:
        txt = ''
    count = 0
    for l in lists:
        path = './labels/' + txt + '%c' % chr(count+97) + '.txt'
        with open( path, 'w') as outfile:
            np.savetxt(outfile, l, fmt='%f', delimiter=' ')
            count += 1


def mergetxt(path):
    # 合并多个txt, path不需要最后一个斜杠
    # 获取当前文件夹中的文件名称列表
    filenames = os.listdir(path)
    result = "./labels/merge.txt"
    # 打开当前目录下的result.txt文件，如果没有则创建
    file = open(result, 'w+', encoding="utf-8")
    # 向文件中写入字符
    # 先遍历文件名
    for filename in filenames:
        filepath = path + '/'
        filepath = filepath + filename
        # 遍历单个文件，读取行数
        for line in open(filepath, encoding="utf-8"):
            file.writelines(line)
        # file.write('\n')
    # 关闭文件
    file.close()


def position(name, width, height, cut):
    length = len(name)
    for i in range(length):
        a, b = cut[i]
        p = ord(name[i]) - 97
        if i == 0:
            x = p // b
            y = p % b
            lx = width / a * x
            ly = height / b * y
            rx = width / a * (x + 1)
            ry = height / b * (y + 1)
            width = rx-lx
            height = ry-ly
            continue
        else:
            x = p // b
            y = p % b
            lx = width / a * x + lx
            ly = height / b * y + ly
            rx = width / a + lx
            ry = height / b + ly
            width = rx-lx
            height = ry-ly
    return int(lx), int(ly), int(rx), int(ry)


def mask(lx, ly, rx, ry, img):
    for i in range(ly, ry):
        img[i][lx] = [255, 255, 255]
        img[i][rx] = [255, 255, 255]
        # img[i][lx] = [0, 0, 0]
        # img[i][rx] = [0, 0, 0]
    for i in range(lx, rx):
        img[ly][i] = [255, 255, 255]
        img[ry][i] = [255, 255, 255]
        # img[ly][i] = [0, 0, 0]
        # img[ry][i] = [0, 0, 0]
    return img


def videoShow(dic, cut):
    cap = cv2.VideoCapture('D:/python/DATA/exp10/2.mp4')
    while (cap.isOpened()):
        ret, img = cap.read()
        for x in dic.values():
            for y in x:
                lx, ly, rx, ry = position(y, 1280, 720, cut)
                rx = rx - 1
                ry = ry - 1
                # print(lx,ly,rx,ry)
                img = mask(lx, ly, rx, ry, img)
        cv2.imshow('1', img)
        cv2.waitKey(0)


def picShow(dic, cut):
    img = cv2.imread(r"D:\Python\DATA\0020.jpg")
    for x in dic.values():
        for y in x:
            lx, ly, rx, ry = position(y, 1280, 720, cut)
            rx = rx - 1
            ry = ry - 1
            img = mask(lx, ly, rx, ry, img)
    cv2.imshow('1', img)
    cv2.waitKey(0)


def cluster(dic):
    # 需要优化，仅满足四叉树
    lists = dic[3]
    length = len(lists)
    i = 0
    while i < length - 3:
        head = lists[i][:2]
        if lists[i] == head + 'a' and lists[i + 1] == head + 'b' and lists[i + 2] == head + 'c' and lists[
             i + 3] == head + 'd':
            del lists[i:i + 4]
            lists.append(head)
            length -= 3
        else:
            i += 1
    lists = dic[4]
    length = len(lists)
    i = 0
    while i < length - 3:
        head = lists[i][:3]
        if lists[i] == head + 'a' and lists[i + 1] == head + 'b' and lists[i + 2] == head + 'c' and lists[
             i + 3] == head + 'd':
            del lists[i:i + 4]
            lists.append(head)
            length -= 3
        else:
            i += 1
    return dic


def command(dic, cut, QP, src_path, result_path, name):
    print(QP)
    L = '/home/eynnzerr/h264_qpblock/build/h264_qpblock '
    L += src_path + ' ' + result_path + ' -baseqp ' + str(QP[0])
    # change the name of video

    for x in dic.keys():
        if x == 0:
            continue
        for y in dic[x]:
            lx, ly, rx, ry = position(y, 1280, 720, cut)
            qp = QP[x]
            L += ' ' + str(lx) + ',' + str(ly) + ',' + str(rx) + ',' + str(ry) + ':' + str(qp)
    # print(L)
    begin = time.time_ns()
    # os.system(L)
    end = time.time_ns()
    code = (end - begin)/1000000000
    command = 'python /home/eynnzerr/YOLOV5/yolov5/detect.py '
    command += ' --source ' + result_path + ' --name ' + name
    print(command)
    # change the name of the video
    # os.system(command)
    return code


def quarter(QP, src_path, testpath, name, cut):
    path = "./labels/"  # 需要最后一个斜杠
    txt = 'merge.txt'  # 需要.txt
    dic = {0: [], 1: [], 2: [], 3: [], 4: []}

    counter, num, lists = statistic(path, txt, cut[0][0], cut[0][1], 'relativesize', 2)
    # counter里是物体平均大小，num里是物体数目占所有的比例
    lists_save(txt, lists)
    for i in range(cut[0][0]*cut[0][1]):
        if num[i] == 0:
            dic[0].append(chr(i+97))
        else:
            dic[1].append(chr(i+97))
    # print('离线阶段')
    # print(dic)
    # print(counter)
    # print(num)

    sum = 0
    for i in num:
        sum += i

    counter_1, num_1 = [], []
    for i in range(cut[0][0]*cut[0][1]):
        if counter[i] < 1/3 and num[i] > sum/(cut[0][0]*cut[0][1]):
            dic[1].remove(chr(i+97))
            txt = chr(i + 97)+'.txt'
            counter_1_temp, num_1_temp, lists_1 = statistic(path, txt, cut[1][0], cut[1][1], 'relativesize', 2)
            for j in range(cut[1][0]*cut[1][1]):
                if num_1_temp[j] == 0:
                    dic[0].append(chr(i+97) + chr(j+97))
                else:
                    dic[2].append(chr(i+97) + chr(j+97))
            counter_1.append(counter_1_temp)
            num_1.append(num_1_temp)
            lists_save(txt, lists_1)
    # print('第一层')
    # print(dic)
    # print(counter_1)
    # print(num_1)

    keys = dic[2].copy()
    length = len(keys)
    counter_2, num_2 = [], []
    for i in range(length):
        index1 = i // (cut[1][0]*cut[1][1])
        index2 = i % (cut[1][0]*cut[1][1])
        if counter_1[index1][index2] < 1/3 and num_1[index1][index2] > sum/(cut[0][0]*cut[0][1]*cut[1][0]*cut[1][1]):
            temp = keys[i]
            dic[2].remove(keys[i])
            txt = temp + '.txt'
            counter_2_temp, num_2_temp, lists_2 = statistic(path, txt, cut[2][0], cut[2][1], 'relativesize', 2)
            for j in range(cut[2][0]*cut[2][1]):
                if num_2_temp[j] == 0:
                    dic[1].append(temp + chr(j+97))
                else:
                    dic[3].append(temp + chr(j+97))
            counter_2.append(counter_2_temp)
            num_2.append(num_2_temp)
            lists_save(txt, lists_2)
    # print('第二层')
    # print(dic)
    # print(counter_2)
    # print(num_2)

    keys = dic[3].copy()
    length = len(keys)
    counter_3, num_3 = [], []
    for i in range(length):
        index1 = i // 4
        index2 = i % 4
        if counter_2[index1][index2] < 1/3 and num_2[index1][index2] > sum/(cut[0][0]*cut[0][1]*cut[1][0]*cut[1][1]*cut[2][0]*cut[2][1]):
            temp = keys[i]
            dic[3].remove(keys[i])
            # dic[4].append(temp)
            txt = temp + '.txt'
            counter_3_temp, num_3_temp, lists_3 = statistic(path, txt, 2, 2, 'relativesize', 2)
            for j in range(4):
                if num_3_temp[j] == 0:
                    dic[2].append(temp + chr(j + 97))
                else:
                    dic[4].append(temp + chr(j + 97))
            counter_3.append(counter_3_temp)
            num_3.append(num_3_temp)
            lists_save(txt, lists_3)
    # print('第三层')
    # print(dic)
    # print(counter_3)
    # print(num_3)

    # dic = cluster(dic)
    # print(dic)

    code = command(dic, cut, QP, src_path, testpath, name)

    picShow(dic, cut)
    # videoShow(dic, cut)
    return code


# 根据DNA解码得到的QP方案进行视频编码，并求得适应度。由于适应度为正向函数，最后需数据逆向化
def get_fitness(pop, cut):
    threshold = 0.5
    stdtxt = '/home/eynnzerr/YOLOV5/yolov5/runs/detect/exp/labels/'  # 标注txt路径,高质量
    testtxt = '/home/eynnzerr/YOLOV5/yolov5/runs/detect/test/labels/'  # 测试txt路径,低质量
    stdpath = '0_10'
    testpath = 'test'
    label = 2
    src_path = '/home/eynnzerr/YOLOV5/yolov5/data/output.mp4'
    videopath = '/home/eynnzerr/YOLOV5/yolov5/data/video/test.mp4'
    name = 'test'

    parameter = -1/8
    bandwidth = 300000
    T = 3.5

    fs = np.zeros(pop.shape[0])
    for j, i in enumerate(pop):
        code = quarter(i, src_path, videopath, name, cut)
        score = F1_score(stdtxt, testtxt, stdpath, testpath, label, threshold)
        print(score)
        video_size = os.path.getsize(videopath)
        print((video_size/bandwidth) + code - T)
        f = score + parameter * max(((video_size/bandwidth) + code - T), 0)
        print(f)
        fs[j] = f

    return (np.max(fs) - fs) + 1e-3
    # x, y = translateDNA(pop)
    # # pred = Rosenbrock(x, y)
    # pred = 1
    # # 逆向化。 原[min, max]，现[0, max - min]
    # return (np.max(pred) - pred) + 1e-3  # 逆向化


# 对DNA的信息解码得到有用的信息。因不同求解任务而异
def translateDNA(pop):  # pop表示种群矩阵，一行表示一个个体的DNA，矩阵的行数为种群数目
    pass


# 产生后代的同时产生基因重组与基因突变
def crossover_and_mutation(pop):
    new_pop = []
    for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
        child = father  # 孩子先得到父亲的全部基因(QP选择方案)
        if np.random.rand() < CROSSOVER_RATE:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
            mother = pop[np.random.randint(POP_SIZE)]  # 种群中选择另一个个体，并将该个体作为母亲
            cross_points = np.random.randint(low=0, high=DNA_SIZE)  # 随机产生交叉的点
            child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
        mutation(child)  # 每个后代有一定的机率发生变异
        new_pop.append(child)

    return np.array(new_pop)  # 最终返回的仍是ndarray


def mutation(child):
    if np.random.rand() < MUTATION_RATE:  # 以MUTATION_RATE的概率进行变异
        mutate_point = np.random.randint(0, DNA_SIZE)  # 随机产生一个实数，代表要变异基因的位置
        child[mutate_point] += np.random.randint(-5, 5)  # 该位置qp随机突变为其它值
        if child[mutate_point] < 0:
            child[mutate_point] = 0
        elif child[mutate_point] > 51:
            child[mutate_point] = 51


# 自然选择。根据适应度对种群优胜劣汰
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                           p=fitness / (fitness.sum()))
    return pop[idx]


def print_info(pop):
    # fitness = get_fitness(pop)
    # max_fitness_index = np.argmax(fitness)
    # print("max_fitness:", fitness[max_fitness_index])
    # x, y = translateDNA(pop)
    # print("best gene type：", pop[max_fitness_index])
    # print("(x, y):", (x[max_fitness_index], y[max_fitness_index]))
    # # print("best solution: ", Rosenbrock(x[max_fitness_index], y[max_fitness_index]))
    # fitness = get_fitness(pop)
    print(pop)
    # print(fitness)


# 初始种群
'''
DNA_SIZE DNA 序列长度。DNA为ndarray，表示一种可行的QP选择方案
POP_SIZE 种群个体数目
CROSSOVER_RATE 交叉发生概率
MUTATION_RATE 突变概率。突变的意义在于防止陷入局部最优
N_GENERATIONS 种群繁衍迭代次数
'''
DNA_SIZE = 4
POP_SIZE = 20
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.08
N_GENERATIONS = 5


def sortnp(pop):
    for index, item in enumerate(pop):
        pop[index] = abs(np.sort(-item))


def pipline():
    count = 0
    scores = []
    times = []
    while count < 20:
        mergetxt('/home/eynnzerr/YOLOV5/yolov5/runs/detect/%d/labels' % count)
        videopath = '/home/eynnzerr/YOLOV5/yolov5/data/chunk_%d.mp4' % count  # 原视频地址
        store_path = '/home/eynnzerr/YOLOV5/yolov5/data/video/chunk_%d.mp4' % count  # 处理后视频地址
        name = '%d' % count  # yolo检测后的文件夹名称
        # cut = [[5, 5], [2, 3], [2, 1]]
        cut = [[4, 3], [2, 2], [2, 2], [2, 2]]
        t = quarter([48, 30, 16, 13, 10], videopath, store_path, name, cut)
        times.append(times)
        print("Process time: %.2f s" % t)
        stdtxt = '/home/eynnzerr/YOLOV5/yolov5/runs/detect/%d/labels/' % count  # 标注txt路径,高质量
        testtxt = '/home/eynnzerr/YOLOV5/yolov5/runs/detect/%d/labels/' % count  # 测试txt路径,低质量
        stdpath = 'output'
        testpath = 'dds_1'
        label = 2
        threshold = 0.5
        score = F1_score(stdtxt, testtxt, stdpath, testpath, label, threshold)
        scores.append(score)
        print("F1 score is ", score)


if __name__ == '__main__':
    # mergetxt('/home/eynnzerr/YOLOV5/yolov5/runs/detect/dds_1/labels')
    # videopath = '/home/eynnzerr/YOLOV5/yolov5/data/output_1.mp4'  # source name
    # store_path = '/home/eynnzerr/YOLOV5/yolov5/data/video/dds_1.mp4'  # restore name
    # name = 'dds_1_test'  # yolo file name
    # # cut = [[5, 5], [2, 3], [2, 1]]
    # cut = [[4, 3], [2, 2], [2, 2], [2, 2]]
    # start = time.time_ns()
    # t = quarter([48, 30, 16, 13, 10], videopath, store_path, name, cut)
    # end = time.time_ns()
    # print("Process time: %.2f s" % t)
    # stdtxt = '/home/eynnzerr/YOLOV5/yolov5/runs/detect/dds_1/labels/'  # 标注txt路径,高质量
    # testtxt = '/home/eynnzerr/YOLOV5/yolov5/runs/detect/dds_1_test/labels/'  # 测试txt路径,低质量
    # stdpath = 'output'
    # testpath = 'dds_1'
    # label = 2
    # threshold = 0.5
    # score = F1_score(stdtxt, testtxt, stdpath, testpath, label, threshold)
    # print("F1 score is ", score)

    pipline()

    #
    # QP的取值应为[0, 51]
    # cut = [[4, 3], [2, 2], [2, 2]]
    # pop = np.random.randint(52, size=(POP_SIZE, DNA_SIZE))
    # data = open("/home/eynnzerr/YOLOV5/yolov5/QPselector/gene_1.txt", 'a', encoding='utf-8')
    # pop = np.array([[48, 19, 17, 3], [47, 40, 24, 10], [46, 28, 27, 19], [47, 45, 28, 10], [48, 19, 17, 3],
    #                 [47, 42, 38, 17], [48, 44, 17,  3], [44, 40, 19,  9], [40, 25, 19,  6], [47, 42, 38, 17],
    #                 [41, 28, 24, 10], [48, 19, 17,  3], [29, 25, 23, 15], [40, 19, 17, 3], [42, 31, 24, 10],
    #                 [40, 25, 19, 15], [48, 44, 17,  3], [29, 25, 23, 15], [48, 19, 17, 3], [46, 28, 27, 19]])
    # sortnp(pop)
    # for _ in range(N_GENERATIONS):
    #     pop = crossover_and_mutation(pop)  # 基因重组与突变
    #     sortnp(pop)
    #     print(pop, file=data)
    #     fitness = get_fitness(pop, cut)  # 对经过一次繁殖后的种群获取适应度
    #     print(fitness, file=data)
    #     pop = select(pop, fitness)  # 根据适应度选择生成新的种群
    #     print(pop, file=data)
    #     # TODO 可以设定阈值提前终止
    #
    # print_info(pop)



