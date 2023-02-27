import os
import cv2
import ffmpeg


def change_qp(src: str, qp: int, des: str):
    # 调整视频的QP
    command = 'ffmpeg -i source -qp num destination'
    command = command.replace('source', src).replace('num', str(qp)).replace('destination', des)
    print(command)
    inf = os.system(command)
    # print(inf)


def change_qp_crop(src: str, qp: int, pos: list[str], des: str):
    # 调整视频的QP并截取视频的部分区域
    # pos里依此是 x坐标比例，y坐标比例，长比例，宽比例 的字符串
    # crop里依次是 长，宽，x坐标，y坐标
    command = 'ffmpeg -i source -qp num -vf crop=iw*w_ratio:ih*h_ratio:iw*w_origin_ratio:ih*h_origin_ratio destination'
    command = command.replace('source', src).replace('num', str(qp)).replace('destination', des)\
        .replace('w_ratio', pos[2]).replace('h_ratio', pos[3])\
        .replace('w_origin_ratio', pos[0]).replace('h_origin_ratio', pos[1])
    print(command)
    inf = os.system(command)
    # print(inf)


def video_joint(src1: str, src2: str, size: list[int], pos: list[str], des: str):
    # 将两个视频拼接在一起
    # source1 为完整画面， source2 为部分画面
    # 将小画面source2拼接到大画面source1上
    # size为视频大小 [iw, ih]
    # pos里是source2对应的位置，依此是 x坐标比例，y坐标比例，长比例，宽比例 的字符串
    command = "ffmpeg -i source1 -i source2 -filter_complex \" pad=width:height[x0]; " \
              "[0:v]scale=width:height[inn0]; [x0][inn0]overlay=0:0[x1]; " \
              "[1:v]scale=width*w_ratio:height*h_ratio[inn1]; " \
              "[x1][inn1]overlay=width*w_origin_ratio:height*h_origin_ratio \" -qp 48 destination"
    command = command.replace('source1', src1).replace('source2', src2).replace('destination', des)\
        .replace('width', str(size[0])).replace('height', str(size[1])) \
        .replace('w_ratio', pos[2]).replace('h_ratio', pos[3]) \
        .replace('w_origin_ratio', pos[0]).replace('h_origin_ratio', pos[1])
    print(command)
    inf = os.system(command)
    # print(inf)


def get_vedio_height_width(filename: str) -> list[int]:
    cap = cv2.VideoCapture(filename)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(width, height)
    return [width, height]


def get_video_kbps(src: str):
    probe = ffmpeg.probe(src)
    format = probe['format']
    bit_rate = format['bit_rate']
    kbps = int(bit_rate) / 1000
    return kbps


def all_qp():
    video = 'D:/python/DATA/10_20.mp4'
    size = get_vedio_height_width(video)
    pos = ['2/5', '1/5', '1/5', '2/5']
    des = 'D:/python/DATA/video9/'
    # for most in range(1, 51):
    #     mostVideo = des + 'all' + str(most) + '.mp4'
    #     change_qp(video, most, mostVideo)
    # for part in range(1, 51):
    #     partVideo = des + 'part' + str(part) + '.mp4'
    #     change_qp_crop(video, part, pos, partVideo)
    # for most in range(1, 25):
    #     most = 2 * most
    #     for part in range(most, 25):
    #         part = 2 * part
    #         mostVideo = des + 'all' + str(most) + '.mp4'
    #         partVideo = des + 'part' + str(part) + '.mp4'
    #         joint = des + str(most) + '_' + str(part) + '.mp4'
    #         video_joint(mostVideo, partVideo, size, pos, joint)



if __name__ == '__main__':
    # all_qp()
    src1 = "D:/python/DATA/video8/all48.mp4"
    src2 = "D:/python/DATA/video8/part2.mp4"
    des = "D:/python/DATA/result.mp4"
    size = get_vedio_height_width(src1)
    pos = ['2/5', '1/5', '1/5', '2/5']
    video_joint(src1, src2, size, pos, des)

