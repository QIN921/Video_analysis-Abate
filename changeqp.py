import os
import time

begin = time.time_ns()
command = 'ffmpeg -i D:/Python/DATA/0_10.mp4 -qp 50 D:/Python/DATA/result/50.mp4'
os.system(command)
end = time.time_ns()
print("Process time: %.2f ms" % ((end - begin)/1000000))
