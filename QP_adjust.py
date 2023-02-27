import os

command = "ffmpeg -i "
input = " ./video1/10_20.mp4 "
output = " ./video1/low.mp4 "

# act = " -filter_complex addroi=iw/5:ih*3/5:iw*3/5:ih/5:-1 "
# act = " -filter_complex addroi=0:0:iw:ih*2/5:1 "
# act = " -qp 45 -vf crop=iw:ih*2/5:0:0 "
# act = " -qp 18 -vf crop=iw:ih*3/5:0:ih*2/5 "
act = " -qp 50 "

final = command + input + act + output

# final_act = "ffmpeg -i ./video1/qp1.mp4 -i ./video1/qp2.mp4 -filter_complex " \
#             "'pad=1280:720[x0]; [0:v]scale=1280:288[inn0]; [x0][inn0]overlay=0:0[x1]; [1:v]scale=1280:432[inn1]; [x1][inn1]overlay=0:288'" \
#             " ./video1/sum.mp4"
a = os.system(final)

print(a)

