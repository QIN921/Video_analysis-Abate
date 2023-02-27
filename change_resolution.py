import cv2

# TODO 更改视频分辨率

cap = cv2.VideoCapture("./video/ten_second.mp4")
videowriter = cv2.VideoWriter("./video/ten_256_144"+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (256, 144))

success, _ = cap.read()
COUNT = 0
while success:
    success, img1 = cap.read()
    COUNT += 1
    print(COUNT)
    try:
         img = cv2.resize(img1, (256, 144), interpolation=cv2.INTER_LINEAR)
         videowriter.write(img)
    except:
         break