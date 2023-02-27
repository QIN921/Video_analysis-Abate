import cv2


cap = cv2.VideoCapture("./video/ten_second.mp4")
background = cv2.imread("./background.jpg")
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
videowriter = cv2.VideoWriter("./video/sub1"+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

success, img = cap.read()
COUNT = 0
while success:
    # if COUNT == 299: break
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = image - background
    image[image > 200] = 0
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    videowriter.write(img)
    print(COUNT)
    COUNT += 1
    success, img = cap.read()

