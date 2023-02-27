import cv2

cap = cv2.VideoCapture(r"D:\python\video_process\experiment\video1\0_10.mp4")
knn_sub = cv2.createBackgroundSubtractorKNN()
# mog2_sub = cv2.createBackgroundSubtractorMOG2()
videowriter = cv2.VideoWriter("./video1/knn_sub"+".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # mog_sub_mask = mog2_sub.apply(frame)
    knn_sub_mask = knn_sub.apply(frame)

    # cv2.imshow('original', frame)
    # cv2.imshow('MOG2', mog_sub_mask)
    # cv2.imshow('KNN', knn_sub_mask)
    img = cv2.cvtColor(knn_sub_mask, cv2.COLOR_BGR2RGB)
    videowriter.write(img)

    key = cv2.waitKey(30) & 0xff
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()