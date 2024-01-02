import cv2

cap = cv2.VideoCapture("/home/Videos/train/video.mp4")
count = 0
while True:
    ret, frame = cap.read()
    # cv2.imshow("capture", frame)
    cv2.imwrite("/home/Videos/train/seqs/%05d.png"%count, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    count = count+1
    if(count%100==0):
        print("save ", count, " pngs ...")
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
# cv2.destroyAllWindows()
