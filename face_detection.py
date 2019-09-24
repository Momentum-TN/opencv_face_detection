import cv2, glob

gimage = glob.glob("*.jpg")

detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

for timage in gimage:
    image = cv2.imread(timage)
    resized = cv2.resize(image, ( int(image.shape[1]/2), int(image.shape[0]/2) ) )
    grayimage = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
    face = detect.detectMultiScale(grayimage,1.05,5)
    
    for (x,y,w,h) in face:
        cv2.rectangle(resized, (x,y), (x+w,y+h), (0,255,0),2)
    cv2.imshow("image title",resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
