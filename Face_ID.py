import cv2

image_path = '103.jpg' #imagem
cascade_path = 'haarcascade_frontalface_default.xml' #modelo de reconhecimento

clf = cv2.CascadeClassifier(cascade_path) #Cria o classificador

img = cv2.imread(image_path)#ler a imagem

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Conveter para preto e branco

face = clf.detectMultiScale(gray, 2.0, 10) #2.0,1 reconhce o douglas

for (x, y, w, h) in face: #percorrer o rosto
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255, 255,0),2) #marcaro o rsoto

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindowns()
