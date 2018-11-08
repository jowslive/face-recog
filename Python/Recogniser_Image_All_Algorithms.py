# ------------------------------ RECONHECEDOR FACIAL PARA TODOS OS ALGORITMOS  ---------------------------------

import cv2                  #  Importando o OPENCV
import numpy as np          #  Importando o Numerical Python
import NameFind

# ---------------------------------------------------------------------------------------
# Importando os arquivos Haar Cascades para reconhecimento da face e olho

face_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_frontalcatface.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_eye.xml")
spec_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_eye_tree_eyeglasses.xml")


# ---------------------------------------------------------------------------------------
# Objeto reconhecedor de face
LBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 20)

# ---------------------------------------------------------------------------------------
# Carregando os dados de treinamento feitos pelo treinador para reconhecer as faces

LBPH.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataLBPH.xml")


# ---------------------------------------------------------------------------------------
# ------------------------------------  ADICIONANDO A FOTO  -----------------------------------------------------
img = cv2.imread("C:/Users/joaob/Documents/Github/face-recog/Python/dataSet/User.3.6.jpg")    #Caminho da foto
imgx = cv2.imread("C:/Users/joaob/Documents/Github/face-recog/Python/dataSet/User.3.5.jpg")   #Caminho da foto

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Converte a foto para escala de cinza
grayx = cv2.cvtColor(imgx, cv2.COLOR_BGR2GRAY)              # Converte a segunda foto para escala de cinza

faces = face_cascade.detectMultiScale(gray, 1.3, 4)         # Detecta as faces e guarda as posições
facesx = face_cascade.detectMultiScale(grayx, 1.3, 4)       # Detecta as faces na segunda foto e guarda as posições
print(faces)
print(facesx)

#    IMAGEM 1
for (x, y, w, h) in faces:                                  # EIXOS X, Y, LARGURA, ALTURA

    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))   # A face é isolada e cortada

    ID, conf = LBPH.predict(Face)                           # MÉTODO LBPH RECONHECIMENTO FACIAL
    print(ID)
    NAME = NameFind.ID2Name(ID, conf)
    NameFind.DispID(x, y, w, h, NAME, gray)

#    IMAGEM 2 
for (x, y, w, h) in facesx:                                   # EIXOS X, Y, LARGURA, ALTURA

    Facex = cv2.resize((grayx[y: y+h, x: x+w]), (110, 110))   # A face é isolada e cortada

    ID, confx = LBPH.predict(Facex)                           # MÉTODO LBPH RECONHECIMENTO FACIAL
    print(ID)
    NAMEX = NameFind.ID2Name(ID, confx)
    NameFind.DispID(x, y, w, h, NAMEX, grayx)


crop_img = gray[y:y+h, x:x+w]
crop_imgx = grayx[y:y+h, x:x+w]


cv2.imshow('LBPH Face 1', gray)            # Mostrando a imagem
cv2.imshow('LBPH Face 2', grayx)           # Mostrando a imagem 2
cv2.waitKey()
cv2.destroyAllWindows()
