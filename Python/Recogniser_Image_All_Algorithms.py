# ------------------------------ RECONHECEDOR FACIAL PARA TODOS OS ALGORITMOS  ---------------------------------

import cv2                  #  Importando o OPENCV
import numpy as np          #  Importando o Numerical Python - Não necessário
import NameFind

# ---------------------------------------------------------------------------------------
# Importando os arquivos Haar Cascades para reconhecimento da face e olho

#JÃO
face_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_frontalcatface.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_eye.xml")
spec_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_eye_tree_eyeglasses.xml")

#ALEK's

#face_cascade = cv2.CascadeClassifier("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Haar/haarcascade_frontalcatface.xml")
#eye_cascade = cv2.CascadeClassifier("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Haar/haarcascade_eye.xml")
#spec_cascade = cv2.CascadeClassifier("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Haar/haarcascade_eye_tree_eyeglasses.xml")

# ---------------------------------------------------------------------------------------
# Objeto reconhecedor de face
LBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 20)
#EIGEN = cv2.face.EigenFaceRecognizer_create(10, 5000)
#FISHER = cv2.face.FisherFaceRecognizer_create(5, 500)

# ---------------------------------------------------------------------------------------
# Carregando os dados de treinamento feitos pelo treinador para reconhecer as faces

#JÃO
LBPH.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataLBPH.xml")
#EIGEN.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataEigan.xml")
#FISHER.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataFisher.xml")

#ALEK'S
#LBPH.read("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Recogniser/trainingDataLBPH.xml")
#EIGEN.read("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Recogniser/trainingDataEigan.xml")
#FISHER.read("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Recogniser/trainingDataFisher.xml")

# ---------------------------------------------------------------------------------------
# ------------------------------------  ADICIONANDO A FOTO  -----------------------------------------------------
# Endereço / Caminho da foto
#img = cv2.imread("C:/Users/Alek's/Documents/GitHub/face-recog/Python/1_1.jpg")      
img = cv2.imread("C:/Users/joaob/Documents/Github/face-recog/Python/User.20.40.jpg")    
imgx = cv2.imread("C:/Users/joaob/Documents/Github/face-recog/Python/User.20.32.jpg")    

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # Converte a foto para escala de cinza
grayx = cv2.cvtColor(imgx, cv2.COLOR_BGR2GRAY)              # Converte a foto para escala de cinza

faces = face_cascade.detectMultiScale(gray, 1.3, 4)         # Detecta as faces e guarda as posições
facesx = face_cascade.detectMultiScale(grayx, 1.3, 4)       # Detecta as faces e guarda as posições
print(faces)
print(facesx)

#    IMAGEM 1
for (x, y, w, h) in faces:                                  # EIXOS X, Y, LARGURA, ALTURA

    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))   # A face é isolada e cortada

    ID, conf = LBPH.predict(Face)                           # MÉTODO LBPH RECONHECIMENTO FACIAL
    print(ID)
    NAME = NameFind.ID2Name(ID, conf)
    NameFind.DispID(x, y, w, h, NAME, gray)

#   ID, conf = EIGEN.predict(Face)                          # MÉTODO EIGEN RECONHECIMENTO FACIAL
#   NAME = NameFind.ID2Name(ID, conf)
#   NameFind.DispID3(x, y, w, h, NAME, gray)

#    IMAGEM 2 
for (x, y, w, h) in facesx:                                   # EIXOS X, Y, LARGURA, ALTURA

    Facex = cv2.resize((grayx[y: y+h, x: x+w]), (110, 110))   # A face é isolada e cortada

    ID, confx = LBPH.predict(Facex)                           # MÉTODO LBPH RECONHECIMENTO FACIAL
    print(ID)
    NAMEX = NameFind.ID2Name(ID, confx)
    NameFind.DispID(x, y, w, h, NAMEX, grayx)

#   ID, conf = FISHER.predict(Face)                        # MÉTODO FISHER RECONHECIMENTO
#   NAME = NameFind.ID2Name(ID, conf)
#   NameFind.DispID2(x, y, w, h, NAME, gray)

crop_img = gray[y:y+h, x:x+w]
crop_imgx = grayx[y:y+h, x:x+w]


cv2.imshow('LBPH Face 1', gray)            # Mostrando a imagem
cv2.imshow('LBPH Face 2', grayx)           # Mostrando a imagem 2
cv2.waitKey()
cv2.destroyAllWindows()
