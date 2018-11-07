# ------------------------------ RECONHECEDOR FACIAL PARA TODOS OS ALGORITMOS  ---------------------------------

import cv2                   #   Importando o OPENCV
#import numpy as np          #  Importando o Numerical Python - Não necessário
import NameFind
import numpy as np
from matplotlib import pyplot as plt

# ---------------------------------------------------------------------------------------
# Importando os arquivos Haar Cascades para reconhecimento da face e olho

#JÃO
#face_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_frontalcatface.xml")
#eye_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_eye.xml")
#spec_cascade = cv2.CascadeClassifier("C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_eye_tree_eyeglasses.xml")

#ALEK's

face_cascade = cv2.CascadeClassifier("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Haar/haarcascade_frontalcatface.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Haar/haarcascade_eye.xml")
spec_cascade = cv2.CascadeClassifier("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Haar/haarcascade_eye_tree_eyeglasses.xml")

# ---------------------------------------------------------------------------------------
# Objeto reconhecedor de face
LBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 20)
#EIGEN = cv2.face.EigenFaceRecognizer_create(10, 5000)
#FISHER = cv2.face.FisherFaceRecognizer_create(5, 500)

# ---------------------------------------------------------------------------------------
# Carregando os dados de treinamento feitos pelo treinador para reconhecer as faces

#JÃO
#LBPH.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataLBPH.xml")
#EIGEN.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataEigan.xml")
#FISHER.read("C:/Users/joaob/Documents/Github/face-recog/Python/Recogniser/trainingDataFisher.xml")

#ALEK'S
LBPH.read("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Recogniser/trainingDataLBPH.xml")
#EIGEN.read("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Recogniser/trainingDataEigan.xml")
#FISHER.read("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Recogniser/trainingDataFisher.xml")

# ---------------------------------------------------------------------------------------
# ------------------------------------  ADICIONANDO A FOTO  -----------------------------------------------------
# Endereço / Caminho da foto
img = cv2.imread("C:/Users/Alek's/Documents/GitHub/face-recog/Python/Reconhecimento Facial/Nova Pasta/Aaron_Peirsol/Aaron_Peirsol_0002.jpg")
#img = cv2.imread("C:/Users/joaob/Documents/Github/face-recog/Python/1_1.jpg")


gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                # Converte a foto para escala de cinza
faces = face_cascade.detectMultiScale(gray, 1.3, 4)         # Detecta as faces e guarda as posições
#print(faces)

for (x, y, w, h) in faces:                                  # EIXOS X, Y, LARGURA, ALTURA

    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))   # A face é isolada e cortada

    ID, conf = LBPH.predict(Face)                           # MÉTODO LBPH RECONHECIMENTO FACIAL
    #print(ID)
    NAME = NameFind.ID2Name(ID, conf)
    NameFind.DispID(x, y, w, h, NAME, gray)

#   ID, conf = EIGEN.predict(Face)                         # MÉTODO EIGEN RECONHECIMENTO FACIAL
#   NAME = NameFind.ID2Name(ID, conf)
#   NameFind.DispID3(x, y, w, h, NAME, gray)

#   ID, conf = FISHER.predict(Face)                        # MÉTODO FISHER RECONHECIMENTO
#   NAME = NameFind.ID2Name(ID, conf)
#   NameFind.DispID2(x, y, w, h, NAME, gray)

crop_img = gray[y:y+h, x:x+w]

#cv2.imshow("cropped", crop_img)

#cv2.imshow('LBPH Face Recognition System', gray)            # Mostrando a imagem
#cv2.waitKey()
#cv2.destroyAllWindows()
############################################################################################################################################
a=img.shape[0]
l=img.shape[1]
def GetPixels(img):
    pixels=[]
    for x in range(x,l):
        row =[]
        for y in range(y,a):
            r, g, b, _ = img.At(x, y).RGBA()
            pixel = (float32(r) * 0.299) + (float32(g) * 0.587) + (float32(b) * 0.114)
            row = append(row,uint8(pixel))
            pixels = append(pixels, row)
            return pixels

def Calculate(img, radius, neighbors):
    lbpPixels=[x][y]
    pixels=GetPixels(img)
    for x in range(x,l):
        currentRow=[]
        for y in range(y,a):
            threshold = pixels[x][y]
            binaryResult = ""
            for tempX=(x-1) in range(x-1,x+2):
				for tempX=(y-1) in range(y-1, y+2):
					if tempX != x || tempY != y :
						binaryResult += getBinaryString(int(pixels[tempX][tempY]), threshold)




			// Convert the binary string to a decimal integer
			dec, err := strconv.ParseUint(binaryResult, 2, 64)
			if err != nil {
				return lbpPixels, errors.New("Error converting binary to uint in the ApplyLBP function")
			} else {
				// Append the decimal do the result slice
				// ParseUint returns a uint64 so we need to convert it to uint8
				currentRow = append(currentRow, uint64(dec))
			}
		}
		// Append the slice to the 'matrix'
		lbpPixels = append(lbpPixels, currentRow)
	}
	return lbpPixels, nil
}
