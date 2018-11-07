# ------------ COLETA A DATA USANDO O MÉTODO LBPH EM UMA CERTA IMAGEM E SALVA A DATA EM UM ARQUIVO DE TEXTO -------------
# ---------------------- SALVA A DATA GERADA EM 3 ARQUIVOS DE TEXTO E OS "LOTEIA" --------------------------------------


import os               # importing the OS for path
import cv2              # importing the OpenCV library
import numpy as np      # importing Numpy library
from PIL import Image   # importing Image library
import matplotlib.pyplot as plt # Importing Plot library
import NameFind

face_cascade = cv2.CascadeClassifier('C:/Users/joaob/Documents/Github/face-recog/Python/Haar/haarcascade_frontalcatface.xml')
path = 'dataSet'        # path to the photos

img = cv2.imread('C:/Users/joaob/Documents/Github/face-recog/Python/dataSet/User.2.1.jpg')        # -------------->>>>>>>>>>>>>>>>>>  The Image to be checked

def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    FaceList = []
    IDs = []

    for imagePath in imagePaths:
        faceImage = Image.open(imagePath).convert('L')          # Open image and convert to gray
        print(str((faceImage.size)))
        faceImage = faceImage.resize((110, 110))                # resize the image
        faceNP = np.array(faceImage, 'uint8')                   # convert the image to Numpy array
        print(str((faceNP.shape)))
        ID = int(os.path.split(imagePath)[-1].split('.')[1])    # Get the ID of the array
        FaceList.append(faceNP)                                 # Append the Numpy Array to the list
        IDs.append(ID)                                          # Append the ID to the IDs list

    return np.array(IDs), FaceList                              # The IDs are converted in to a Numpy array

face_number = 1
IDs, FaceList = getImageWithID(path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                   # Convert the Camera to gray
faces = face_cascade.detectMultiScale(gray, 1.3, 4)            # Detect the faces and store the positions
radTrain = open("C:/Users/joaob/Documents/Github/face-recog/Python/SaveData/LBPHLBPH_PIXEL_RADIUS.txt", "w+")   # open the file to write data
neiTrain = open("C:/Users/joaob/Documents/Github/face-recog/Python/SaveData/LBPHLBPH_NEIGHBOURS.txt", "w+")     # open the file to write data
cellTrain = open("C:/Users/joaob/Documents/Github/face-recog/Python/SaveData/LBPHLBPH_CELLS.txt", "w+")         # open the file to write data

for (x, y, w, h) in faces:
    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))
    radPix = 1
    rad_tabal_ID = []
    rad_tabal_conf = []
    # --------------------------- Run tests for the radius from the centre ----------------
    for _ in range(54):
        recog = cv2.face.LBPHFaceRecognizer_create(radPix)     # creating EIGEN FACE RECOGNISER
        print('Treinando para  ' + str(radPix) + ' pixels em relação ao centro')
        recog.train(FaceList, IDs)                            # The recogniser is trained using the images
        print('Método LBPH de reconhecimento facial treinado!')
        ID, conf = recog.predict(Face)
        rad_tabal_ID.append(ID)
        rad_tabal_conf.append(conf)
        radTrain.write(str(ID) + "," + str(conf) + "\n")
        print ("No raio: " + str(radPix) + " o ID é: " + str(ID) + " A confiança é: " + str(conf))
        radPix = radPix + 1
    # ---------------------------------------- 1ST PLOT -----------------------------------------------------
    plt.subplot(2, 1, 1)
    plt.plot(rad_tabal_ID)
    plt.title('ID contrário ao raio dos pixels', fontsize=10)
    plt.axis([0, radPix, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Raio (em pixels)', fontsize=8)
    p2 = plt.subplot(2, 1, 2)
    plt.plot(rad_tabal_conf, 'red')
    plt.title('Confiança em relação ao raio dos pixels', fontsize=10)
    p2.set_xlim(xmin=0)
    p2.set_xlim(xmax=radPix)
    plt.ylabel('Confiança', fontsize=8)
    plt.xlabel('Raio (em pixels)', fontsize=8)
    plt.tight_layout()
    plt.show()
    # ---------------------------  Run tests for the neighbours -----------------------------
    radPixel = input('Digite o raio de pixels ideal: ')    # ------>> CHANGE THE PIXEL RADIUS IF A BETTER VALUE IS FOUND
    neighbour = 1
    nei_ID = []
    nei_conf = []
    for _ in range(13):
        recog = cv2.face.LBPHFaceRecognizer_create(int(radPixel), int(neighbour))  # creating FACE RECOGNISER
        print('TRAINING FOR  ' + str(neighbour) + ' NEIGHBOURS')
        recog.train(FaceList, IDs)                                      # The recogniser is trained using the images
        print('LBPH FACE RECOGNISER TRAINED')
        ID, conf = recog.predict(Face)
        nei_ID.append(ID)
        nei_conf.append(conf)
        neiTrain.write(str(ID) + "," + str(conf) + '\n')
        print ('FOR RADIUS: ' + str(radPixel) + " AND " + str(neighbour) + "NEIGHBOURS, ID IS: " + str(ID) + " THE CONFIDENCE: " + str(conf))
        neighbour = neighbour + 1
    # ---------------------------------------- 2ND PLOT -----------------------------------------------------
    plt.subplot(2, 1, 1)
    plt.plot(nei_ID)
    plt.title('ID against number of neighbours', fontsize=10)
    plt.axis([0, neighbour, 10, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Number of neighbours', fontsize=8)
    p2 = plt.subplot(2, 1, 2)
    plt.plot(nei_conf, 'red')
    plt.title('ID against number of neighbours', fontsize=10)
    p2.set_xlim(xmin=0)
    p2.set_xlim(xmax=neighbour)
    plt.ylabel('Confidence', fontsize=8)
    plt.xlabel('Number of neighbours', fontsize=8)
    plt.tight_layout()
    plt.show()
    # ---------------------------  Run tests for the Cell Number -----------------------------
    neighbour = input('ENTER THE IDEAL NUMBER OF NEIGHBOURS ')   # ------>> CHANGE THE NEIGHBOUR IF BETTER VALUE IS FOUND
    cellVal = 1
    cell_ID = []
    cell_conf = []
    for _ in range(50):
        recog = cv2.face.LBPHFaceRecognizer_create(radPixel, neighbour, cellVal, cellVal)  # creating FACE RECOGNISER
        print('TRAINING FOR  ' + str(cellVal) + ' CELLS')
        recog.train(FaceList, IDs)                                          # The recogniser is trained using the images
        print('LBPH FACE RECOGNISER TRAINED')
        ID, conf = recog.predict(Face)
        cell_ID.append(ID)
        cell_conf.append(conf)
        cellTrain.write(str(ID) + "," + str(conf) + "\n")
        print ('FOR RADIUS: ' + str(radPixel) + " , " + str(neighbour) + "NEIGHBOURS AND CELL VALUE " + str(cellVal) + ", ID IS: " + str(ID) + " THE CONFIDENCE: " + str(conf))
        cellVal = cellVal + 1
    NameFind.tell_time_passed()


    # ------------------------------------------- ALL SIX PLOTS -----------------------------------------------------------
    plt.subplot(3, 2, 1)
    plt.plot(rad_tabal_ID)
    plt.title('ID against Pixel Radius', fontsize=10)
    plt.axis([0, 53, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Radius (Pixels)', fontsize=8)
    plt.subplot(3, 2, 2)
    plt.plot(rad_tabal_conf, 'red')
    plt.title('Confidence against Pixel Radius', fontsize=10)
    plt.ylabel('Confidence', fontsize=8)
    plt.xlabel('Radius (Pixels)', fontsize=8)
    plt.subplot(3, 2, 3)
    plt.plot(nei_ID)
    plt.title('ID against number of neighbours', fontsize=10)
    plt.axis([0, 12, 10, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Number of neighbours', fontsize=8)
    plt.subplot(3, 2, 4)
    plt.plot(nei_conf, 'red')
    plt.title('ID against number of neighbours', fontsize=10)
    plt.ylabel('Confidence', fontsize=8)
    plt.xlabel('Number of neighbours', fontsize=8)
    plt.subplot(3, 2, 5)
    plt.plot(cell_ID)
    plt.title('ID against number of cells', fontsize=10)
    plt.axis([0, 49, 0, 25])
    plt.ylabel('ID', fontsize=8)
    plt.xlabel('Number of Cells', fontsize=8)
    plt.subplot(3, 2, 6)
    plt.plot(cell_conf, 'red')
    plt.title('ID against number of cells', fontsize=10)
    plt.ylabel('Confidence', fontsize=8)
    plt.xlabel('Number of Cells', fontsize=8)
    plt.tight_layout()
    plt.show()

    face_number = face_number + 1

radTrain.close()
neiTrain.close()
cellTrain.close()
cv2.destroyAllWindows()