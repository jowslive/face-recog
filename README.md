# ------------------------------------------------------------------
#											RECONHECIMENTO DE ROSTO E DETECÇÃO DE ROSTO
# -------------------------------------------------------------------

# - IMPORTANTE: PARA EXECUTAR AS APLICAÇÕES VOCÊ PRECISA TER A OPENCV CONTRIB INSTALADA -

#Detector_Video.py: 	      Este arquivo detecta rostos usando cascatas de Haar. Funciona bem com vários rostos.


#Face_Capture_With_Rotate.py: A execução deste arquivo irá capturar 50 imagens de uma pessoa na frente da câmera. Isso fará com que as fotos não sejam escuras e também fará com que o rosto fique "reto".


#Free_Rotate.py:  Este arquivo mostra a função de rotação. Lembrar de remover o comentário da linha 153 em NameFind.py. Isso mostrará a imagem corrigindo o deslocamento.


 
#NameFind.py:     Este arquivo contém todas as funções.


#Trainer_All.py:  Este arquivo irá treinar todos os algoritmos de reconhecimento usando as imagens na pasta dataSet.


#Recogniser_Image_All_Algorithms.py: Esta aplicação irá detectar e reconhecer rostos de imagens. Imagens diferentes podem ser selecionadas.


#Recogniser_Video_EigenFace.py:  Este arquivo é o reconhecimento de rostos da alimentação da câmera usando o algoritmo Eigen face.


#Recogniser_video_FisherFace.py: Este arquivo é o reconhecimento de faces da alimentação da câmera usando o algoritmo de face da Fisher.


#Recogniser_Video_LBPHFace.py:   Este arquivo é o reconhecimento de rostos da alimentação da câmera usando o algoritmo de face LBPH.


#TestDataCollector_EiganFace.py: Este arquivo é o aplicativo de teste. Ele irá receber uma imagem que o conjunto de dados será carregado. Um loop será executado 200 vezes a cada vez, aumentando o número de componentes. Cada vez que um reconhecedor facial Eigen for treinado e previsto na imagem de entrada. Depois que o loop for for concluído, ID e confiança serão plotados.


#TestDataCollector_EiganFace.py: Este arquivo é o aplicativo de teste. Ele irá receber uma imagem que o conjunto de dados será carregado. Um loop será executado 200 vezes a cada vez, aumentando o número de componentes. Cada vez que um reconhecedor de rosto Fisher for treinado e previsto na imagem de entrada. Depois que o loop for for concluído, ID e confiança serão plotados.


#TestDataCollector_EiganFace.py: Este arquivo é o aplicativo de teste. Ele irá receber uma imagem que o conjunto de dados será carregado. Um loop será executado 54, 13, 50 #vezes. A cada vez que são criados os parâmetros. Cada vez que um reconhecedor facial LBPH será treinado e previsto na imagem de entrada. Depois que o loop for for completado, ID e confiança serão plotados.

# ------------ PASTAS ------------

#dataSet --> Contém as imagens que serão usadas para treinar o reconhecedor.

#FlowCharts --> Contém fluxograma projetado usando o Microsoft Visio e arquivos png

#Haar --> Contém as Cascatas Haar do OpenCV usadas nas aplicações

#Plots --> Contém os gráficos obtidos usando Me4.jpg e Sam.jpg

#Recogniser --> Contém os arquivos XML salvos por reconisers

#SaveData --> Contém os dados salvos pelos aplicativos testadores


