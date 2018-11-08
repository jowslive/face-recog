# ---------------------------------------------------------------------------------------------------------------------
#											RECONHECIMENTO DE ROSTO E DETECÇÃO DE ROSTO
# ---------------------------------------------------------------------------------------------------------------------

#NameFind.py:     Este arquivo contém todas as funções.


#Trainer_All.py:  Este arquivo irá treinar todos os algoritmos de reconhecimento usando as imagens na pasta dataSet.


#Recogniser_Image_All_Algorithms.py: Esta aplicação irá detectar e reconhecer rostos de imagens. Imagens diferentes podem ser selecionadas.



#TestDataCollector_LBPH.py: Este arquivo é o que testará as imagens fornecidas em uma pasta depois de elas terem sido treinadas pelo arquivo Trainer_ALL.py . Ele irá receber uma imagem em que o conjunto de dados será carregado. Um loop será executado +- 200 vezes e a cada vez será aumentando o número de componentes. Cada vez que o método LBPH reconhecedor de rosto  for treinado, será previsto na imagem de entrada. Depois que o loop for concluído, o ID e a confiança serão salvos e mostrados na tela.


# ------------ PASTAS ------------

#dataSet --> Contém as imagens que serão usadas para treinar o reconhecedor.

#Haar --> Contém as Cascatas Haar do OpenCV usadas nas aplicações

#Recogniser --> Contém os arquivos XML salvos pelos reconhecedores

#SaveData --> Contém os dados salvos pelos aplicativos testadores

#Testes --> Imagens aleatórias para possíveis testes    


# ------------ LEMBRETES ------------

#Detecção de rosto: it has the objective of finding the faces (location and size) in an image and probably extract them to be used by the face recognition algorithm.
#Tem o objetivo de achar as faces ( posições e tamanhos ) em uma imagem e provavelmnete extrair essas informações para serem usadas em um algoritmo de reconhecimento facial

#Reconhecimento facial: Com as faces já extraídas , recortadas , redimensionadas e usualmente convertidas para escala de cinza , o algoritmo de reconhecimento facial é responsável por achar características que melhor descrevem a imagem.

#Basicamente compara as imagens que contem um rosto com todas as faces de uma pasta chamada Dataset com a intenção de achar algum usuário que de "match" com essa face. É uma comparação 1 para N (1xN) .


#Raio : o raio é usado para construir o padrão binário local circular e representa o raio ao redor do pixel central. Geralmente está definido como 1.

#Vizinhos : o número de pontos de amostra para construir o padrão binário local circular. Quanto mais pontos de amostra , maior será o custo computacional. Geralmente está definido como 8.

#Grid X : o número de células na direção horizontal. Quanto mais células, quanto mais fina a grade, maior a dimensionalidade do vetor de recursos resultante. Geralmente está definido como 8.

#Grid Y : o número de células na direção vertical. Quanto mais células, quanto mais fina a grade, maior a dimensionalidade do vetor de recursos resultante. Geralmente está definida como 8.

#*imagem da area de trabalho mostrando como funciona o processo após o treinamento*.

# Ordem de execução : Trainer_All.py -> TestDataCollector_LBPH.py -> NameFind.py -> Recogniser_Image_All_Algorithms.py