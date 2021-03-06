import time
import pickle                                       #pip install pickle-mixin  
from scipy.spatial import distance
from DistanceClassifier import DistanceClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix

class Mahalanobis():
    #Construtor do objeto mahalanobis
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.rede = None

    # Método para atualizar os atributos da classe
    def constructor(self, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
    
    #Método para treinar e testar a rede
    def train(self):
        self.initial = time.time()
        self.rede = DistanceClassifier()
        print("Início do treino")
        self.rede.fit(self.train_X ,self.train_Y)
        self.save()
        print("Início do teste")
        prediction = self.rede.predict(self.test_X)
        self.final = time.time()
        return "Acurracy: " + str(self.rede.score(self.test_Y, prediction))

    #Método para salvar os dados de treino da rede
    def save(self):
        pickle.dump(self.rede, open("treinoMDC.mdc", 'wb'))
        print("Treino salvo com sucesso!")

    #Método para carregar os dados de treino da rede
    def load(self):
        try:
            self.rede = pickle.load(open("treinoMDC.mdc", 'rb'))
            return "Treino carregado com sucesso!"
        except:
            return "Erro. O arquivo de treino não foi encontrado."

    #Método para testar a rede
    def test(self, image):
        prediction = self.rede.predict(image)
        return "Resposta: " +  str(prediction)

    #Método que retorna o tempo gasto no treino e teste da rede
    def getTime(self):
        minutos = int((self.final - self.initial) / 60)
        segundos = int((self.final - self.initial) % 60)
        return "Tempo total de execução: " +  str(minutos) + " min, " + str(segundos) + " segundos."