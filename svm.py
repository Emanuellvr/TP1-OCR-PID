import time
import pickle
from sklearn import svm
from sklearn import metrics

class SVM():
    #Construtor do objeto SVM
    def __init__(self):
        self.train_X = None
        self.train_Y = None
        self.test_X = None
        self.test_Y = None
        self.rede = None

    #Método para atualizar os atributos da classe
    def constructor(self, train_X, train_Y, test_X, test_Y):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y

    #Método para treinar e testar a rede
    def train(self):
        initial = time.time()
        self.rede = svm.SVC(C=0.05, kernel="linear")
        print("Início do treino")
        self.rede.fit(self.train_X, self.train_Y)
        self.save()
        print("Início do teste")
        prediction = self.rede.predict(self.test_X)
        final = time.time()
        print("Acurracy:", metrics.accuracy_score(self.test_Y, y_pred = prediction))

        minutos = int((final - initial) / 60)
        segundos = int((final - initial) % 60)

        print("Tempo total de execução:", minutos, "min, ", segundos, "segundos.")

    #Método para salvar os dados de treino da rede
    def save(self):
        pickle.dump(self.rede, open("treinoSVM.svm", 'wb'))
        print("Treino salvo")

    #Método para carregar os dados de treino da rede
    def load(self):
        self.rede = pickle.load(open("treinoSVM.svm", 'rb'))
        print("Treino carregado")

    #Método para testar a rede
    def test(self, image):
        prediction = self.rede.predict(image)
        print("Resposta", prediction)