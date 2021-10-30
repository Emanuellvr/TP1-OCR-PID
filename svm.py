from sklearn import svm
from sklearn import metrics

class SVM():
    def __init__(self, train_X, train_Y, test_X, test_Y, ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.make()

    def make(self):
        print("Make")
        rede = svm.SVC(c=100, kernel="sigmoid")
        print("Criou rede")
        rede.fit(self.train_X, self.train_Y)
        print("Treinou rede")
        prediction = rede.predict(self.test_X)
        print("Acurracy: ", metrics.accuracy_score(self.test_Y, y_pred = prediction))