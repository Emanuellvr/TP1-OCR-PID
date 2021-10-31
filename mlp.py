from sklearn.neural_network import MLPClassifier
from sklearn import metrics

class MLP():
    def __init__(self, train_X, train_Y, test_X, test_Y, ):
        self.train_X = train_X
        self.train_Y = train_Y
        self.test_X = test_X
        self.test_Y = test_Y
        self.make()

    def make(self):
        print("Make")
        rede = MLPClassifier(random_state=1, max_iter=300).fit(self.train_X, self.train_Y)
        print("Criou rede")
        rede.predict_proba(self.test_X[:1])
        print("Treinou rede")
        prediction = rede.predict(self.test_X[:5, :])
        # clf.score(X_test, y_test)
        print("Acurracy: ", metrics.accuracy_score(self.test_Y, y_pred = prediction))