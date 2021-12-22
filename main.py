import numpy as np # for matrix operations
import pandas as pd # for reading csv data
from sklearn.preprocessing import MinMaxScaler # for feature normalization
from sklearn.model_selection import train_test_split as tts # for splitting dataset
from sklearn.metrics import accuracy_score


class SVM:
    def __init__(self, learning_rate=0.000001, C=10000, epochs=5000):
        self.lr = learning_rate
        self.C = C
        self.epochs = epochs
        self.optimized_theta = None
        self.accuracy = None

    def init(self):
        data = pd.read_csv('data.csv')

        # the SVM only accepts numerical values
        # So, we will transform the categories M & B into values 1 and -1
        diagnosis_map = {'M': 1, 'B': -1}
        data['diagnosis'] = data['diagnosis'].map(diagnosis_map)

        # drop last column (extra column added by pd)
        # and unnecessary first column (id)
        data.drop(data.columns[[-1, 0]], axis=1, inplace=True)
        # return data

        # Putting features & outputs in different DataFrames for convenience
        Y = data.loc[:, 'diagnosis']  # all rows of 'diagnosis'
        X = data.iloc[:, 1:]  # all rows of column 1 and ahead (features)

        # normalize the features using MinMaxScalar from sklearn.preprocessing Normalization is the process of converting
        # a range of values, into a standard range of values, typically in the interval[−1, 1] or [0, 1]
        X_normalized = MinMaxScaler().fit_transform(X.values)
        # pd.Dataframe structures the normalized features in a tabular data structure
        X = pd.DataFrame(X_normalized)

        # Inserting the intercept term 1 in every row. a separate column named intercept will be created
        X.insert(loc=len(X.columns), column='intercept', value=1)

        print('Splitting dataset into Train & Test Sets..')
        X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42) # X_train = (455, 33), X_test = (114, 33), y_train = (455,), y_test = (114,)
        return X_train, X_test, y_train, y_test

    def compute_cost(self, X, y, theta):
        # calculating hinge loss
        N = X.shape[0]
        distances = 1 - y * (np.dot(X, theta))
        distances[distances < 0] = 0
        hinge_loss = self.C * (np.sum(distances) / N)

        # calculating cost
        J = (1/2) * np.dot(theta, theta) + hinge_loss
        # J = (0.0001/2) * np.dot(theta, theta) + hinge_loss
        return J

    def calc_gradient(self, X_i, y_i, theta, C, N):
        if (type(y_i)) == np.float64:
            X_i = np.array([X_i])
            y_i = np.array([y_i])

        # print(X_i.shape, y_i.shape)
        distance = np.array([1 - y_i * (np.dot(X_i, theta))])
        dw = np.zeros(len(theta))
        # print(distance)
        for i, d in enumerate(distance):
            if ((max(0, d)) == 0):
                di = 0
            else:
                # print(y_i.shape, X_i.shape)
                di = theta - (self.C * y_i * X_i)
            dw+=di

        # average
        dw = dw/(N)
        return dw

    def stochastic_gradient_descent(self, X, y):
        self.optimized_theta = np.zeros(X.shape[1])
        for epoch in range(1, self.epochs):
            for i, x_i in enumerate(X):
                grad = self.calc_gradient(x_i, y[i], self.optimized_theta, self.C, len(y))
                self.optimized_theta = self.optimized_theta - (self.lr * grad)

            print(f'Epoch: {epoch} || Cost: {self.compute_cost(X, y, self.optimized_theta)}')

    def calc_accuracy(self, X_test, y_test):
        y_test_predicted = np.array([])
        for i in range(X_test.shape[0]):
            yp = np.sign(np.dot(self.optimized_theta, X_test.to_numpy()[i]))
            y_test_predicted = np.append(y_test_predicted, yp)

        self.accuracy = accuracy_score(y_test.to_numpy(), y_test_predicted)


if __name__ == "__main__":

    svm = SVM()

    X_train, X_test, y_train, y_test = svm.init()

    svm.stochastic_gradient_descent(X_train.to_numpy(), y_train.to_numpy())
    print(f'Optimized Theta: {svm.optimized_theta}')

    svm.calc_accuracy(X_train, y_train)
    print(f'Accuracy on test set: {svm.accuracy}')

