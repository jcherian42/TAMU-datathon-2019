import pandas as pd
import numpy as np
import settings
from sklearn import neighbors
from sklearn.metrics import f1_score


if __name__ == "__main__":
    settings.init()

    df_training = pd.read_csv(settings.training_data)
    X_training = df_training.iloc[:, 2:]  # First two columns are id and target
    Y_training = np.array(df_training.iloc[:, 1])

    alg = neighbors.KNeighborsClassifier(n_neighbors=2)
    alg.fit(X_training, Y_training)

    df_test = pd.read_csv(settings.test_data)
    X_test = df_test.iloc[:, 2:]  # First two columns are id and target
    Y_test = np.array(df_test.iloc[:, 1])

    Y_pred = alg.predict(X_test)

    fscore = f1_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_test))
    print(fscore)


def gen_output(predictions):
    # columns = ['id', 'target']
    pass
