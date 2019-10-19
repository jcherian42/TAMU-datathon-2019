import pandas as pd
import numpy as np
import settings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.metrics import f1_score


def clean_data(df_clean):
    # Drop columns with greater than 30% of values equal to na
    columns = list(df_clean)
    for col in columns:
        try:
            if df_clean[col].str.count('na').sum() > (60001 * .30):
                df_clean = df_clean.drop(columns=[col])
        except:
            pass
    # Count na's in row and drop rows with greater than 30%
    df_clean['sum_na'] = df_clean.apply(lambda row: sum(row[0:60000]=='na'), axis=1)
    df_done = df_clean[~(df_clean['sum_na'] >= (172 * .3))]
    df_done = df_done.drop(columns=['sum_na'])
    df_done = df_done.replace('na', -1)

    return df_done


def gen_output(predictions):
    # columns = ['id', 'target']
    pass


if __name__ == "__main__":
    settings.init()

    df = pd.read_csv(settings.training_data_cleaned)
    X = df.iloc[:, 2:]  # First two columns are id and target
    Y = np.array(df.iloc[:, 1])

    knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100)
    alg = svm.SVC(kernel='linear')
    cv = StratifiedKFold(n_splits=10)

    fscores = []
    for train, test in cv.split(X, Y):
        print('Cross Validation')
        model = alg.fit(X.iloc[train], Y[train])
        print(1)
        Y_pred = model.predict(X.iloc[test])
        print(2)
        fscore = f1_score(Y[test], Y_pred, average='macro', labels=np.unique(Y[test]))
        print(3)
        fscores.append(fscore)
        print(fscore)

    print('Average F-measure:', sum(fscores) / len(fscores))
