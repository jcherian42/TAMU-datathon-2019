import pandas as pd
import numpy as np
import settings
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
# from sklearn import neighbors
from sklearn.metrics import f1_score


def clean_data(df_clean, df_clean2):
    # Drop columns with greater than 30% of values equal to na
    columns = list(df_clean)
    for col in columns:
        #if
        #    x_array = np.array(df['total_bedrooms'])
        #    normalized_X = preprocessing.normalize([x_array])
        try:
            if df_clean[col].str.count('na').sum() > (60001 * .30):
                df_clean = df_clean.drop(columns=[col])
                df_clean2 = df_clean2.drop(columns=[col])
        except:
            pass
    # Count na's in row and drop rows with greater than 30%
    df_done = df_clean.replace('na', -1)
    df_done2 = df_clean2.replace('na', -1)

    return df_done, df_done2


def gen_output(predictions):
    # columns = ['id', 'target']
    pass

if __name__ == "__main__":
    settings.init()
    df_train = pd.read_csv(settings.training_data)
    df_test = pd.read_csv(settings.test_data)
    d1, d2 = clean_data(df_train, df_test)
    d1.to_csv('equip_failures_train_clean.csv')
    d2.to_csv('equip_failures_test_clean.csv')

    df = pd.read_csv(settings.training_data_cleaned)
    X = df.iloc[:, 2:]  # First two columns are id and target
    Y = np.array(df.iloc[:, 1])

    # Algorithms
    # knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    alg = RandomForestClassifier(n_estimators=100)
    # sv = svm.SVC(kernel='linear')
    # ada = AdaBoostClassifier(n_estimators=100)
    
    cv = StratifiedKFold(n_splits=10)

    fscores = []
    for train, test in cv.split(X, Y):
        model = alg.fit(X.iloc[train], Y[train])
        Y_pred = model.predict(X.iloc[test])
        fscore = f1_score(Y[test], Y_pred, average='weighted', labels=np.unique(Y[test]))
        fscores.append(fscore)
        print(fscore)

    print('Average F-measure:', sum(fscores) / len(fscores))

