import pandas as pd
import numpy as np
import settings
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
# from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
# from sklearn import neighbors
from sklearn.metrics import f1_score


def clean_data(df_clean, df_clean2):
    # Drop columns with greater than 30% of values equal to na
    df_clean = df_clean.replace('na', np.nan)
    df_clean2 = df_clean2.replace('na', np.nan)
    columns = list(df_clean)
    colnum = 0
    for col in columns:
        # if colnum != 0 and colnum != 1:
        #     null_values = df_clean[col].isnull()
        #     null_values2 = df_clean2[col].isnull()
        #     df_clean.loc[~null_values, [col]] = RobustScaler().fit_transform(df_clean.loc[~null_values, [col]])
        #     df_clean2.loc[~null_values2, [col]] = RobustScaler().fit_transform(df_clean2.loc[~null_values2, [col]])
            # temp = df_clean[col].values.reshape(-1, 1)
            # temp = RobustScaler().fit_transform(temp)
            # df_clean[col] = temp
            # temp2 = df_clean2[col].values.reshape(-1, 1)
            # temp2 = RobustScaler().fit_transform(temp2)
            # df_clean2[col] = temp2
            # df_clean[col]=(df_clean[col]-df_clean[col].min())/(df_clean[col].max()-df_clean[col].min())
            # df_clean2[col]=(df_clean2[col]-df_clean2[col].min())/(df_clean2[col].max()-df_clean2[col].min())

        colnum += 1
        #    x_array = np.array(df['total_bedrooms'])
        #    normalized_X = preprocessing.normalize([x_array])
        try:
            if df_clean[col].isnull().sum() > (60001 * .79):
                df_clean = df_clean.drop(columns=[col])
                df_clean2 = df_clean2.drop(columns=[col])
        except:
            pass
    # Count na's in row and drop rows with greater than 30%
    df_done = df_clean.fillna(-1)
    df_done2 = df_clean2.fillna(-1)
    print(df_done)
    print(df_done2)
    return df_done, df_done2


def gen_output(X, Y, output):
    columns = ['id', 'target']

    df = pd.read_csv(settings.test_data_cleaned)
    X_test = df.iloc[:, 1:]  # First one columns are id
    model = alg.fit(X, Y)
    predictions = model.predict(X_test)
    id = list(range(16002))
    id = id[1::]
    csv = pd.DataFrame()
    csv['id'] = id
    csv['target'] = predictions
    csv.to_csv('sample_drop79.csv', index = False)


def write_clean_csv():
    df_train = pd.read_csv(settings.training_data)
    df_test = pd.read_csv(settings.test_data)
    d1, d2 = clean_data(df_train, df_test)
    d1.to_csv('equip_failures_train_clean_drop_79.csv', index = False)
    d2.to_csv('equip_failures_test_clean_drop_79.csv', index = False)

if __name__ == "__main__":
    settings.init()
    #write_clean_csv()

    df = pd.read_csv(settings.training_data_cleaned)
    X = df.iloc[:, 2:]  # First two columns are id and target
    Y = np.array(df.iloc[:, 1])

    # Algorithms
    # knn = neighbors.KNeighborsClassifier(n_neighbors=5)
    alg = RandomForestClassifier(n_estimators=250)
    # sv = svm.SVC(kernel='linear')
    # ada = AdaBoostClassifier(n_estimators=100)
    
    gen_output(X, Y, alg)

    cv = StratifiedKFold(n_splits=10)

    fscores = []
    for train, test in cv.split(X, Y):
        model = alg.fit(X.iloc[train], Y[train])
        Y_pred = model.predict(X.iloc[test])
        fscore = f1_score(Y[test], Y_pred, average='weighted', labels=np.unique(Y[test]))
        fscores.append(fscore)
        print(fscore)

    

    print('Average F-measure:', sum(fscores) / len(fscores))

