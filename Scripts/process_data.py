import pandas as pd
import numpy as np
import settings
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
    df_clean['sum_na'] = df_clean.apply(lambda row: sum(row[0:60000]=='na') ,axis=1)
    df_done = df_clean[~(df_clean['sum_na'] >= (172*.3))]
    df_done = df_done.drop(columns=['sum_na'])
    df_done = df_done.replace('na', -1)
    
    return df_done

def gen_output(predictions):
    # columns = ['id', 'target']
    pass

if __name__ == "__main__":
    settings.init()

    df_training = pd.read_csv(settings.training_data)

    X_training = cleaned_data.iloc[:, 2:]  # First two columns are id and target
    Y_training = np.array(cleaned_data.iloc[:, 1])

    alg = neighbors.KNeighborsClassifier(n_neighbors=2)
    alg.fit(X_training, Y_training)

    df_test = pd.read_csv(settings.test_data)

    X_test = df_test.iloc[:, 2:]  # First two columns are id and target
    Y_test = np.array(df_test.iloc[:, 1])

    Y_pred = alg.predict(X_test)

    fscore = f1_score(Y_test, Y_pred, average='macro', labels=np.unique(Y_test))
    print(fscore)
