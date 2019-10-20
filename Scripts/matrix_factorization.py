import pandas as pd
import settings
import numpy as np
<<<<<<< HEAD
from sklearn.decomposition import NMF
=======
>>>>>>> 43c13171e3bf84008c78600d60845fc4eb56a071
import pickle


if __name__ == "__main__":
    settings.init()

    df_training = pd.read_csv(settings.training_data_cleaned)
    df_drop = df_training.drop(columns=['id','target'])
    df_drop = df_drop + 1

    np_drop = df_drop.to_numpy()
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(np_drop)

    with open('new_matrix.pickle', 'wb') as handle:
        pickle.dump(handle, W)
