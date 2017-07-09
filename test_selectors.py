import numpy as np
import pandas as pd
import timeit
from asl_data import AslDb
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorCV

asl = AslDb()
asl.df.head() # displays the first five rows of the asl database, indexed by video and frame

words_to_train = ['FISH', 'BOOK', 'VEGETABLE', 'FUTURE', 'JOHN']

df_means = asl.df.groupby('speaker').mean()
df_std = asl.df.groupby('speaker').std()
asl.df['left-x-mean']= asl.df['speaker'].map(df_means['left-x'])

features_ground = ['grnd-rx','grnd-ry','grnd-lx','grnd-ly']
asl.df['grnd-ry'] = asl.df['right-y'] - asl.df['nose-y']
asl.df['grnd-rx'] = asl.df['right-x'] - asl.df['nose-x']
asl.df['grnd-ly'] = asl.df['left-y'] - asl.df['nose-y']
asl.df['grnd-lx'] = asl.df['left-x'] - asl.df['nose-x']

features_norm = ['norm-rx', 'norm-ry', 'norm-lx','norm-ly']
for norm_feature, attribute in zip(features_norm, ['right-x', 'right-y', 'left-x', 'left-y']):
    mean = asl.df['speaker'].map(df_means[attribute])
    std = asl.df['speaker'].map(df_std[attribute])
    asl.df[norm_feature] = (asl.df[attribute] - mean) / std

features_polar = ['polar-rr', 'polar-rtheta', 'polar-lr', 'polar-ltheta']
asl.df['polar-rr'] = np.sqrt(asl.df['grnd-rx']**2 + asl.df['grnd-ry']**2)
asl.df['polar-rtheta'] = np.arctan2(asl.df['grnd-rx'], asl.df['grnd-ry'])
asl.df['polar-lr'] = np.sqrt(asl.df['grnd-lx']**2 + asl.df['grnd-ly']**2)
asl.df['polar-ltheta'] = np.arctan2(asl.df['grnd-lx'], asl.df['grnd-ly'])

features_delta = ['delta-rx', 'delta-ry', 'delta-lx', 'delta-ly']
asl.df['delta-rx'] = asl.df['right-x'].fillna(0).diff().fillna(0)
asl.df['delta-ry'] = asl.df['right-y'].fillna(0).diff().fillna(0)
asl.df['delta-lx'] = asl.df['left-x'].fillna(0).diff().fillna(0)
asl.df['delta-ly'] = asl.df['left-y'].fillna(0).diff().fillna(0)

def selector_BIC():
    print("\n***** BIC *****\n")
    for features in [features_ground, features_norm, features_polar, features_delta]:
        print("\nFeatures: {}".format(features))
        training = asl.build_training(features)
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()
        for word in words_to_train:
            start = timeit.default_timer()
            model = SelectorBIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
            end = timeit.default_timer()-start
            if model is not None:
                print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
            else:
                print("Training failed for {}".format(word))

def selector_DIC():
    print("\n***** DIC *****\n")
    for features in [features_ground, features_norm, features_polar, features_delta]:
        print("\nFeatures: {}".format(features))
        training = asl.build_training(features_ground)
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()
        for word in words_to_train:
            start = timeit.default_timer()
            model = SelectorDIC(sequences, Xlengths, word, 
                    min_n_components=2, max_n_components=15, random_state = 14).select()
            end = timeit.default_timer()-start
            if model is not None:
                print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
            else:
                print("Training failed for {}".format(word))

def selector_CV():
    print("\n***** CV *****\n")
    for features in [features_ground, features_norm, features_polar, features_delta]:
        print("\nFeatures: {}".format(features))
        training = asl.build_training(features_ground)
        sequences = training.get_all_sequences()
        Xlengths = training.get_all_Xlengths()
        for word in words_to_train:
            start = timeit.default_timer()
            model = SelectorCV(sequences, Xlengths, word, 
            min_n_components=2, max_n_components=15, random_state = 14).select()
            end = timeit.default_timer()-start
            if model is not None:
                print("Training complete for {} with {} states with time {} seconds".format(word, model.n_components, end))
            else:
                print("Training failed for {}".format(word))

#selector_BIC()
selector_DIC()
#selector_CV()
