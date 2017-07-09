import numpy as np
import pandas as pd
import timeit
from asl_data import AslDb
from my_model_selectors import SelectorBIC
from my_model_selectors import SelectorDIC
from my_model_selectors import SelectorCV
from my_model_selectors import SelectorConstant
from asl_utils import show_errors
from asl_utils import output_stats
from my_recognizer import recognize

asl = AslDb()

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

features_custom_1 = features_ground + features_polar + features_delta
features_custom_2 = features_norm + features_polar + features_delta
features_custom_3 = features_ground + features_norm + features_polar + features_delta

def train_all_words(features, model_selector):
    training = asl.build_training(features)
    sequences = training.get_all_sequences()
    Xlengths = training.get_all_Xlengths()
    model_dict = {}
    for word in training.words:
        model = model_selector(sequences, Xlengths, word, 
                n_constant=3).select()
        model_dict[word]=model
    return model_dict

# Choose a feature set and model selector
model_list = [ SelectorConstant, SelectorBIC, SelectorDIC, SelectorCV ]
features_list = [ features_ground, features_norm, features_polar, features_delta, features_custom_1, features_custom_2, features_custom_3 ]

print("{},{},{},{},{},{}".format("Model", "Features", "Number of Words", "Correct", "Total", "WER"))
for model_selector in model_list:
    for features in features_list:
        models = train_all_words(features, model_selector)
        test_set = asl.build_test(features)
        probabilities, guesses = recognize(models, test_set)
        output_stats(guesses, test_set, model_selector, features)
