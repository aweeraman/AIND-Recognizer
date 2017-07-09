import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("inf")
        best_model = None

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            try:
                model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            
                # BIC = -2 * logL + p * logN
                logL = model.score(self.X, self.lengths)
                p = n_component**2 + 2*self.X.shape[1] * n_component - 1
                logN = np.log(len(self.X))
                bic = -2 * logL + p * logN

                if bic < best_score:
                    best_score = bic
                    best_model = model
            except:
                continue

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None

        m = len((self.words).keys())
        
        for n_component in range(self.min_n_components, self.max_n_components + 1):
                antiLogL = 0.0
                count = 0

                try:
                    model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
                    logL = model.score(self.X, self.lengths)

                    for a_word in self.hwords:
                        if a_word == self.this_word:
                            continue
                        X_word, a_word_lengths = self.hwords[a_word]
                        antiLogL += model.score(X_word, a_word_lengths)
                        count += 1

                    antiLogL /= float(count)

                    #DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
                    dic = logL - (1/m-1) * antiLogL

                    if dic > best_score:
                        best_score = dic
                        best_model = model
                except:
                    continue

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float("-inf")
        best_model = None
        n_splits = 3
        split_method = KFold(n_splits)

        for n_component in range(self.min_n_components, self.max_n_components + 1):
            scores = []
            model = None

            if len(self.sequences) < n_splits:
                    break

            for cv_train, cv_test in split_method.split(self.sequences):
                X_train, lengths_train = combine_sequences(cv_train, self.sequences)
                X_test, lengths_test = combine_sequences(cv_test, self.sequences)
                try:
                    model = GaussianHMM(n_components=n_component, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                    scores.append(model.score(X_test, lengths_test))
                except:
                    continue

            if len(scores) > 0:
                avg = np.average(scores)
            else:
                avg = float("-inf")

            if avg > best_score:
                best_score = avg
                best_model = model

        if best_model is None:
            return self.base_model(self.n_constant)

        return best_model
