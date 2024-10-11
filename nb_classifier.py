"""Pure Python Naive Bayes classifier

Simple nb_classifier.

Initial Author: Kevin Molloy and ???
Minor modifications by: Nathan Sprague
"""

import numpy as np
from scipy.stats import norm
import math


class NBClassifier:
    """A naive bayes classifier for use with categorical and real-valued attributes/features.

    Public Attributes:
        classes (list): The set of integer classes this tree can classify.
        smoothing_flag (boolean): Indicator whether or not to perform
                                  Laplace smoothing
        feature_dists (list):  A placeholder for each feature/column in X
                               that holds the class-conditional distributions
                               for that feature. The distributions are
                               represented as dictionaries that map from
                               class labels to:
                                -- for continuous features, a tuple with
                                the distribution parameters for a Gaussian
                                (mean, variance)
                                -- for discrete features, another dictionary
                                where the keys are the individual domain
                                values for the feature,and the value is the
                                computed probability from the training data
        priors (dictionary): A dictionary mapping from class labels to
                             probabilities.

    """

    def __init__(self, smoothing_flag=False):
        """
        NBClassifier constructor.

        :param smoothing_flag: for discrete attributes only
        """
        self.smoothing_flag = smoothing_flag
        self.feature_dists = []
        self.priors = dict()
        self.classes = []

    def fit(self, X, X_categorical, y):
        """
        Construct the NB using the provided data and labels.

        :param X: Numpy array with shape (num_samples, num_features).
                  This is the training data.
        :param X_categorical: numpy boolean array with length num_features.
                              True values indicate that the feature is discrete.
                              False values indicate that the feature is continuous.
        :param y: Numpy integer array with length num_samples
                  These are the training labels.

        :return: Stores results in class variables, nothing returned.

        An example of how my dictionary looked after running fit on the
        loan classification problem in the textbook without smoothing:
        [{0: {'No': 0.5714285714285714, 'Yes': 0.42857142857142855},
         1: {'No': 1.0}   },
        {0: {'Divorced': 0.14285714285714285, 'Married': 0.5714285714285714, 'Single': 0.2857142857142857},
         1: {'Divorced': 0.3333333333333333, 'Single': 0.6666666666666666}   },
        {0: (110.0, 54.543560573178574),
         1: (90.0, 5.0)}]
        """

        # Need a category for each column in X
        assert (X.shape[1] == X_categorical.shape[0])

        # each row in training data needs a label
        assert (X.shape[0] == y.shape[0])
        
        # SET THE CLASSES
        for label in y:
            if label not in self.classes:
                self.classes.append(label)
        
        # SET THE X CATEGORICAL
        self.X_categorical = X_categorical
        
        # SET THE PRIORS
        for label in y: # Set the count of each label in the priors class item
            self.priors[label] = self.priors[label] + 1 if label in self.priors.keys() else 1
        for label in self.priors: # Turn counts into probabilities
            self.priors[label] = self.priors[label] / len(y)
        
        for feature in range(len(X_categorical)): # Iterate over the features in
            temp = dict() # Creates dictionary for each feature

            for row in range(X.shape[0]): # Iterate over each item and set the counts
                sample = X[row]
    
                if X_categorical[feature]:  # Discrete
                    if y[row] not in temp.keys():
                        temp[y[row]] = dict([(sample[feature], 1)])
                    else:
                        # Increment the count for the specific sample value for the current label
                        if sample[feature] in temp[y[row]]:
                            temp[y[row]][sample[feature]] += 1
                        else:
                            temp[y[row]][sample[feature]] = 1
                else: #Continuous
                    if y[row] not in temp:
                        temp[y[row]] = []
                    temp[y[row]].append(float(sample[feature]))
            
            if X_categorical[feature]: # Discrete
                for label in temp:
                    if self.smoothing_flag:
                        all_vals = set()
                        for row in range(len(X)):
                            all_vals.add(X[row][feature])
                        
                        total = sum(temp[label].values()) + len(all_vals)
                        for val in all_vals:
                            if val not in temp[label]:
                                temp[label][val] = 1 / total
                            else:
                                temp[label][val] = ((temp[label][val] + 1) / total)
                    else:
                        total = sum(temp[label].values())
                        temp[label] = {val:((count)/ total) for val, count in temp[label].items()}
            else: # Continuous
                for label in temp:
                    if temp[label]:
                        mean = np.mean(temp[label])
                        variance = np.var(temp[label], ddof=1)
                        if variance == 0:
                            variance = 1e-9
                        temp[label] = (mean, variance)
                    else:
                        temp[label] = (None, None)
            
            self.feature_dists.append(temp)


    def feature_class_prob(self, feature_index, x, class_label):
        """
        Compute a single class-conditional probability.  You can call
        this function in your predict function if you wish.

        Example: For the loan default problem:
            feature_class_prob(1, 'Single', 0) returns 0.2857

        :param feature_index:  index into the feature set (column of X)
        :param x: the data value
        :param class_label: the label used in the probability (see return below)

        :return: p(x_{feature_index} | y=class_label)
        """

        feature_dist = self.feature_dists[feature_index]

        # validate feature_index
        assert feature_index < self.X_categorical.shape[0], \
            'Invalid feature index passed to feature_class_prob'

        # validate class_label
        assert class_label < len(self.classes), \
            'invalid class label passed to feature_class_prob'
            
        if(self.X_categorical[feature_index]):  # Discrete
            return feature_dist[class_label].get(x, 0)
        else:
            mean, variance = feature_dist[class_label]
            
            if variance is not None and variance > 0:  
                return norm.pdf(x, loc=mean, scale=np.sqrt(variance))
            else:
                return 1e-9

    def predict(self, X):
        """
        Predict labels for test matrix X

        Parameters/returns
        ----------
        :param X:  Numpy array with shape (num_samples, num_features)
        :return: Numpy array with shape (num_samples, )
            Predicted labels for each entry/row in X.
        """

        # validate that x contains exactly the number of features
        assert (X.shape[1] == self.X_categorical.shape[0])

        return_arr = []
        
        for row in X:
            label_probs = []
            
            for label in self.classes:
                prob = math.log(self.priors[label])
                
                for item in range(len(row)):
                    value = row[item]
                    
                    if not self.X_categorical[item]:
                        value = float(value)

                    temp = self.feature_class_prob(item, value, label)
                    if temp > 0:
                        prob += math.log(temp)
                    else:
                        prob += float('-inf')
                label_probs.append(prob)
            return_arr.append(self.classes[label_probs.index(max(label_probs))])
        return np.array(return_arr)

def nb_demo():
    # data from table Figure 4.8 in the textbook

    X = np.array([['Yes', 'Single', 125],
                  ['No', 'Married', 100],
                  ['No', 'Single', 70],
                  ['Yes', 'Married', 120],
                  ['No', 'Divorced', 95],
                  ['No', 'Married', 60],
                  ['Yes', 'Divorced', 220],
                  ['No', 'Single', 85],
                  ['No', 'Married', 75],
                  ['No', 'Single', 90]
                  ])

    # first two features are categorical and 3rd is continuous
    X_categorical = np.array([True, True, False])

    # class labels (default borrower)
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

    nb = NBClassifier(smoothing_flag=True)

    nb.fit(X, X_categorical, y)
    test_pts = np.array([['No', 'Married', 120],
                         ['No', 'Divorced', 95]])
    yhat = nb.predict(test_pts)
    print(nb.feature_class_prob(feature_index=0, x='No', class_label=0))

    # the book computes this as 0.0016 * alpha
    print('Predicted class for someone who is not a homeowner,')
    print('is married, and earns 120K a year is:', yhat[0])

    print('Predicted class for someone who is not a homeowner,')
    print('is divorced, and earns 95K a year is:', yhat[1])


def main():
    nb_demo()


if __name__ == "__main__":
    main()
