import numpy as np
from nb_classifier import NBClassifier
from sklearn import datasets

class sandbox:
    def __init__(self):
        self.X = np.array([['Yes', 'Single', 125],
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

        self.X_cat = np.array([True, True, False])

        self.y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

        self.iris = datasets.load_iris()

        self.X_multiclass = []
    
    def test_priors(self):
        """
        Test that the correct priors are computed
        """
        nb = NBClassifier(False)
        nb.fit(self.X, self.X_cat, self.y)
        print(nb.feature_dists)
        
if __name__ == '__main__':
    s = sandbox()
    s.test_priors()