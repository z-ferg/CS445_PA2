""" Simple unit tests for text_gen functions.

Author: Kevin Molloy (minor modifications by Nathan Sprague)
Version: Feb 2021
"""

import unittest
import numpy as np
from nb_classifier import NBClassifier
from sklearn import datasets

class Test_NB(unittest.TestCase):
    def setUp(self):
        """
        Data for loan borrower taken from Introduction to Data Mining 2nd ed
        by Tan, Steinback, Karpatne, and Kumar -- Section 4.4

        col 0 is home owner
        col 1 is marital status
        col 2 is annual income

        labels are 0 (did not default) and 1 (did default)

        The other dataset is Fischer's IRIS dataset
        and is used to introduce a 3-class problem.
        """

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

        expected = 0.7
        prior_0 = nb.priors[0]
        err_msg = "prior for class 0 for loan default:"

        np.testing.assert_almost_equal(prior_0, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        expected = 0.3
        prior_1 = nb.priors[1]

        err_msg = "prior for class 1 for loan default:"
        np.testing.assert_almost_equal(prior_1, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)


    def loan_class_work(self, nb):
        """
        Perform testing on the loan default data.
        Utilize nb's smoothing attribute and thus
        supporting testing both smoothed and nonsmoothed versions
        """

        # index 0 is homeowner

        x_pt = 'No'
        class_label = 0

        p = nb.feature_class_prob(feature_index=0, x=x_pt, class_label=class_label)
        err_msg = "Homeowner = " + x_pt + " for class " + str(class_label) \
                  + "  wrong prob"
        if nb.smoothing_flag:
            expected = 0.55555555
            err_msg += " with smoothing"
        else:
            expected = 0.5714285714

        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        # index 1 is martial status
        x_pt = 'Married'
        class_label = 0
        p = nb.feature_class_prob(feature_index=1, x=x_pt, class_label=class_label)
        err_msg = "Martial status = " + x_pt + "for class " \
                  + str(class_label) + " wrong prob"
        if nb.smoothing_flag:
            err_msg += " with smoothing"
            expected = 0.5
        else:
            expected = 0.5714285714

        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        # other label
        class_label = 1
        p = nb.feature_class_prob(feature_index=1, x=x_pt, class_label=class_label)
        err_msg = "Martial status = " + x_pt + "for class " \
                  + str(class_label) + " wrong prob"
        if nb.smoothing_flag:
            err_msg += " with smoothing"
            expected = 0.166666666666667
        else:
            expected = 0.0

        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        # now test a continuous attribute
        x_pt = 120.
        class_label = 0
        p = nb.feature_class_prob(feature_index=2, x=x_pt, class_label=class_label)
        expected = 0.0072
        err_msg = "Income = " + str(x_pt) + "for class " \
                  + str(class_label) + " wrong prob"
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg="Income = 120 class 0 wrong prob",
                                       verbose=True)

    # Loan Classification Tests
    def test_loan_class(self):
        nb = NBClassifier(False)
        nb.fit(self.X, self.X_cat, self.y)
        self.loan_class_work(nb)
        return 200

    def test_loan_class_smoothing(self):
        nb = NBClassifier(True)
        nb.fit(self.X, self.X_cat, self.y)
        self.loan_class_work(nb)

    def test_iris(self):
        nb = NBClassifier(True)
        nb.fit(self.iris.data, np.array([False, False, False, False]),
               self.iris.target)

        sepal_length = 4.9
        p = nb.feature_class_prob(feature_index=0, x=sepal_length, class_label=0)
        err_msg = "sepal length for class 0"
        expected = 1.0817
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        p = nb.feature_class_prob(feature_index=0, x=sepal_length, class_label=1)
        err_msg = "sepal length for class 0"
        expected = 0.1031
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        p = nb.feature_class_prob(feature_index=0, x=sepal_length, class_label=2)
        err_msg = "sepal length for class 0"
        expected = 0.018
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        sepal_width = 3.2
        p = nb.feature_class_prob(feature_index=1, x=sepal_width, class_label=0)
        err_msg = "sepal width for class 0"
        expected = 0.5563720880377253

        err_msg = "sepal width for class 1"
        expected = 0.51995719

        p = nb.feature_class_prob(feature_index=1, x=sepal_width, class_label=1)
        err_msg = "sepal width for class 1"
        expected = 0.9718584132539115

        p = nb.feature_class_prob(feature_index=1, x=sepal_width, class_label=2)
        err_msg = "sepal width for class 2"
        expected = 1.2330295149586672

        p = nb.feature_class_prob(feature_index=2, x=6., class_label=2)

        err_msg = "petal length for class 0"
        expected = 0.51995719
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        petal_length = 6.
        p = nb.feature_class_prob(feature_index=2, x=petal_length, class_label=1)
        err_msg = "petal length for class 0"
        expected = 0.0008945427445283696
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

        p = nb.feature_class_prob(feature_index=2, x=petal_length, class_label=0)
        err_msg = "petal length for class 0"
        expected = 0.0
        np.testing.assert_almost_equal(p, expected, decimal=3,
                                       err_msg=err_msg,
                                       verbose=True)

    def test_predict_iris_single(self):
        nb = NBClassifier(True)
        nb.fit(self.iris.data, np.array([False, False, False, False]),
               self.iris.target)

        X = np.array([[5.9, 2.4, 4.25, 1.2]])  # versicolor

        yhat = nb.predict(X)
        errmsg = 'Predict should return a numpy array, not' + str(type(yhat))
        self.assertTrue(isinstance(yhat, np.ndarray), errmsg)

        y = np.array([1])
        np.testing.assert_allclose(y, yhat)

    def test_predict_iris(self):
        nb = NBClassifier(True)
        nb.fit(self.iris.data, np.array([False, False, False, False]),
               self.iris.target)

        X = np.array([[5.9, 2.4, 4.25, 1.2],  # versicolor
                      [6.3, 3., 5.9, 2.01],  # virginica
                      [5.0, 3.8, 1.7, 0.2],  # setosa
                      [5.9, 3.2, 4.8, 1.8]])  # really is a versicolor (but
        # NB predicts virginica)
        y = np.array([1, 2, 0, 2])
        yhat = nb.predict(X)
        np.testing.assert_allclose(y, yhat)

if __name__ == '__main__':
    unittest.main()
