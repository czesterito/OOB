import numpy as np
from sklearn.ensemble import BaseEnsemble
from sklearn.base import ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class OurOOB(BaseEnsemble, ClassifierMixin):

    # initialize function
    def __init__(self, base_estimator=None, n_estimators=5, time_decay_factor=0.9):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.time_decay_factor = time_decay_factor

    # fit function
    def fit(self, X, y):
        self.partial_fit(X, y)
        return self

    # classes size update function (minority and majority)
    def update_size(self):
        for i, class_label in enumerate(self.y_):
            if class_label == 0:
                self.update_minority_size()
            else:
                self.update_majority_size()
            self.examples_sizes[i] = self.current_training_example

        self.last_instance_sizes = self.current_training_example

    # update minority of classes
    def update_minority_size(self):
        self.current_training_example[0, 0] = (self.current_training_example[0, 0] * self.time_decay_factor) + (
                    1 - self.time_decay_factor)
        self.current_training_example[0, 1] = (self.current_training_example[0, 1] * self.time_decay_factor)

    # uptade majority of classes
    def update_majority_size(self):
        self.current_training_example[0, 1] = (self.current_training_example[0, 1] * self.time_decay_factor) + (
                    1 - self.time_decay_factor)
        self.current_training_example[0, 0] = (self.current_training_example[0, 0] * self.time_decay_factor)

    # improved OOB alghorithm based on article
    def improved_OOB(self):
        self.class_weights = []
        for i, class_label in enumerate(self.y_):
            if (
                    class_label == 1
                    and self.examples_sizes[i][1] < self.examples_sizes[i][0]
            ):
                lambda_poisson = self.examples_sizes[i][0] / \
                                 self.examples_sizes[i][1]
                K = np.asarray(
                    [np.random.poisson(lambda_poisson, 1)[0]
                     for i in range(self.n_estimators)]
                )
            elif (
                    class_label == 0
                    and self.examples_sizes[i][0] < self.examples_sizes[i][1]
            ):
                lambda_poisson = self.examples_sizes[i][1] / \
                                 self.examples_sizes[i][0]
                K = np.asarray(
                    [np.random.poisson(lambda_poisson, 1)[0]
                     for i in range(self.n_estimators)]
                )
            else:
                lambda_poisson = 1
                K = np.asarray(
                    [np.random.poisson(lambda_poisson, 1)[0]
                     for i in range(self.n_estimators)]
                )
            self.class_weights.append(K)

        self.class_weights = np.asarray(self.class_weights).T

    def partial_fit(self, X, y, classes=None):
        # perform data fit
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = [
                clone(self.base_estimator) for i in range(self.n_estimators)
            ]

        # check consistency of examples
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # if class is not given return unique values of class
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # check and update size of instance
        # if last_instance_sizes is not given create new matrix [X_00, X_01] and fill values with zeros
        if not hasattr(self, "last_instance_sizes"):
            self.current_training_example = np.zeros((1, 2))
        else:
            self.current_training_example = self.last_instance_sizes

        # declare matrix of example sizes and fill values with ones
        self.examples_sizes = np.ones((self.X_.shape[0], self.classes_.shape[0]))

        # update size of classes
        self.update_size()

        # improved OOB
        self.improved_OOB()

        # fit data for all weights
        for w, base_model in enumerate(self.ensemble_):
            base_model.partial_fit(
                self.X_, self.y_, self.classes_, sample_weight=self.class_weights[w]
            )
        return self

    # create support matrix for ensemble
    def ensemble_support_matrix(self, X):
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    # make prediction
    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        # Return prediction
        return self.classes_[prediction]
