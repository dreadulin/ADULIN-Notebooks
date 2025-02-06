import numpy as np


def compute_RMSE(y_true, y_pred):
    """Computes the Root Mean Squared Error (RMSE) given the ground truth
    values and the predicted values.

    Arguments:
        y_true {np.ndarray} -- A numpy array of shape (N, 1) containing
        the ground truth values.
        y_pred {np.ndarray} -- A numpy array of shape (N, 1) containing
        the predicted values.

    Returns:
        float -- Root Mean Squared Error (RMSE)
    """

    # TODO: Compute the Root Mean Squared Error
    squared_difference = (y_true - y_pred)**2
    mean_squared = np.mean(squared_difference)
    rmse = np.sqrt(mean_squared)

    return rmse


class AnalyticalMethod(object):

    def __init__(self):
        """Class constructor for AnalyticalMethod
        """
        self.W = None

    def feature_transform(self, X):
        """Appends a vector of ones for the bias term.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) consisting of N
            samples each of dimension D.

        Returns:
            np.ndarray -- A numpy array of shape (N, D + 1)
        """

        # TODO: Append a vector of ones across the dimension of your input
        # data. This accounts for the bias or the constant in your
        # hypothesis function.
        ones_column = np.ones((X.shape[0],1))
        f_transform = np.column_stack((ones_column,X))
        return f_transform

    def compute_weights(self, X, y):
        """Compute the weights based on the analytical solution.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N, D) containing the
            training data; there are N training samples each of dimension D.
            y {np.ndarray} -- A numpy array of shape (N, 1) containing the
            ground truth values.

        Returns:
            np.ndarray -- weight vector; has shape (D, 1) for dimension D
        """
        # TODO: Call the feature_transform() method
        X = self.feature_transform(X)

        # TODO: Calculate for the weights using the closed form.
        # Hint: Use np.linalg.pinv.
        self.W = np.linalg.pinv(X).dot(y)

        return self.W

    def predict(self, X):
        """Predict values for test data using analytical solution.

        Arguments:
            X {np.ndarray} -- A numpy array of shape (num_test, D) containing
            test data consisting of num_test samples each of dimension D.

        Returns:
            np.ndarray -- A numpy array of shape (num_test, 1) containing
            predicted values for the test data, where y[i] is the predicted
            value for the test point X[i].
        """

        # TODO: Since you transformed your training data to include the bias
        # y-intercept, also transform the features for the test to match.
        transformed = np.c_[np.ones(X.shape[0]), X]

        # TODO: Compute for the predictions of the model on new data using the
        # learned weight vectors.
        prediction = transformed.dot(self.W)

        return prediction


class PolyFitMethod(object):

    def __init__(self):
        """Class constructor for PolyFitMethod
        """
        self.W = None

    def compute_weights(self, X, y):
        """Compute the weights using np.polyfit().

        Arguments:
            X {np.ndarray} -- A numpy array of shape (N,) containing the
            training data; there are N training samples
            y {np.ndarray} -- A numpy array of shape (N,) containing the
            ground truth values.

        Returns:
            np.ndarray -- weight vector; has shape (D,)
        """

        # TODO: Calculate for the weights using np.polyfit()
        X = X.flatten()
        y = y.flatten()
        self.W = np.polyfit(X,y,1)

        return self.W

    def predict(self, X):
        """Predict values for test data using np.poly1d().

        Arguments:
            X {np.ndarray} -- A numpy array of shape (num_test, ) containing
            test data consisting of num_test samples.

        Returns:
            np.ndarray -- A numpy array of shape (num_test, 1) containing
            predicted values for the test data, where y[i] is the predicted
            value for the test point X[i].
        """

        # TODO: Compute for the predictions of the model on new data using the
        # learned weight vectors.
        # Hint: Use np.poly1d().
        X = X.flatten()
        poly_model = np.poly1d(self.W)
        prediction = poly_model(X)
        prediction = prediction.reshape(-1,1)

        return prediction
