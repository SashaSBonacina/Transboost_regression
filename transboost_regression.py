import numpy as np
from sklearn.tree import DecisionTreeRegressor

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

"""Second attempt at transboost"""
class TransBoostRegressor2:
  """
  TransBoost Regression Model with Same-Structure Boosting Trees

  Args:
    n_estimators: Number of boosting iterations 
    learning_rate: Learning rate for boosting 
    max_depth: Maximum depth of boosting trees
    lambda_: Regularization parameter for the target domain loss
  """

  def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, lambda_=1.0):
    self.n_estimators = n_estimators
    self.learning_rate = learning_rate
    self.max_depth = max_depth
    self.lambda_ = lambda_
    self.main_gbdt_ = None
    self.ancillary_gbdt_ = None
    self.weights_ = None

  def fit(self, X_target, y_target, X_source, y_source):

    self.main_gbdt_ = GradientBoostingRegressor(n_estimators=1, learning_rate=self.learning_rate, max_depth=self.max_depth)
    self.ancillary_gbdt_ = None
    self.weights_ = np.ones(len(X_source))
    
    subsample_indices = np.random.choice(len(X_target), size=int(0.1 * len(X_target)))
    self.main_gbdt_.fit(X_target.iloc[subsample_indices], y_target.iloc[subsample_indices])

    for _ in range(self.n_estimators):
      # Train main GBDT on target domain with weighted loss
      weighted_loss = lambda y_true, y_pred: self.lambda_ * mse_loss(y_true, y_pred) + 1.0 - self.weights_
      self.main_gbdt_.fit(X_target, y_target)

      # Train ancillary GBDT on source domain with same tree structure
      self.ancillary_gbdt_ = train_ancillary_gbdt(X_source, y_source, self.main_gbdt_.estimators_[0], self.learning_rate)

      # Update weights based on predictions
      self.weights_ = update_weights(X_source, y_source, self.main_gbdt_, self.ancillary_gbdt_)
      print('completed iteration')

  def predict(self, X):
    """
    Predicts target values for new data points.

    Args:
      X: Feature matrix for new data points (N_test x M)

    Returns:
      Predicted target values (N_test,)
    """
    return self.main_gbdt_.predict(X)

  def mse(self, X, y):
    """
    Calculates Mean Squared Error on test data.

    Args:
      X: Feature matrix for test data (N_test x M)
      y: True target values for test data (N_test,)

    Returns:
      Mean Squared Error
    """
    return mse_loss(y, self.predict(X))

# Helper functions
def mse_loss(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

def train_ancillary_gbdt(X_source, y_source, main_tree, learning_rate):
  ancillary_tree = main_tree.tree_.copy()
  ancillary_tree.tree_ = tree_to_leaf_prediction(ancillary_tree, X_source)
  return ancillary_tree

def update_weights(X_source, y_source, main_gbdt, ancillary_gbdt):
  weights = np.zeros(len(X_source))
  for i in range(len(X_source)):
    leaf_id = main_gbdt.apply(X_source[i].reshape(1, -1))[0]
    n_target_instances = np.sum(main_gbdt.tree)


"""First attempt at transboost"""

class TransBoostRegressor:
    def __init__(self, num_iterations=100, learning_rate=0.1, lambda_param=1.0):
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.target_tree = None
        self.source_tree = None

    def fit(self, X_source, y_source, X_target, y_target):
        # Initialize target and source trees
        self.target_tree = DecisionTreeRegressor()
        self.source_tree = DecisionTreeRegressor()

        for iteration in range(self.num_iterations):
            # Update source tree first
            source_residuals = y_source - self.target_tree.predict(X_source)
            self.source_tree.fit(X_source, source_residuals)

            # Update target tree
            target_residuals = y_target - self.source_tree.predict(X_target)
            self.target_tree.fit(X_target, target_residuals, sample_weight=np.ones_like(target_residuals))

            # Update weights (beta)
            target_predictions = self.target_tree.predict(X_source)
            source_predictions = self.source_tree.predict(X_source)
            weights_update = self._calculate_weights_update(y_source, target_predictions, source_predictions)
            beta = weights_update / np.sum(weights_update)

            # Update target tree using weights
            self.target_tree.fit(X_target, target_residuals, sample_weight=beta)


    def predict(self, X):
        # Ensure both trees are fitted before making predictions
        if self.target_tree is None or self.source_tree is None:
            raise ValueError("Both target and source trees must be fitted before making predictions.")

        # Final prediction is the sum of predictions from both trees
        target_predictions = self.target_tree.predict(X)
        source_predictions = self.source_tree.predict(X)
        return target_predictions + self.lambda_param * source_predictions

    def _calculate_weights_update(self, y_true, target_predictions, source_predictions):
        return np.abs(y_true - target_predictions - self.lambda_param * source_predictions)

