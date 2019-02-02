
from .multiprocessing import Parallel_Reference
from .base import y_count
from ..base import *
from ..linalg import *


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):

	def __init__(self, reg_param = 0.001, store_covariance = False,
				n_jobs = 1):

		assert type(reg_param) in FloatInt
		assert type(store_covariance) is bool
		assert type(n_jobs) is int

		self.reg_param = reg_param
		self.store_covariance = store_covariance
		self.n_jobs = n_jobs

		self.scalings_, self.rotations_, self.means_ = [], [], []
		self.log_scalings_, self.covariance_, self.scaled_rotations_ = [], [], []


	@n2t
	def fit(self, X, y):

		r_1 = 1-self.reg_param
		Y = row_np(y)

		self.classes_, self.priors_ = y_count(Y)
		self.priors_ /= X.shape[0]

		self.scalings_, self.log_scalings_, self.rotations_, \
		self.means_, self.covariance_, self.scaled_rotations_ = \
		\
			Parallel_Reference(QuadraticDiscriminantAnalysis_partial_fit,
				n_jobs = self.n_jobs, reference = 2)(
				X, Y, self.classes_, self.reg_param, r_1, \
				self.store_covariance)

		if not self.store_covariance: self.covariance_ = []

		self.log_scalings_ = stack(self.log_scalings_)
		return self


	@n2t
	def decision_function(self, X):
		distances = []

		for VS, means in zip(self.scaled_rotations_, self.means_):
			partial_X = (X - means).matmul(   VS   )

			distances.append(  squareSum(partial_X) )
			#distances.append(  (partial_X**2).sum(1)  )

		distances = T( stack(distances) )
		decision = -0.5 * (distances + self.log_scalings_) + self.priors_
		return decision


	def predict_proba(self, X):

		decision = self.decision_function(X)

		likelihood = (decision - T(decision.max(1)[0])).exp()
		sum_softmax = T(  rowSum(likelihood)  )
		#sum_softmax = T(likelihood.sum(1))
		softmax = likelihood / sum_softmax

		return softmax.numpy()


	def predict_log_proba(self, X):

		probas_ = self.predict_proba(X)
		return np_log(probas_)


	def predict(self, X):

		decision = self.decision_function(X).argmax(1)
		y_hat = self.classes_.take(decision)

		return y_hat


def QuadraticDiscriminantAnalysis_partial_fit(
	X, Y, x, reg_param, r_1, store_covariance):

	partial_X = X[toTensor(Y == x)]
	partial_mean = partial_X.mean(0)
	partial_X -= partial_mean

	S, VT = _svd(partial_X, U = False)
	V = T(VT)
	scale = (S**2) / (len(partial_X) -1)

	scale = reg_param + (r_1 * scale)

	partial_cov = None
	if store_covariance:
		partial_cov = (V * scale).matmul(VT)

	scalings_ = scale
	log_scalings_ = scale.log().sum()
	#rotations_ = V
	means_ = partial_mean
	scaled_rotations_ = V / scale**0.5

	return scalings_, log_scalings_, None, means_, partial_cov, scaled_rotations_

