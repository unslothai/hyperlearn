
from sklearn.externals import six
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator, TransformerMixin
from ..numba import mean
from ..utils import _float


class _basePCA(six.with_metaclass(ABCMeta, BaseEstimator, TransformerMixin)):
    """
    Base PCA class. Do not use this for applications.
    This is just a class wrapper for the other classes.

    Implements TRANSFORM, INVERSE_TRANSFORM, FIT
    """
    def transform(self, X):
        if hasattr(self, 'singular_values_'):
            if self.centre: X -= self.mean_
                
            X_transformed = X @ self.components_.T
            
            if self.centre: X += self.mean_
            return X_transformed
        else:
            print('PCA has not been fit. Call .fit(X)')
            

    def inverse_transform(self, X_transformed):
        if hasattr(self, 'singular_values_'):
            X = X_transformed @ self.components_
            if self.centre: X += self.mean_
            return X
        else:
            print('PCA has not been fit. Call .fit(X)')
            
            
    def _store_components(self, S2, VT):
        explained_variance_ = S2 / (self.n - 1)
        total_var = explained_variance_.sum()

        k = self.n_components
        self.components_ = VT[:k]
        self.explained_variance_ = explained_variance_[:k]
        self.explained_variance_ratio_ = explained_variance_[:k] / total_var
        self.singular_values_ = S2[:k]**0.5
        
        
    def fit(self, X):
        self.n, self.p = X.shape
        if self.centre: 
            self.mean_ = mean(X)
            X -= self.mean_  # Inplace mean removal
            
        S2, VT = self.decomp(X)
        self._store_components(S2, VT)
        
        if self.centre: X += self.mean_  # Recover old X
        return self
