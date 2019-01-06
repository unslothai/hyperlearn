

class FutureExceedsMemory(BaseException):
	def __init__(self, text = 'Operation done in the future uses more'
								' memory than what is free. HyperLearn'):
		self.text = text
	def __str__(self):
		return self.text


class PartialWrongShape(BaseException):
	def __init__(self, text = 'Partial SVD or Eig needs the same number of'
								' columns in both the previous iteration and'
								' the future iteration. Currenlty, the number'
								' of columns is different.'):
		self.text = text
	def __str__(self):
		return self.text