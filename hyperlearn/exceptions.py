

class FutureExceedsMemory(BaseException):
	def __init__(self, text = 'Operation done in the future uses more'
								' memory than what is free. HyperLearn'):
		self.text = text
	def __str__(self):
		return self.text

		