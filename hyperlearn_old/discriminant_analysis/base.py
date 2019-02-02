
from collections import Counter
from ..base import toTensor, array

def y_count(y):
	counts = Counter(y)
	classes = array(counts.keys())
	counts = toTensor(counts.values()).type(float32)
	return classes, counts

