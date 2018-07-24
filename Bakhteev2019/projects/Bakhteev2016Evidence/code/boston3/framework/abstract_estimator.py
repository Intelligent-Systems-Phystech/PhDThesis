class AbstractEstimator:
	def __init__(self,  param_length, param_scale,  X, Y, builder, **kwargs):
		"""
		param_length -int,
		param_scale   - list
		builder: (params, X,Y)  -> neg log prob
		"""
		pass
	def update(self, X,Y):
		pass
	
	def evidence(self,X,Y):
		return 0.0

