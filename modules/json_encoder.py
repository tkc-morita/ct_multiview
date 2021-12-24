# coding: utf-8

import numpy as np
import json

class Numpy2JSONEncoder(json.JSONEncoder):
	def default(self, obj):
		print('default called')
		if isinstance(obj, np.integer) or isinstance(obj, ):
			return int(obj)
		elif isinstance(obj, np.floating):
			return float(obj)
		elif isinstance(obj, np.ndarray):
			return obj.tolist()
		else:
			return super(MyEncoder, self).default(obj)