import odl
import numpy as np

from torch import Tensor 
from odl import uniform_discr
from odl.contrib.torch import OperatorModule
from odl.discr import uniform_partition
try:
	from torch_radon import Radon
except ModuleNotFoundError:
	pass 
from .utils import filter_sinogram


class SimpleTrafo():
	def __init__(self, im_shape, num_angles, impl='odl'): # iradon
		domain = uniform_discr(
				[-im_shape[0]//2, -im_shape[1]//2],
				[im_shape[0]//2, im_shape[1]//2],
				(im_shape[0],im_shape[1]),
				dtype=np.float32
			)

		geometry = odl.tomo.parallel_beam_geometry(
			domain, num_angles=num_angles)
		self._angles = geometry.angles

		if impl == 'odl': 
			ray_trafo_op = odl.tomo.RayTransform(domain, geometry, impl='astra_cuda')
			obs_shape = ray_trafo_op.range.shape
			ray_trafo_op_fun = OperatorModule(ray_trafo_op)
			ray_trafo_adjoint_op_fun = OperatorModule(ray_trafo_op.adjoint)
			fbp_fun = OperatorModule(odl.tomo.fbp_op(ray_trafo_op))

		elif impl == 'iradon':
			ray_trafo_op = Radon(angles=geometry.angles, resolution=im_shape[0], 
				det_count=geometry.detector.shape[0])
			obs_shape = (len(geometry.angles), geometry.detector.shape[0])
			ray_trafo_op_fun = ray_trafo_op.forward
			ray_trafo_adjoint_op_fun = ray_trafo_op.backprojection
			fbp_fun = lambda x: ray_trafo_op.backprojection(filter_sinogram(x))

		else: 
			raise NotImplementedError

		#super().__init__(im_shape=im_shape, obs_shape=obs_shape)

		self.ray_trafo_op_fun = ray_trafo_op_fun
		self.ray_trafo_adjoint_op_fun = ray_trafo_adjoint_op_fun
		self.fbp_fun = fbp_fun

	@property
	def angles(self) -> np.ndarray:
		""":class:`np.ndarray` : The angles (in radian)."""
		return self._angles

	def trafo(self, x: Tensor):
		return self.ray_trafo_op_fun(x)

	def trafo_adjoint(self, x: Tensor):
		return self.ray_trafo_adjoint_op_fun(x)

	def fbp(self, x: Tensor):
		return self.fbp_fun(x)