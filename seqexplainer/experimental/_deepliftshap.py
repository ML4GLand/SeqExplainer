import warnings
from bpnetlite.attributions import _ProfileLogitScaling
class DeepLiftShap():
	"""A vectorized version of the DeepLIFT/SHAP algorithm from Captum.

	This approach is based on the Captum approach of assigning hooks to
	layers that modify the gradients to implement the rescale rule. This
	implementation is vectorized in a manner that can accept unique references
	for each example to be explained as well as multiple references for each
	example.

	The implementation is minimal and currently only supports the operations
	used in bpnet-lite. This is not meant to be a general-purpose implementation
	of the algorithm and may not work with custom architectures.
	

	Parameters
	----------
	model: bpnetlite.BPNet or bpnetlite.ChromBPNet
		A BPNet or ChromBPNet module as implemented in this repo.

	attribution_func: function or None, optional
		This function is used to aggregate the gradients after calculation.
		Useful when trying to handle the implications of one-hot encodings. If
		None, return the gradients as calculated. Default is None.

	eps: float, optional
		An epsilon with which to threshold gradients to ensure that there
		isn't an explosion. Default is 1e-10.

	warning_threshold: float, optional
		A threshold on the convergence delta that will always raise a warning
		if the delta is larger than it. Normal deltas are in the range of
		1e-6 to 1e-8. Note that convergence deltas are calculated on the
		gradients prior to the attribution_func being applied to them. Default 
		is 0.001. 

	verbose: bool, optional
		Whether to print the convergence delta for each example that is
		explained, regardless of whether it surpasses the warning threshold.
		Note that convergence deltas are calculated on the gradients prior to 
		the attribution_func being applied to them. Default is False.
	"""

	def __init__(self, model, attribution_func=None, eps=1e-6, 
		warning_threshold=0.001, verbose=False):
		for module in model.named_modules():
			if isinstance(module[1], torch.nn.modules.pooling._MaxPoolNd):
				raise ValueError("Cannot use this implementation of " + 
					"DeepLiftShap with max pooling layers. Please use the " +
					"implementation in Captum.")

		self.model = model
		self.attribution_func = attribution_func
		self.eps = eps
		self.warning_threshold = warning_threshold
		self.verbose = verbose

		self.forward_handles = []
		self.backward_handles = []

	def attribute(self, inputs, baselines, target=None, args=None):
		assert inputs.shape[1:] == baselines.shape[2:]
		n_inputs, n_baselines = baselines.shape[:2]

		inputs = inputs.repeat_interleave(n_baselines, dim=0).requires_grad_()
		baselines = baselines.reshape(-1, *baselines.shape[2:]).requires_grad_()

		if args is not None:
			args = (arg.repeat_interleave(n_baselines, dim=0) for arg in args)
		else:
			args = None

		try:
			self.model.apply(self._register_hooks)
			inputs_ = torch.cat([inputs, baselines])

			# Calculate the gradients using the rescale rule
			with torch.autograd.set_grad_enabled(True):
				if args is not None:
					args = (torch.cat([arg, arg]) for arg in 
						args)
					outputs = self.model(inputs_, *args)
				else:
					outputs = self.model(inputs_)
				if target is not None:
					outputs = outputs[:, target]
					assert outputs[0].numel() == 1, (
                        "Target not provided when necessary, cannot"
                        " take gradient with respect to multiple outputs."
                    )
					
				outputs_ = torch.chunk(outputs, 2)[0].sum()
				gradients = torch.autograd.grad(outputs_, inputs)[0]
				
                
			output_diff = torch.sub(*torch.chunk(outputs[:,0], 2))
			input_diff = torch.sum((inputs - baselines) * gradients, dim=(1, 2)) 
			convergence_deltas = abs(output_diff - input_diff)

			if any(convergence_deltas > self.warning_threshold):
				warnings.warn("Convergence deltas too high: " +   
					str(convergence_deltas))

			if self.verbose:
				print(convergence_deltas)

			# Process the gradients to get attributions
			if self.attribution_func is None:
				attributions = gradients
			else:
				attributions = self.attribution_func((gradients,), (inputs,), 
					(baselines,))[0]

		finally:
			for forward_handle in self.forward_handles:
				forward_handle.remove()
			for backward_handle in self.backward_handles:
				backward_handle.remove()

		###

		attr_shape = (n_inputs, n_baselines) + attributions.shape[1:]
		attributions = torch.mean(attributions.view(attr_shape), dim=1, 
			keepdim=False)
		return attributions

	def _forward_pre_hook(self, module, inputs):
		module.input = inputs[0].clone().detach()

	def _forward_hook(self, module, inputs, outputs):
		module.output = outputs.clone().detach()

	def _backward_hook(self, module, grad_input, grad_output):
		delta_in_ = torch.sub(*module.input.chunk(2))
		delta_out_ = torch.sub(*module.output.chunk(2))

		delta_in = torch.cat([delta_in_, delta_in_])
		delta_out = torch.cat([delta_out_, delta_out_])

		delta = delta_out / delta_in

		grad_input = (torch.where(
			abs(delta_in) < self.eps, grad_input[0], grad_output[0] * delta),
		)
		return grad_input

	def _can_register_hook(self, module):
		if len(module._backward_hooks) > 0:
			return False
		if not isinstance(module, (torch.nn.ReLU, _ProfileLogitScaling)):
			return False
		return True

	def _register_hooks(self, module, attribute_to_layer_input=True):
		if not self._can_register_hook(module) or (
			not attribute_to_layer_input and module is self.layer
		):
			return

		# adds forward hook to leaf nodes that are non-linear
		forward_handle = module.register_forward_hook(self._forward_hook)
		pre_forward_handle = module.register_forward_pre_hook(self._forward_pre_hook)
		backward_handle = module.register_full_backward_hook(self._backward_hook)

		self.forward_handles.append(forward_handle)
		self.forward_handles.append(pre_forward_handle)
		self.backward_handles.append(backward_handle)