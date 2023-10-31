try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata

package_name = "seqexplainer"
__version__ = importlib_metadata.version(package_name)


from ._layer_outs import get_layer_outputs
from .attributions import get_reference, attribute
from .filters import get_activators_max_seqlets, get_activators_n_seqlets, get_pfms, plot_filter_logo, plot_filter_logos, annotate_pfms
from .generative import evolution