from .detection_subnetwork import base_models
from .endovis_model import endovis_models

model_dict = {
    'detection_subnetwork' : base_models,
    'endovis_network' : endovis_models,
}
