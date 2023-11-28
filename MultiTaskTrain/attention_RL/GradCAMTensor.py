from .BaseCAMTensor import BaseCAMTensor
import torch

class GradCAMTensor(BaseCAMTensor):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(
            GradCAMTensor,
            self).__init__(
            model,
            target_layers,
            use_cuda,
            reshape_transform)

    def get_cam_weights(self,
                        input_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        return torch.mean(grads, axis=(2, 3))
