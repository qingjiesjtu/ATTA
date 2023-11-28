from .BaseCAMTensor import BaseCAMTensor, get_2d_projection

# https://arxiv.org/abs/2008.00299


class EigenCAMTensor(BaseCAMTensor):
    def __init__(self, model, target_layers, use_cuda=False,
                 reshape_transform=None):
        super(EigenCAMTensor, self).__init__(model,
                                       target_layers,
                                       use_cuda,
                                       reshape_transform,
                                       uses_gradients=False)

    def get_cam_image(self,
                      input_tensor,
                      target_layer,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth):
        return get_2d_projection(activations)
