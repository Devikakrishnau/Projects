import torch
import cv2
import numpy as np


class GradCAMAgent:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, tensor, image_path):

        output = self.model(tensor)
        class_idx = output.argmax()

        self.model.zero_grad()
        output[0, class_idx].backward()

        grads = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze()

        cam = torch.relu(cam).detach().numpy()
        cam = cv2.resize(cam, (224,224))
        cam = cam / cam.max()

        heatmap = cv2.applyColorMap(
            np.uint8(255*cam),
            cv2.COLORMAP_JET
        )

        img = cv2.imread(image_path)
        img = cv2.resize(img,(224,224))

        overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        out_path = "gradcam_output.jpg"
        cv2.imwrite(out_path, overlay)

        return out_path
