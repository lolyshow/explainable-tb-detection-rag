# explainability.py
import torch
import torch.nn.functional as F


# =========================
# GRAD-CAM
# =========================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, images, input_ids, attention_mask, class_idx=None):
        self.model.zero_grad()

        logits, _, _ = self.model(images, input_ids, attention_mask)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1).squeeze())

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


# =========================
# GRAD-CAM++
# =========================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def compute(self, images, input_ids, attention_mask, class_idx=None):
        self.model.zero_grad()

        logits, _, _ = self.model(images, input_ids, attention_mask)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        logits[0, class_idx].backward()

        grads = self.gradients
        acts = self.activations

        d2 = grads ** 2
        d3 = grads ** 3

        denom = 2 * d2 + acts * d3.sum(dim=(2, 3), keepdim=True)
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))

        alpha = d2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = F.relu((weights * acts).sum(dim=1).squeeze())

        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


# =========================
# SCORE-CAM
# =========================
class ScoreCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def compute(self, images, input_ids, attention_mask, class_idx=None, stride=16):
        img_h, img_w = images.shape[2], images.shape[3]

        with torch.no_grad():
            logits, _, _ = self.model(images, input_ids, attention_mask)

            if class_idx is None:
                class_idx = logits.argmax(dim=1).item()

        acts = self.activations
        n_channels = acts.shape[1]

        score_cam = torch.zeros((img_h, img_w), device=images.device)

        for c in range(0, n_channels, stride):
            act_c = acts[0, c]

            a_min, a_max = act_c.min(), act_c.max()
            if a_max <= a_min:
                continue

            act_norm = (act_c - a_min) / (a_max - a_min)

            act_resized = F.interpolate(
                act_norm.unsqueeze(0).unsqueeze(0),
                size=(img_h, img_w),
                mode="bilinear",
                align_corners=False
            ).squeeze()

            masked_img = images * act_resized.unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                m_logits, _, _ = self.model(masked_img, input_ids, attention_mask)
                m_conf = torch.softmax(m_logits, dim=1)[0, class_idx].item()

            score_cam += m_conf * act_resized

        score_cam = torch.relu(score_cam)

        s_min, s_max = score_cam.min(), score_cam.max()
        if s_max > s_min:
            score_cam = (score_cam - s_min) / (s_max - s_min)

        return score_cam.cpu().numpy(), class_idx