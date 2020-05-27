from mmdet.core import AnchorGenerator
import torch

class MyAnchorGenerator(AnchorGenerator):

    def __init__(self, base_size, ori_size, base_anchors, scale_major=True, ctr=None):
        self.ori_size = list(ori_size)  # list [w, h]
        self.base_anchors_scale = torch.Tensor(base_anchors)
        super(MyAnchorGenerator, self).__init__(base_size, [], [], scale_major=scale_major, ctr=ctr)
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self, factor=[1.0, 1.0]):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        base_anchors_scale = self.base_anchors_scale * torch.Tensor(factor)[None, :]
        base_anchor_permute = base_anchors_scale.permute(1, 0)
        ws = base_anchor_permute[0]
        hs = base_anchor_permute[1]
        # yapf: disable
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable

        return base_anchors

    def grid_anchors(self, featmap_size, stride=16, device='cuda', image_shape = (800, 1333, 3)):
        # import pdb;pdb.set_trace()
        """
        img_h, img_w, _ = image_shape
        factor = [img_w / self.ori_size[0], img_h / self.ori_size[1]]
        base_anchors = self.gen_base_anchors(factor=factor)
        """
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors