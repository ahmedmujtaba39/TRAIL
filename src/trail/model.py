"""Small pose-space transition model with maskable phonological tokens."""

from __future__ import annotations

import torch
from torch import nn


class TransitionTransformer(nn.Module):
    def __init__(self, joints: int, descriptor_dim: int, width: int = 256, heads: int = 8, layers: int = 4):
        super().__init__()
        self.joints = joints
        self.pose_projection = nn.Linear(joints * 3, width)
        self.descriptor_projection = nn.Linear(descriptor_dim, width)
        self.duration_projection = nn.Linear(1, width)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.query = nn.Parameter(torch.randn(1, 64, width) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(width, heads, 4 * width, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.output = nn.Linear(width, joints * 3)

    def forward(
        self,
        left_pose: torch.Tensor,
        right_pose: torch.Tensor,
        left_descriptor: torch.Tensor,
        right_descriptor: torch.Tensor,
        duration: torch.Tensor,
        *,
        use_phonology: bool,
    ) -> torch.Tensor:
        """Generate a residual over straight-line interpolation.

        Pose tensors have shape `[batch, boundary_frames, joints, 3]` and
        descriptors `[batch, descriptor_dim]`. Output has one frame per requested
        duration; variable-length batches should be grouped by duration.
        """
        batch, boundary_frames, joints, channels = left_pose.shape
        if joints != self.joints or channels != 3:
            raise ValueError("Unexpected pose shape.")
        if duration.ndim != 1 or torch.unique(duration).numel() != 1:
            raise ValueError("Batch items must share one transition duration.")
        m = int(duration[0].item())
        if m > self.query.shape[1]:
            raise ValueError("Requested duration exceeds configured maximum of 64 frames.")
        pose_tokens = torch.cat([left_pose, right_pose], dim=1).reshape(batch, 2 * boundary_frames, joints * 3)
        pose_tokens = self.pose_projection(pose_tokens)
        if use_phonology:
            desc_tokens = torch.stack([self.descriptor_projection(left_descriptor), self.descriptor_projection(right_descriptor)], dim=1)
        else:
            desc_tokens = self.mask_token.expand(batch, 2, -1)
        duration_token = self.duration_projection(duration.float().view(batch, 1, 1)).squeeze(1).unsqueeze(1)
        memory = self.encoder(torch.cat([pose_tokens, desc_tokens, duration_token], dim=1))
        query = self.query[:, :m].expand(batch, -1, -1)
        residual = self.output(self.decoder(query, memory)).view(batch, m, joints, 3)
        start, end = left_pose[:, -1:], right_pose[:, :1]
        steps = torch.linspace(0, 1, m + 2, device=left_pose.device)[1:-1].view(1, m, 1, 1)
        reference = start * (1 - steps) + end * steps
        return reference + residual


def interpolate(left_pose: torch.Tensor, right_pose: torch.Tensor, duration: int) -> torch.Tensor:
    start, end = left_pose[:, -1:], right_pose[:, :1]
    steps = torch.linspace(0, 1, duration + 2, device=left_pose.device)[1:-1].view(1, duration, 1, 1)
    return start * (1 - steps) + end * steps
