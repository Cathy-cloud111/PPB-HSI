import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.houston_cls import (
    HOUSTON2013_CLASS_NAMES,
    HoustonPatchClassification,
    confusion_to_metrics,
    fuse_hsi_lidar,
    load_houston_hsi,
    load_houston_lidar,
    parse_class_sessions,
)


class HoustonPatchCNN(nn.Module):
    def __init__(self, in_channels, num_classes=15, dropout=0.2):
        super().__init__()
        self.features = HoustonFeatureExtractor(in_channels)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def extract_features(self, x):
        return self.features(x)

    def forward(self, x):
        return self.classifier(self.extract_features(x))


class HoustonFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return torch.flatten(self.net(x), 1)


class SpectralPromptBranch(nn.Module):
    """Lightweight HSI spectral prompt applied before the spatial feature extractor."""

    def __init__(self, in_channels, mode="band_gate"):
        super().__init__()
        self.in_channels = int(in_channels)
        self.mode = str(mode)
        if self.mode == "fft_gate":
            num_freq = self.in_channels // 2 + 1
            self.prompt_logits = nn.Parameter(torch.zeros(1, num_freq, 1, 1))
        else:
            self.prompt_logits = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))

    def forward(self, x):
        if self.mode == "fft_gate":
            original_dtype = x.dtype
            spectrum = torch.fft.rfft(x.float(), n=self.in_channels, dim=1)
            gate = torch.tanh(self.prompt_logits).to(device=x.device, dtype=spectrum.real.dtype)
            residual = torch.fft.irfft(spectrum * gate, n=self.in_channels, dim=1)
            return residual.to(dtype=original_dtype)

        gate = torch.tanh(self.prompt_logits).to(device=x.device, dtype=x.dtype)
        return x * gate


def orthogonality_penalty(matrix):
    if matrix.size(0) <= 1:
        return matrix.new_tensor(0.0)
    normalized = F.normalize(matrix, dim=1)
    gram = normalized @ normalized.t()
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    return (gram - eye).pow(2).mean()


def spectral_descriptor_from_images(images, mode="mean_grad"):
    signature = images.float().mean(dim=(2, 3))
    if mode == "mean":
        return signature
    if signature.size(1) <= 1:
        return signature
    gradient = signature[:, 1:] - signature[:, :-1]
    return torch.cat([signature, gradient], dim=1)


def spectral_descriptor_from_coords(hsi_array, coords, patch_size, mode="mean_grad"):
    half = int(patch_size) // 2
    padded = np.pad(hsi_array, ((half, half), (half, half), (0, 0)), mode="edge")
    signatures = []
    for y, x in coords:
        patch = padded[int(y):int(y) + int(patch_size), int(x):int(x) + int(patch_size), :]
        signatures.append(patch.mean(axis=(0, 1)))
    signature = np.asarray(signatures, dtype=np.float32)
    if mode == "mean" or signature.shape[1] <= 1:
        return signature
    gradient = signature[:, 1:] - signature[:, :-1]
    return np.concatenate([signature, gradient], axis=1)


class DualPromptPool(nn.Module):
    """PDP-style shared/private prompt pool for patch classification features."""

    def __init__(
        self,
        feature_dim,
        num_classes=15,
        shared_size=4,
        prompt_scale=0.2,
        temperature=0.2,
        ortho_weight=0.01,
        tri_pool_similarity="max",
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.shared_size = int(shared_size)
        self.prompt_scale = float(prompt_scale)
        self.temperature = float(temperature)
        self.ortho_weight = float(ortho_weight)
        self.tri_pool_similarity = str(tri_pool_similarity)

        self.shared_keys = nn.Parameter(torch.empty(self.shared_size, feature_dim))
        self.shared_prompts = nn.Parameter(torch.empty(self.shared_size, feature_dim))
        self.private_keys = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        self.private_prompts = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        self.old_private_keys = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        self.old_private_prompts = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        self.new_private_keys = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        self.new_private_prompts = nn.Parameter(torch.empty(self.num_classes, feature_dim))
        self.norm = nn.LayerNorm(feature_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.shared_keys)
        nn.init.normal_(self.shared_prompts, std=0.02)
        nn.init.orthogonal_(self.private_keys)
        nn.init.normal_(self.private_prompts, std=0.02)

        rng_state = torch.get_rng_state()
        nn.init.orthogonal_(self.old_private_keys)
        nn.init.normal_(self.old_private_prompts, std=0.02)
        nn.init.orthogonal_(self.new_private_keys)
        nn.init.normal_(self.new_private_prompts, std=0.02)
        torch.set_rng_state(rng_state)

    def forward(self, features, active_class_ids):
        active_indices = torch.tensor(
            [int(class_id) - 1 for class_id in active_class_ids],
            device=features.device,
            dtype=torch.long,
        )
        if active_indices.numel() == 0:
            return self.norm(features), features.new_tensor(0.0)

        return self.forward_joint(features, active_indices)

    def attend(self, features, keys, prompts):
        query = F.normalize(features, dim=1)
        norm_keys = F.normalize(keys, dim=1)
        attention = torch.softmax((query @ norm_keys.t()) / max(self.temperature, 1e-6), dim=1)
        return attention @ prompts

    def group_similarity(self, features, keys):
        if keys.numel() == 0:
            return features.new_full((features.size(0), 1), -1e4)
        query = F.normalize(features, dim=1)
        norm_keys = F.normalize(keys, dim=1)
        similarities = query @ norm_keys.t()
        if self.tri_pool_similarity == "mean":
            return similarities.mean(dim=1, keepdim=True)
        if self.tri_pool_similarity == "logsumexp":
            return torch.logsumexp(similarities, dim=1, keepdim=True)
        return similarities.max(dim=1, keepdim=True).values

    def ids_to_indices(self, class_ids, device):
        class_ids = class_ids or []
        return torch.tensor(
            [int(class_id) - 1 for class_id in class_ids],
            device=device,
            dtype=torch.long,
        )

    def forward_joint(self, features, active_indices):
        prompt, prompt_loss = self.build_joint_prompt(features, active_indices)
        prompted_features = self.norm(features + self.prompt_scale * prompt)
        return prompted_features, prompt_loss

    def build_joint_prompt(self, features, active_indices):
        keys = torch.cat([self.shared_keys, self.private_keys.index_select(0, active_indices)], dim=0)
        prompts = torch.cat([self.shared_prompts, self.private_prompts.index_select(0, active_indices)], dim=0)
        prompt = self.attend(features, keys, prompts)

        prompt_loss = features.new_tensor(0.0)
        if self.ortho_weight > 0:
            prompt_loss = prompt_loss + self.ortho_weight * orthogonality_penalty(keys)
            prompt_loss = prompt_loss + self.ortho_weight * orthogonality_penalty(prompts)
        return prompt, prompt_loss

    def build_prompt_loss(self, private_keys, private_prompts, features):
        prompt_loss = features.new_tensor(0.0)
        if self.ortho_weight > 0:
            keys = torch.cat([self.shared_keys, private_keys], dim=0)
            prompts = torch.cat([self.shared_prompts, private_prompts], dim=0)
            prompt_loss = prompt_loss + self.ortho_weight * orthogonality_penalty(keys)
            prompt_loss = prompt_loss + self.ortho_weight * orthogonality_penalty(prompts)
        return prompt_loss

    def forward_gated(self, features, active_class_ids, gate):
        active_indices = torch.tensor(
            [int(class_id) - 1 for class_id in active_class_ids],
            device=features.device,
            dtype=torch.long,
        )
        if active_indices.numel() == 0:
            return self.norm(features), features.new_tensor(0.0)

        private_keys = self.private_keys.index_select(0, active_indices)
        private_prompts = self.private_prompts.index_select(0, active_indices)
        shared_prompt = self.attend(features, self.shared_keys, self.shared_prompts)
        private_prompt = self.attend(features, private_keys, private_prompts)
        gate = gate.clamp(0.0, 1.0)
        prompt = (1.0 - gate) * shared_prompt + gate * private_prompt
        prompted_features = self.norm(features + self.prompt_scale * prompt)
        prompt_loss = self.build_prompt_loss(private_keys, private_prompts, features)
        return prompted_features, prompt_loss

    def forward_tri_pool(self, features, old_class_ids, new_class_ids):
        prompt, prompt_loss = self.build_tri_prompt(features, old_class_ids, new_class_ids)
        prompted_features = self.norm(features + self.prompt_scale * prompt)
        return prompted_features, prompt_loss

    def build_tri_prompt(
        self,
        features,
        old_class_ids,
        new_class_ids,
        include_shared=True,
        old_score=None,
        new_score=None,
    ):
        old_indices = self.ids_to_indices(old_class_ids, features.device)
        new_indices = self.ids_to_indices(new_class_ids, features.device)

        prompt_parts = []
        group_scores = []
        regularized_keys = []
        regularized_prompts = []

        if include_shared:
            shared_prompt = self.attend(features, self.shared_keys, self.shared_prompts)
            shared_score = self.group_similarity(features, self.shared_keys)
            prompt_parts.append(shared_prompt)
            group_scores.append(shared_score)
            regularized_keys.append(self.shared_keys)
            regularized_prompts.append(self.shared_prompts)

        if old_indices.numel() > 0:
            old_keys = self.old_private_keys.index_select(0, old_indices)
            old_prompts = self.old_private_prompts.index_select(0, old_indices)
            prompt_parts.append(self.attend(features, old_keys, old_prompts))
            group_scores.append(old_score if old_score is not None else self.group_similarity(features, old_keys))
            regularized_keys.append(old_keys)
            regularized_prompts.append(old_prompts)

        if new_indices.numel() > 0:
            new_keys = self.new_private_keys.index_select(0, new_indices)
            new_prompts = self.new_private_prompts.index_select(0, new_indices)
            prompt_parts.append(self.attend(features, new_keys, new_prompts))
            group_scores.append(new_score if new_score is not None else self.group_similarity(features, new_keys))
            regularized_keys.append(new_keys)
            regularized_prompts.append(new_prompts)

        if not prompt_parts:
            return torch.zeros_like(features), features.new_tensor(0.0)

        group_weights = torch.softmax(
            torch.cat(group_scores, dim=1) / max(self.temperature, 1e-6),
            dim=1,
        )
        stacked_prompts = torch.stack(prompt_parts, dim=1)
        prompt = (group_weights.unsqueeze(-1) * stacked_prompts).sum(dim=1)
        prompted_features = self.norm(features + self.prompt_scale * prompt)

        prompt_loss = features.new_tensor(0.0)
        if self.ortho_weight > 0 and regularized_keys:
            keys = torch.cat(regularized_keys, dim=0)
            prompts = torch.cat(regularized_prompts, dim=0)
            prompt_loss = prompt_loss + self.ortho_weight * orthogonality_penalty(keys)
            prompt_loss = prompt_loss + self.ortho_weight * orthogonality_penalty(prompts)
        return prompt, prompt_loss

    def forward_joint_tri_pool(
        self,
        features,
        active_class_ids,
        old_class_ids,
        new_class_ids,
        alpha,
        old_score=None,
        new_score=None,
    ):
        active_indices = self.ids_to_indices(active_class_ids, features.device)
        if active_indices.numel() == 0:
            return self.norm(features), features.new_tensor(0.0)

        joint_prompt, joint_loss = self.build_joint_prompt(features, active_indices)
        tri_prompt, tri_loss = self.build_tri_prompt(
            features,
            old_class_ids,
            new_class_ids,
            include_shared=False,
            old_score=old_score,
            new_score=new_score,
        )
        if torch.is_tensor(alpha):
            alpha_value = alpha.to(device=features.device, dtype=features.dtype)
        else:
            alpha_value = features.new_tensor(float(alpha))
        prompt = joint_prompt + alpha_value * tri_prompt
        prompted_features = self.norm(features + self.prompt_scale * prompt)
        tri_loss_scale = alpha_value.detach().abs().clamp(max=1.0)
        return prompted_features, joint_loss + tri_loss_scale * tri_loss

    @torch.no_grad()
    def promote_new_to_old(self, class_ids):
        indices = self.ids_to_indices(class_ids, self.new_private_keys.device)
        if indices.numel() == 0:
            return
        self.old_private_keys[indices].copy_(self.new_private_keys.index_select(0, indices))
        self.old_private_prompts[indices].copy_(self.new_private_prompts.index_select(0, indices))

    @torch.no_grad()
    def copy_joint_to_old(self, class_ids):
        indices = self.ids_to_indices(class_ids, self.private_keys.device)
        if indices.numel() == 0:
            return
        self.old_private_keys[indices].copy_(self.private_keys.index_select(0, indices))
        self.old_private_prompts[indices].copy_(self.private_prompts.index_select(0, indices))

    @torch.no_grad()
    def copy_joint_to_new(self, class_ids):
        indices = self.ids_to_indices(class_ids, self.private_keys.device)
        if indices.numel() == 0:
            return
        self.new_private_keys[indices].copy_(self.private_keys.index_select(0, indices))
        self.new_private_prompts[indices].copy_(self.private_prompts.index_select(0, indices))

    def forward_residual_gated(self, features, active_class_ids, gate, alpha):
        active_indices = torch.tensor(
            [int(class_id) - 1 for class_id in active_class_ids],
            device=features.device,
            dtype=torch.long,
        )
        if active_indices.numel() == 0:
            return self.norm(features), features.new_tensor(0.0)

        private_keys = self.private_keys.index_select(0, active_indices)
        private_prompts = self.private_prompts.index_select(0, active_indices)
        joint_keys = torch.cat([self.shared_keys, private_keys], dim=0)
        joint_prompts = torch.cat([self.shared_prompts, private_prompts], dim=0)
        joint_prompt = self.attend(features, joint_keys, joint_prompts)

        shared_prompt = self.attend(features, self.shared_keys, self.shared_prompts)
        private_prompt = self.attend(features, private_keys, private_prompts)
        gate = gate.clamp(0.0, 1.0)
        gated_prompt = (1.0 - gate) * shared_prompt + gate * private_prompt
        prompt = joint_prompt + float(alpha) * (gated_prompt - joint_prompt)
        prompted_features = self.norm(features + self.prompt_scale * prompt)

        prompt_loss = self.build_prompt_loss(private_keys, private_prompts, features)
        return prompted_features, prompt_loss


class HoustonPDPClassifier(nn.Module):
    """Classification adaptation of PDP: feature extractor + dual prompt pool + prototypes."""

    def __init__(
        self,
        in_channels,
        num_classes=15,
        dropout=0.2,
        prompt_shared_size=4,
        prompt_scale=0.2,
        prompt_temperature=0.2,
        prompt_ortho_weight=0.01,
        num_prototypes_per_class=1,
        prompt_fusion="joint",
        prototype_gate_hidden=128,
        prototype_gate_temperature=0.2,
        prototype_gate_alpha=0.2,
        spectral_descriptor="mean_grad",
        use_prototype_offsets=0,
        prototype_offset_scale=0.1,
        prototype_offset_start_session=2,
        use_role_aware_offsets=0,
        prototype_offset_old_scale=0.01,
        prototype_offset_new_scale=0.03,
        use_prototype_offset_gates=0,
        prototype_offset_gate_init=0.0,
        use_prototype_reliability=0,
        prototype_reliability_power=0.5,
        prototype_reliability_min=0.5,
        prototype_reliability_max=1.5,
        prototype_reliability_normalize=1,
        use_confusion_aware_offsets=0,
        confusion_offset_strength=1.0,
        confusion_offset_min=0.5,
        confusion_offset_max=1.5,
        confusion_offset_normalize=1,
        tri_pool_similarity="max",
        tri_pool_alpha=0.2,
        tri_pool_routing="prompt",
        tri_pool_proto_weight=0.5,
        tri_pool_consistency_weight=0.0,
        tri_pool_learnable_scale=0,
        tri_pool_scale_init=-4.0,
        tri_pool_start_session=2,
        use_spectral_prompt=0,
        spectral_prompt_alpha=0.1,
        spectral_prompt_mode="band_gate",
        spectral_prompt_start_session=1,
        use_prototype_aware_spectral_prompt=0,
        spectral_prompt_gate_threshold=0.35,
        spectral_prompt_gate_temperature=0.1,
        spectral_prompt_gate_min=0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_prototypes_per_class = max(1, int(num_prototypes_per_class))
        self.prompt_fusion = str(prompt_fusion)
        self.prototype_gate_temperature = float(prototype_gate_temperature)
        self.prototype_gate_alpha = float(prototype_gate_alpha)
        self.tri_pool_alpha = float(tri_pool_alpha)
        self.tri_pool_routing = str(tri_pool_routing)
        self.tri_pool_proto_weight = float(tri_pool_proto_weight)
        self.tri_pool_consistency_weight = float(tri_pool_consistency_weight)
        self.tri_pool_start_session = int(tri_pool_start_session)
        self.use_prototype_offsets = bool(use_prototype_offsets)
        self.prototype_offset_scale = float(prototype_offset_scale)
        self.prototype_offset_start_session = int(prototype_offset_start_session)
        self.use_role_aware_offsets = bool(use_role_aware_offsets)
        self.prototype_offset_old_scale = float(prototype_offset_old_scale)
        self.prototype_offset_new_scale = float(prototype_offset_new_scale)
        self.use_prototype_offset_gates = bool(use_prototype_offset_gates)
        self.prototype_offset_gate_init = float(prototype_offset_gate_init)
        self.use_prototype_reliability = bool(use_prototype_reliability)
        self.prototype_reliability_power = float(prototype_reliability_power)
        self.prototype_reliability_min = float(prototype_reliability_min)
        self.prototype_reliability_max = float(prototype_reliability_max)
        self.prototype_reliability_normalize = bool(prototype_reliability_normalize)
        self.use_confusion_aware_offsets = bool(use_confusion_aware_offsets)
        self.confusion_offset_strength = float(confusion_offset_strength)
        self.confusion_offset_min = float(confusion_offset_min)
        self.confusion_offset_max = float(confusion_offset_max)
        self.confusion_offset_normalize = bool(confusion_offset_normalize)
        self.use_spectral_prompt = bool(use_spectral_prompt)
        self.spectral_prompt_alpha = float(spectral_prompt_alpha)
        self.spectral_prompt_start_session = int(spectral_prompt_start_session)
        self.use_prototype_aware_spectral_prompt = bool(use_prototype_aware_spectral_prompt)
        self.spectral_prompt_gate_threshold = float(spectral_prompt_gate_threshold)
        self.spectral_prompt_gate_temperature = float(spectral_prompt_gate_temperature)
        self.spectral_prompt_gate_min = float(spectral_prompt_gate_min)
        if bool(tri_pool_learnable_scale):
            self.tri_pool_logit_scale = nn.Parameter(torch.tensor(float(tri_pool_scale_init)))
        else:
            self.tri_pool_logit_scale = None
        self.spectral_descriptor_mode = str(spectral_descriptor)
        spectral_dim = int(in_channels) if self.spectral_descriptor_mode == "mean" else int(in_channels) * 2 - 1
        if self.use_spectral_prompt:
            self.spectral_prompt = SpectralPromptBranch(in_channels, mode=spectral_prompt_mode)
        else:
            self.spectral_prompt = None
        self.features = HoustonFeatureExtractor(in_channels)
        self.prompt_pool = DualPromptPool(
            feature_dim=256,
            num_classes=num_classes,
            shared_size=prompt_shared_size,
            prompt_scale=prompt_scale,
            temperature=prompt_temperature,
            ortho_weight=prompt_ortho_weight,
            tri_pool_similarity=tri_pool_similarity,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        if self.prompt_fusion in {"prototype_gate", "residual_gate"}:
            hidden_dim = max(16, int(prototype_gate_hidden))
            self.prototype_gate = nn.Sequential(
                nn.Linear(256 * 2 + 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 1),
            )
        else:
            self.prototype_gate = None
        self.register_buffer("prototypes", torch.zeros(num_classes, self.num_prototypes_per_class, 256))
        self.register_buffer("prototype_counts", torch.zeros(num_classes, self.num_prototypes_per_class))
        self.register_buffer("prototype_confusion_factors", torch.ones(num_classes, self.num_prototypes_per_class))
        if self.use_prototype_offsets:
            self.prototype_offsets = nn.Parameter(torch.zeros(num_classes, self.num_prototypes_per_class, 256))
            if self.use_prototype_offset_gates:
                self.prototype_offset_gates = nn.Parameter(
                    torch.full((num_classes, self.num_prototypes_per_class, 1), self.prototype_offset_gate_init)
                )
            else:
                self.prototype_offset_gates = None
        else:
            self.prototype_offsets = None
            self.prototype_offset_gates = None
        self.register_buffer(
            "spectral_prototypes",
            torch.zeros(num_classes, self.num_prototypes_per_class, spectral_dim),
        )
        self.register_buffer("spectral_prototype_counts", torch.zeros(num_classes, self.num_prototypes_per_class))
        self.prompt_old_class_ids = []
        self.prompt_new_class_ids = list(range(1, self.num_classes + 1))
        self.prompt_session_idx = 1
        self.promoted_prompt_classes = set()

    def set_prompt_task_context(self, old_class_ids=None, new_class_ids=None, session_idx=1):
        self.prompt_old_class_ids = [int(class_id) for class_id in (old_class_ids or [])]
        self.prompt_new_class_ids = [int(class_id) for class_id in (new_class_ids or [])]
        self.prompt_session_idx = int(session_idx)

    def promote_new_prompts_to_old(self, class_ids):
        if self.prompt_fusion not in {"tri_pool", "joint_tri_pool"}:
            return []
        new_promotions = [
            int(class_id)
            for class_id in (class_ids or [])
            if int(class_id) not in self.promoted_prompt_classes
        ]
        if not new_promotions:
            return []
        if self.prompt_fusion == "joint_tri_pool":
            self.prompt_pool.copy_joint_to_old(new_promotions)
        else:
            self.prompt_pool.promote_new_to_old(new_promotions)
        self.promoted_prompt_classes.update(new_promotions)
        return new_promotions

    def initialize_current_prompts_from_joint(self, class_ids):
        if self.prompt_fusion == "joint_tri_pool":
            self.prompt_pool.copy_joint_to_new(class_ids)

    def effective_tri_pool_alpha(self):
        alpha = float(self.tri_pool_alpha)
        if self.tri_pool_logit_scale is not None:
            alpha *= float(torch.sigmoid(self.tri_pool_logit_scale.detach()).cpu())
        return alpha

    def apply_spectral_prompt(self, x, gate=None):
        if self.spectral_prompt is None or self.spectral_prompt_alpha <= 0:
            return x
        if self.prompt_session_idx < self.spectral_prompt_start_session:
            return x
        residual = self.spectral_prompt(x)
        if gate is not None:
            residual = residual * gate.to(device=x.device, dtype=x.dtype)
        return x + self.spectral_prompt_alpha * residual

    def prototype_max_similarity(self, features, class_ids):
        indices = torch.tensor(
            [int(class_id) - 1 for class_id in (class_ids or [])],
            device=features.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            return None

        prototypes = self.feature_prototypes_for_indices(indices, device=features.device, calibrated=True)
        counts = self.prototype_counts.index_select(0, indices).to(features.device)
        valid = counts > 0
        if not bool(valid.any()):
            return None

        flat_prototypes = prototypes[valid]
        similarities = F.normalize(features, dim=1) @ F.normalize(flat_prototypes, dim=1).t()
        return similarities.max(dim=1, keepdim=True).values

    def spectral_prompt_gate(self, features):
        if not self.use_prototype_aware_spectral_prompt:
            return None
        if self.spectral_prompt is None or self.prompt_session_idx < self.spectral_prompt_start_session:
            return None

        old_similarity = self.prototype_max_similarity(features, self.prompt_old_class_ids)
        if old_similarity is None:
            return None

        temperature = max(float(self.spectral_prompt_gate_temperature), 1e-6)
        gate = torch.sigmoid((float(self.spectral_prompt_gate_threshold) - old_similarity) / temperature)
        min_gate = min(max(float(self.spectral_prompt_gate_min), 0.0), 1.0)
        if min_gate > 0:
            gate = min_gate + (1.0 - min_gate) * gate
        return gate.view(-1, 1, 1, 1)

    def prototype_offsets_active(self):
        return (
            self.prototype_offsets is not None
            and self.prototype_offset_scale > 0
            and self.prompt_session_idx >= self.prototype_offset_start_session
        )

    def prototype_offset_scales_for_indices(self, indices, device=None):
        target_device = device if device is not None else indices.device
        scales = torch.full(
            (indices.numel(), 1, 1),
            float(self.prototype_offset_scale),
            device=target_device,
            dtype=torch.float32,
        )
        if not self.use_role_aware_offsets or indices.numel() == 0:
            return scales

        query_indices = indices.to(target_device)
        if self.prompt_old_class_ids:
            old_indices = torch.tensor(
                [int(class_id) - 1 for class_id in self.prompt_old_class_ids],
                device=target_device,
                dtype=torch.long,
            )
            is_old = (query_indices.unsqueeze(1) == old_indices.unsqueeze(0)).any(dim=1)
            scales[is_old] = float(self.prototype_offset_old_scale)
        if self.prompt_new_class_ids:
            new_indices = torch.tensor(
                [int(class_id) - 1 for class_id in self.prompt_new_class_ids],
                device=target_device,
                dtype=torch.long,
            )
            is_new = (query_indices.unsqueeze(1) == new_indices.unsqueeze(0)).any(dim=1)
            scales[is_new] = float(self.prototype_offset_new_scale)
        return scales

    def feature_prototypes_for_indices(self, indices, device=None, calibrated=True):
        prototypes = self.prototypes.index_select(0, indices)
        if device is not None:
            prototypes = prototypes.to(device)
        if calibrated and self.prototype_offsets_active():
            offsets = self.prototype_offsets.index_select(0, indices)
            if device is not None:
                offsets = offsets.to(device)
            if self.prototype_offset_gates is not None:
                gates = self.prototype_offset_gates.index_select(0, indices)
                if device is not None:
                    gates = gates.to(device)
                offsets = offsets * (1.0 + torch.tanh(gates))
            if self.use_prototype_reliability:
                reliability = self.prototype_reliability_for_indices(indices, device=offsets.device)
                offsets = offsets * reliability.unsqueeze(-1)
            if self.use_confusion_aware_offsets:
                confusion_factors = self.prototype_confusion_factors.index_select(0, indices)
                if device is not None:
                    confusion_factors = confusion_factors.to(device)
                offsets = offsets * confusion_factors.unsqueeze(-1)
            scale = self.prototype_offset_scales_for_indices(indices, device=offsets.device)
            prototypes = prototypes + scale.to(dtype=offsets.dtype) * offsets
        return prototypes

    def prototype_offset_scale_stats(self, class_ids):
        if not self.use_role_aware_offsets:
            return None
        indices = [int(class_id) - 1 for class_id in (class_ids or [])]
        if not indices:
            return None
        with torch.no_grad():
            index_tensor = torch.tensor(indices, device=self.prototypes.device, dtype=torch.long)
            scales = self.prototype_offset_scales_for_indices(index_tensor, device=self.prototypes.device)
            return {
                "mean": float(scales.mean().detach().cpu()),
                "min": float(scales.min().detach().cpu()),
                "max": float(scales.max().detach().cpu()),
            }


    def prototype_reliability_for_indices(self, indices, device=None):
        counts = self.prototype_counts.index_select(0, indices).float()
        if device is not None:
            counts = counts.to(device)
        valid = counts > 0
        max_counts = counts.max(dim=1, keepdim=True).values.clamp_min(1.0)
        reliability = (counts.clamp_min(0.0) / max_counts).pow(max(self.prototype_reliability_power, 0.0))
        reliability = torch.where(valid, reliability, torch.ones_like(reliability))
        reliability = reliability.clamp(
            min=float(self.prototype_reliability_min),
            max=float(self.prototype_reliability_max),
        )
        if self.prototype_reliability_normalize and bool(valid.any()):
            mean_value = reliability[valid].mean().clamp_min(1e-6)
            reliability = (reliability / mean_value).clamp(
                min=float(self.prototype_reliability_min),
                max=float(self.prototype_reliability_max),
            )
        return reliability

    def reset_prototype_offsets(self, class_ids):
        if self.prototype_offsets is None:
            return
        indices = [int(class_id) - 1 for class_id in (class_ids or [])]
        if not indices:
            return
        with torch.no_grad():
            index_tensor = torch.tensor(indices, device=self.prototype_offsets.device)
            self.prototype_offsets[index_tensor].zero_()
            if self.prototype_offset_gates is not None:
                self.prototype_offset_gates[index_tensor].fill_(self.prototype_offset_gate_init)

    def prototype_offset_gate_stats(self, class_ids):
        if self.prototype_offset_gates is None:
            return None
        indices = [int(class_id) - 1 for class_id in (class_ids or [])]
        if not indices:
            return None
        with torch.no_grad():
            index_tensor = torch.tensor(indices, device=self.prototype_offset_gates.device)
            factors = 1.0 + torch.tanh(self.prototype_offset_gates[index_tensor])
            return {
                "mean": float(factors.mean().detach().cpu()),
                "min": float(factors.min().detach().cpu()),
                "max": float(factors.max().detach().cpu()),
            }

    def prototype_reliability_stats(self, class_ids):
        if not self.use_prototype_reliability:
            return None
        indices = [int(class_id) - 1 for class_id in (class_ids or [])]
        if not indices:
            return None
        with torch.no_grad():
            index_tensor = torch.tensor(indices, device=self.prototype_counts.device)
            reliability = self.prototype_reliability_for_indices(index_tensor, device=self.prototype_counts.device)
            valid = self.prototype_counts.index_select(0, index_tensor) > 0
            values = reliability[valid] if bool(valid.any()) else reliability.reshape(-1)
            return {
                "mean": float(values.mean().detach().cpu()),
                "min": float(values.min().detach().cpu()),
                "max": float(values.max().detach().cpu()),
            }

    def set_confusion_offset_factors(self, confusion, class_ids):
        if not self.use_confusion_aware_offsets:
            return None
        class_ids = [int(class_id) for class_id in (class_ids or [])]
        if confusion is None or not class_ids:
            return None

        confusion_array = np.asarray(confusion, dtype=np.float64)
        factors = torch.ones(self.num_classes, device=self.prototype_confusion_factors.device)
        scores = []
        valid_class_ids = []
        for class_id in class_ids:
            cls_idx = int(class_id) - 1
            if cls_idx < 0 or cls_idx >= confusion_array.shape[0]:
                continue
            row = confusion_array[cls_idx]
            total = float(row.sum())
            if total <= 0:
                continue
            correct = float(row[cls_idx]) if cls_idx < row.shape[0] else 0.0
            score = max(0.0, 1.0 - correct / total)
            scores.append(score)
            valid_class_ids.append(class_id)

        if not scores:
            return None

        score_tensor = torch.tensor(scores, device=factors.device, dtype=factors.dtype)
        if self.confusion_offset_normalize:
            mean_score = score_tensor.mean().clamp_min(1e-6)
            centered = score_tensor / mean_score - 1.0
            factor_values = 1.0 + float(self.confusion_offset_strength) * centered
        else:
            factor_values = 1.0 + float(self.confusion_offset_strength) * score_tensor
        factor_values = factor_values.clamp(
            min=float(self.confusion_offset_min),
            max=float(self.confusion_offset_max),
        )
        for class_id, factor in zip(valid_class_ids, factor_values):
            factors[int(class_id) - 1] = factor
        self.prototype_confusion_factors.copy_(factors.unsqueeze(1).expand_as(self.prototype_confusion_factors))
        values = factor_values.detach().cpu()
        return {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    def prototype_confusion_factor_stats(self, class_ids):
        if not self.use_confusion_aware_offsets:
            return None
        indices = [int(class_id) - 1 for class_id in (class_ids or [])]
        if not indices:
            return None
        with torch.no_grad():
            index_tensor = torch.tensor(indices, device=self.prototype_confusion_factors.device)
            factors = self.prototype_confusion_factors.index_select(0, index_tensor)
            return {
                "mean": float(factors.mean().detach().cpu()),
                "min": float(factors.min().detach().cpu()),
                "max": float(factors.max().detach().cpu()),
            }

    def prototype_group_score(self, features, class_ids):
        indices = torch.tensor(
            [int(class_id) - 1 for class_id in (class_ids or [])],
            device=features.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            return None

        prototypes = self.prototypes.index_select(0, indices).to(features.device)
        counts = self.prototype_counts.index_select(0, indices).to(features.device)
        valid = counts > 0
        if not bool(valid.any()):
            return None

        flat_prototypes = prototypes[valid]
        similarities = F.normalize(features, dim=1) @ F.normalize(flat_prototypes, dim=1).t()
        if self.prompt_pool.tri_pool_similarity == "mean":
            return similarities.mean(dim=1, keepdim=True)
        if self.prompt_pool.tri_pool_similarity == "logsumexp":
            return torch.logsumexp(similarities, dim=1, keepdim=True)
        return similarities.max(dim=1, keepdim=True).values

    def prompt_key_group_score(self, features, class_ids, pool="old"):
        indices = torch.tensor(
            [int(class_id) - 1 for class_id in (class_ids or [])],
            device=features.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            return None
        if pool == "new":
            keys = self.prompt_pool.new_private_keys.index_select(0, indices)
        else:
            keys = self.prompt_pool.old_private_keys.index_select(0, indices)
        return self.prompt_pool.group_similarity(features, keys)

    def prompt_prototype_consistency_score(self, features, class_ids, pool="old"):
        indices = torch.tensor(
            [int(class_id) - 1 for class_id in (class_ids or [])],
            device=features.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            return None

        if pool == "new":
            keys = self.prompt_pool.new_private_keys.index_select(0, indices).to(features.device)
        else:
            keys = self.prompt_pool.old_private_keys.index_select(0, indices).to(features.device)

        prototypes = self.prototypes.index_select(0, indices).to(features.device)
        counts = self.prototype_counts.index_select(0, indices).to(features.device)
        valid = counts > 0
        if not bool(valid.any()):
            return None

        flat_prototypes = prototypes[valid]
        similarities = F.normalize(keys, dim=1) @ F.normalize(flat_prototypes, dim=1).t()
        if self.prompt_pool.tri_pool_similarity == "mean":
            score = similarities.mean()
        elif self.prompt_pool.tri_pool_similarity == "logsumexp":
            score = torch.logsumexp(similarities.reshape(-1), dim=0)
        else:
            score = similarities.max()
        return score.reshape(1, 1).expand(features.size(0), 1)

    def prototype_guided_group_score(self, features, class_ids, pool="old"):
        prompt_score = self.prompt_key_group_score(features, class_ids, pool=pool)
        prototype_score = self.prototype_group_score(features, class_ids)
        if prompt_score is None and prototype_score is None:
            return None
        if prompt_score is None:
            score = prototype_score
        elif prototype_score is None:
            score = prompt_score
        else:
            proto_weight = min(max(self.tri_pool_proto_weight, 0.0), 1.0)
            score = (1.0 - proto_weight) * prompt_score + proto_weight * prototype_score

        consistency_score = self.prompt_prototype_consistency_score(features, class_ids, pool=pool)
        if consistency_score is not None and self.tri_pool_consistency_weight > 0:
            score = score + self.tri_pool_consistency_weight * consistency_score
        return score

    def build_prototype_gate(self, features, active_class_ids):
        if self.prototype_gate is None:
            return None

        indices = torch.tensor(
            [int(class_id) - 1 for class_id in active_class_ids],
            device=features.device,
            dtype=torch.long,
        )
        if indices.numel() == 0:
            return None

        prototypes = self.prototypes.index_select(0, indices).to(features.device)
        counts = self.prototype_counts.index_select(0, indices).to(features.device)
        valid = counts > 0
        if not bool(valid.any()):
            return None

        flat_prototypes = prototypes[valid]
        norm_features = F.normalize(features, dim=1)
        norm_prototypes = F.normalize(flat_prototypes, dim=1)
        similarities = norm_features @ norm_prototypes.t()
        attention = torch.softmax(
            similarities / max(self.prototype_gate_temperature, 1e-6),
            dim=1,
        )
        prototype_context = attention @ flat_prototypes

        if similarities.size(1) > 1:
            top2 = similarities.topk(k=2, dim=1).values
            max_similarity = top2[:, :1]
            similarity_margin = top2[:, :1] - top2[:, 1:2]
        else:
            max_similarity = similarities
            similarity_margin = torch.zeros_like(max_similarity)

        gate_input = torch.cat(
            [features, prototype_context, max_similarity, similarity_margin],
            dim=1,
        )
        return torch.sigmoid(self.prototype_gate(gate_input))

    def forward_for_session(self, x, active_class_ids, return_features=False):
        if self.use_prototype_aware_spectral_prompt and self.spectral_prompt is not None:
            features = self.features(x)
            spectral_gate = self.spectral_prompt_gate(features)
            if spectral_gate is not None:
                x = self.apply_spectral_prompt(x, gate=spectral_gate)
                features = self.features(x)
        else:
            x = self.apply_spectral_prompt(x)
            features = self.features(x)
        if self.prompt_fusion == "tri_pool":
            prompted_features, prompt_loss = self.prompt_pool.forward_tri_pool(
                features,
                self.prompt_old_class_ids,
                self.prompt_new_class_ids,
            )
        elif self.prompt_fusion == "joint_tri_pool":
            if self.prompt_session_idx < self.tri_pool_start_session:
                prompted_features, prompt_loss = self.prompt_pool(features, active_class_ids)
            else:
                old_score = None
                new_score = None
                if self.tri_pool_routing == "prototype":
                    old_score = self.prototype_group_score(features, self.prompt_old_class_ids)
                    new_score = self.prototype_group_score(features, self.prompt_new_class_ids)
                    if old_score is None or new_score is None:
                        old_score = None
                        new_score = None
                elif self.tri_pool_routing == "hybrid":
                    old_score = self.prototype_guided_group_score(
                        features,
                        self.prompt_old_class_ids,
                        pool="old",
                    )
                    new_score = self.prototype_guided_group_score(
                        features,
                        self.prompt_new_class_ids,
                        pool="new",
                    )
                    if old_score is None or new_score is None:
                        old_score = None
                        new_score = None
                tri_pool_alpha = self.tri_pool_alpha
                if self.tri_pool_logit_scale is not None:
                    tri_pool_alpha = tri_pool_alpha * torch.sigmoid(self.tri_pool_logit_scale)
                prompted_features, prompt_loss = self.prompt_pool.forward_joint_tri_pool(
                    features,
                    active_class_ids,
                    self.prompt_old_class_ids,
                    self.prompt_new_class_ids,
                    tri_pool_alpha,
                    old_score=old_score,
                    new_score=new_score,
                )
        else:
            gate = self.build_prototype_gate(features, active_class_ids)
            if gate is None:
                prompted_features, prompt_loss = self.prompt_pool(features, active_class_ids)
            elif self.prompt_fusion == "residual_gate":
                prompted_features, prompt_loss = self.prompt_pool.forward_residual_gated(
                    features,
                    active_class_ids,
                    gate,
                    self.prototype_gate_alpha,
                )
            else:
                prompted_features, prompt_loss = self.prompt_pool.forward_gated(features, active_class_ids, gate)
        logits = self.classifier(prompted_features)
        if return_features:
            return logits, prompted_features, prompt_loss
        return logits, prompt_loss

    def forward(self, x):
        all_class_ids = list(range(1, self.num_classes + 1))
        logits, _ = self.forward_for_session(x, all_class_ids, return_features=False)
        return logits


def get_args_parser():
    parser = argparse.ArgumentParser("HSI patch classification")
    parser.add_argument("--hsi_file", default="", type=str)
    parser.add_argument("--lidar_file", default="", type=str)
    parser.add_argument("--train_label_file", default="", type=str)
    parser.add_argument("--test_label_file", default="", type=str)
    parser.add_argument("--data_dir", default="", type=str)
    parser.add_argument("--hsi_key", default="HSI", type=str)
    parser.add_argument("--lidar_key", default=None, type=str)
    parser.add_argument("--hsi_label_key", default=None, type=str)
    parser.add_argument("--hsi_band_indices", default="", type=str)
    parser.add_argument("--hsi_patch_size", default=15, type=int)
    parser.add_argument("--hsi_normalize_per_band", default=1, type=int)
    parser.add_argument("--use_lidar", default=0, type=int)
    parser.add_argument("--lidar_normalize_per_band", default=1, type=int)
    parser.add_argument("--sessions", default="1-9,10-12,13-15", type=str)
    parser.add_argument(
        "--num_classes",
        default=0,
        type=int,
        help="Total class count. By default it is inferred from --sessions.",
    )
    parser.add_argument("--mode", default="incremental", choices=["incremental", "all"])
    parser.add_argument("--incremental_train", default="current", choices=["current", "cumulative", "replay"])
    parser.add_argument("--model", default="pdp", choices=["pdp", "cnn"])
    parser.add_argument("--max_train_samples_per_class", default=0, type=int)
    parser.add_argument("--max_test_samples_per_class", default=0, type=int)
    parser.add_argument("--replay_old_samples_per_class", default=50, type=int)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--class_weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--class_weight_gamma", default=1.0, type=float)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--prompt_shared_size", default=4, type=int)
    parser.add_argument("--prompt_scale", default=0.2, type=float)
    parser.add_argument("--prompt_temperature", default=0.2, type=float)
    parser.add_argument("--prompt_ortho_weight", default=0.01, type=float)
    parser.add_argument("--lambda_prompt_ortho", default=1.0, type=float)
    parser.add_argument(
        "--prompt_fusion",
        default="joint",
        choices=["joint", "prototype_gate", "residual_gate", "tri_pool", "joint_tri_pool"],
    )
    parser.add_argument("--tri_pool_similarity", default="max", choices=["max", "mean", "logsumexp"])
    parser.add_argument("--tri_pool_alpha", default=0.2, type=float)
    parser.add_argument("--tri_pool_routing", default="prompt", choices=["prompt", "prototype", "hybrid"])
    parser.add_argument("--tri_pool_proto_weight", default=0.5, type=float)
    parser.add_argument("--tri_pool_consistency_weight", default=0.0, type=float)
    parser.add_argument("--tri_pool_learnable_scale", default=0, type=int)
    parser.add_argument("--tri_pool_scale_init", default=-4.0, type=float)
    parser.add_argument("--tri_pool_start_session", default=2, type=int)
    parser.add_argument("--freeze_old_prompts_after_promotion", default=1, type=int)
    parser.add_argument("--prototype_gate_hidden", default=128, type=int)
    parser.add_argument("--prototype_gate_temperature", default=0.2, type=float)
    parser.add_argument("--prototype_gate_alpha", default=0.2, type=float)
    parser.add_argument("--prompt_param_fusion", default=0, type=int)
    parser.add_argument("--prompt_fuse_topk_ori", default=0.5, type=float)
    parser.add_argument("--prompt_fuse_topk_new", default=0.5, type=float)
    parser.add_argument("--lambda_confusion_margin", default=0.05, type=float)
    parser.add_argument("--confusion_margin", default=0.2, type=float)
    parser.add_argument("--confusion_topk", default=1, type=int)
    parser.add_argument("--confusion_margin_scope", default="current", choices=["all", "current"])
    parser.add_argument("--confusion_margin_mode", default="fixed", choices=["fixed", "adaptive_pair"])
    parser.add_argument("--adaptive_confusion_min_rate", default=0.05, type=float)
    parser.add_argument("--adaptive_confusion_gamma", default=1.0, type=float)
    parser.add_argument("--adaptive_confusion_min_weight", default=1.0, type=float)
    parser.add_argument("--lambda_old_logit_distill", default=0.0, type=float)
    parser.add_argument("--old_logit_distill_temperature", default=2.0, type=float)
    parser.add_argument(
        "--old_logit_distill_scope",
        default="old",
        choices=["old", "all"],
        help=(
            "Samples used for old-logit distillation. 'old' preserves the "
            "existing replay-only behavior; 'all' also distills old outputs "
            "on current-class samples, as required by an LwF-style baseline."
        ),
    )
    parser.add_argument("--use_prototypes", default=1, type=int)
    parser.add_argument("--prototype_alpha", default=0.7, type=float)
    parser.add_argument("--prototype_temperature", default=0.2, type=float)
    parser.add_argument("--use_prototype_margin_gate", default=0, type=int)
    parser.add_argument("--prototype_margin_gate_threshold", default=0.2, type=float)
    parser.add_argument("--prototype_margin_gate_temperature", default=0.5, type=float)
    parser.add_argument("--prototype_margin_gate_min", default=0.5, type=float)
    parser.add_argument("--num_prototypes_per_class", default=5, type=int)
    parser.add_argument("--adaptive_prototypes", default=0, type=int)
    parser.add_argument("--adaptive_min_prototypes", default=1, type=int)
    parser.add_argument("--adaptive_dispersion_low", default=0.08, type=float)
    parser.add_argument("--adaptive_dispersion_high", default=0.20, type=float)
    parser.add_argument("--adaptive_k_mode", default="dispersion", choices=["dispersion", "compactness", "elbow"])
    parser.add_argument("--adaptive_compactness_target", default=0.08, type=float)
    parser.add_argument("--adaptive_elbow_min_gain", default=0.03, type=float)
    parser.add_argument("--use_role_aware_adaptive_k", default=0, type=int)
    parser.add_argument("--adaptive_old_elbow_min_gain", default=0.10, type=float)
    parser.add_argument("--adaptive_new_elbow_min_gain", default=0.05, type=float)
    parser.add_argument("--adaptive_old_min_age", default=1, type=int)
    parser.add_argument(
        "--prototype_pooling",
        default="logsumexp",
        choices=["max", "mean", "logsumexp", "logmeanexp", "softmax"],
    )
    parser.add_argument("--prototype_kmeans_iters", default=10, type=int)
    parser.add_argument("--prototype_ema", default=0, type=int)
    parser.add_argument("--prototype_ema_momentum", default=0.7, type=float)
    parser.add_argument("--use_prototype_offsets", default=0, type=int)
    parser.add_argument("--prototype_offset_scale", default=0.1, type=float)
    parser.add_argument("--prototype_offset_start_session", default=2, type=int)
    parser.add_argument("--use_role_aware_offsets", default=0, type=int)
    parser.add_argument("--prototype_offset_old_scale", default=0.01, type=float)
    parser.add_argument("--prototype_offset_new_scale", default=0.03, type=float)
    parser.add_argument("--prototype_offset_epochs", default=20, type=int)
    parser.add_argument("--prototype_offset_lr", default=1e-3, type=float)
    parser.add_argument("--prototype_offset_l2", default=0.01, type=float)
    parser.add_argument("--use_role_aware_offset_l2", default=0, type=int)
    parser.add_argument("--prototype_offset_old_l2", default=0.1, type=float)
    parser.add_argument("--prototype_offset_new_l2", default=0.02, type=float)
    parser.add_argument("--use_balanced_offset_loss", default=0, type=int)
    parser.add_argument("--lambda_offset_new", default=0.5, type=float)
    parser.add_argument("--lambda_offset_old_stability", default=0.2, type=float)
    parser.add_argument("--offset_stability_temperature", default=1.0, type=float)
    parser.add_argument("--use_prototype_offset_gates", default=0, type=int)
    parser.add_argument("--prototype_offset_gate_init", default=0.0, type=float)
    parser.add_argument("--prototype_offset_gate_l2", default=0.001, type=float)
    parser.add_argument("--use_prototype_reliability", default=0, type=int)
    parser.add_argument("--prototype_reliability_power", default=0.5, type=float)
    parser.add_argument("--prototype_reliability_min", default=0.5, type=float)
    parser.add_argument("--prototype_reliability_max", default=1.5, type=float)
    parser.add_argument("--prototype_reliability_normalize", default=1, type=int)
    parser.add_argument("--use_confusion_aware_offsets", default=0, type=int)
    parser.add_argument("--confusion_offset_strength", default=1.0, type=float)
    parser.add_argument("--confusion_offset_min", default=0.5, type=float)
    parser.add_argument("--confusion_offset_max", default=1.5, type=float)
    parser.add_argument("--confusion_offset_normalize", default=1, type=int)
    parser.add_argument("--use_capst", default=0, type=int)
    parser.add_argument("--capst_start_session", default=2, type=int)
    parser.add_argument("--capst_mode", default="quantile", choices=["quantile", "mean_std", "kmeans"])
    parser.add_argument("--capst_score_type", default="positive", choices=["positive", "margin"])
    parser.add_argument("--capst_quantile", default=0.2, type=float)
    parser.add_argument("--capst_beta", default=1.0, type=float)
    parser.add_argument("--capst_temperature", default=0.05, type=float)
    parser.add_argument("--capst_min_weight", default=0.2, type=float)
    parser.add_argument("--capst_use_calibrated_prototypes", default=0, type=int)
    parser.add_argument("--capst_scope", default="all", choices=["all", "old", "current"])
    parser.add_argument("--use_spectral_prototypes", default=0, type=int)
    parser.add_argument("--spectral_descriptor", default="mean_grad", choices=["mean", "mean_grad"])
    parser.add_argument("--spectral_prototype_alpha", default=0.5, type=float)
    parser.add_argument("--spectral_prototype_temperature", default=0.2, type=float)
    parser.add_argument(
        "--spectral_prototype_pooling",
        default="max",
        choices=["max", "mean", "logsumexp", "logmeanexp", "softmax"],
    )
    parser.add_argument("--use_spectral_prompt", default=0, type=int)
    parser.add_argument("--spectral_prompt_alpha", default=0.1, type=float)
    parser.add_argument("--spectral_prompt_mode", default="band_gate", choices=["band_gate", "fft_gate"])
    parser.add_argument("--spectral_prompt_start_session", default=1, type=int)
    parser.add_argument("--use_prototype_aware_spectral_prompt", default=0, type=int)
    parser.add_argument("--spectral_prompt_gate_threshold", default=0.35, type=float)
    parser.add_argument("--spectral_prompt_gate_temperature", default=0.1, type=float)
    parser.add_argument("--spectral_prompt_gate_min", default=0.0, type=float)
    parser.add_argument("--lambda_spectral_consistency", default=0.0, type=float)
    parser.add_argument("--spectral_consistency_temperature", default=0.2, type=float)
    parser.add_argument(
        "--spectral_consistency_pooling",
        default="max",
        choices=["max", "mean", "logsumexp", "logmeanexp", "softmax"],
    )
    parser.add_argument("--spectral_replay_filter", default=0, type=int)
    parser.add_argument("--spectral_replay_mode", default="top", choices=["top", "diverse"])
    parser.add_argument("--spectral_replay_min_keep_ratio", default=1.0, type=float)
    parser.add_argument("--feature_replay_filter", default=0, type=int)
    parser.add_argument("--feature_replay_mode", default="top", choices=["top", "diverse", "hybrid", "hard"])
    parser.add_argument("--feature_replay_min_keep_ratio", default=1.0, type=float)
    parser.add_argument("--feature_replay_chunk_size", default=512, type=int)
    parser.add_argument("--freeze_backbone_after_base", default=1, type=int)
    parser.add_argument("--freeze_shared_prompts_after_base", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--output_dir", default="./houston_cls_outputs", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    return parser


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_paths(args):
    data_dir = Path(args.data_dir) if args.data_dir else None
    if not args.hsi_file:
        if data_dir is None:
            raise ValueError("Pass --hsi_file or --data_dir.")
        args.hsi_file = str(data_dir / "HSI.mat")
    if bool(args.use_lidar) and not args.lidar_file:
        if data_dir is None:
            raise ValueError("Pass --lidar_file or --data_dir when --use_lidar 1.")
        lidar_candidates = ["LiDAR.mat", "LIDAR.mat", "Lidar.mat", "lidar.mat", "DSM.mat", "dsm.mat"]
        for name in lidar_candidates:
            candidate = data_dir / name
            if candidate.exists():
                args.lidar_file = str(candidate)
                break
        if not args.lidar_file:
            raise ValueError(
                "Pass --lidar_file when --use_lidar 1. "
                f"Tried: {', '.join(lidar_candidates)} under {data_dir}."
            )
    if not args.train_label_file:
        if data_dir is None:
            raise ValueError("Pass --train_label_file or --data_dir.")
        args.train_label_file = str(data_dir / "TRLabel.mat")
    if not args.test_label_file:
        if data_dir is None:
            raise ValueError("Pass --test_label_file or --data_dir.")
        args.test_label_file = str(data_dir / "TSLabel.mat")


def make_loader(args, hsi_array, label_file, class_ids, train, max_samples_by_class=None):
    dataset = HoustonPatchClassification(
        hsi_file=args.hsi_file,
        label_file=label_file,
        class_ids=class_ids,
        hsi_key=args.hsi_key,
        label_key=args.hsi_label_key,
        band_indices=args.hsi_band_indices,
        patch_size=args.hsi_patch_size,
        normalize_per_band=bool(args.hsi_normalize_per_band),
        max_samples_per_class=args.max_train_samples_per_class if train else args.max_test_samples_per_class,
        max_samples_by_class=max_samples_by_class,
        sample_filter=getattr(args, "sample_filter", None) if train else None,
        seed=args.seed,
        hsi_array=hsi_array,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return dataset, loader


def patch_tensor_from_coords(hsi_array, coords, patch_size):
    half = int(patch_size) // 2
    padded = np.pad(hsi_array, ((half, half), (half, half), (0, 0)), mode="edge")
    patches = []
    for y, x in coords:
        patch = padded[int(y):int(y) + int(patch_size), int(x):int(x) + int(patch_size), :]
        patches.append(np.moveaxis(patch, 2, 0))
    patches = np.ascontiguousarray(np.asarray(patches, dtype=np.float32))
    return torch.from_numpy(patches).float()


def select_replay_indices_from_scores(scores, keep_count, mode="top", rng=None):
    keep_count = min(int(keep_count), int(scores.shape[0]))
    if keep_count <= 0:
        return np.asarray([], dtype=np.int64)

    rng = rng or np.random.default_rng(0)
    max_scores = scores.max(axis=1)
    if mode == "hard":
        return np.argsort(max_scores)[:keep_count].astype(np.int64)

    if mode == "hybrid":
        prototype_count = keep_count // 2
        coverage_count = keep_count - prototype_count
        prototype_indices = select_replay_indices_from_scores(
            scores,
            prototype_count,
            mode="diverse",
            rng=rng,
        )
        selected = prototype_indices.tolist()
        selected_set = set(selected)
        remaining = [idx for idx in range(scores.shape[0]) if idx not in selected_set]
        if remaining and coverage_count > 0:
            if len(remaining) > coverage_count:
                coverage_indices = rng.choice(remaining, size=coverage_count, replace=False).tolist()
            else:
                coverage_indices = remaining
            selected.extend(coverage_indices)
        if len(selected) < keep_count:
            selected_set = set(selected)
            fallback = [idx for idx in np.argsort(max_scores)[::-1].tolist() if idx not in selected_set]
            selected.extend(fallback[: keep_count - len(selected)])
        return np.asarray(selected[:keep_count], dtype=np.int64)

    if mode == "diverse" and scores.shape[1] > 1:
        assignments = scores.argmax(axis=1)
        selected = []
        num_clusters = int(scores.shape[1])
        base_quota = keep_count // num_clusters
        remainder = keep_count % num_clusters
        for cluster_idx in range(num_clusters):
            cluster_indices = np.where(assignments == cluster_idx)[0]
            if len(cluster_indices) == 0:
                continue
            quota = base_quota + (1 if cluster_idx < remainder else 0)
            quota = min(quota, len(cluster_indices))
            ranked = cluster_indices[np.argsort(max_scores[cluster_indices])[::-1]]
            selected.extend(ranked[:quota].tolist())

        if len(selected) < keep_count:
            selected_set = set(selected)
            remaining = [idx for idx in np.argsort(max_scores)[::-1].tolist() if idx not in selected_set]
            selected.extend(remaining[: keep_count - len(selected)])
        return np.asarray(selected[:keep_count], dtype=np.int64)

    return np.argsort(max_scores)[::-1][:keep_count].astype(np.int64)


def build_feature_replay_filter(args, model, hsi_array, previous_classes, device):
    if (
        model is None
        or not bool(getattr(args, "feature_replay_filter", 0))
        or not previous_classes
        or not hasattr(model, "prototypes")
    ):
        return None

    previous_class_set = {int(class_id) for class_id in previous_classes}
    min_keep_ratio = float(getattr(args, "feature_replay_min_keep_ratio", 1.0))
    replay_mode = str(getattr(args, "feature_replay_mode", "top"))
    chunk_size = max(1, int(getattr(args, "feature_replay_chunk_size", 512)))
    active_class_ids = list(previous_classes)
    rng = np.random.default_rng(int(getattr(args, "seed", 42)))

    def sample_filter(class_id, coords, class_limit):
        class_id = int(class_id)
        if class_id not in previous_class_set or class_limit <= 0:
            return None

        cls_idx = class_id - 1
        counts = model.prototype_counts[cls_idx].detach()
        valid = counts > 0
        if not bool(valid.any()):
            return None

        prototypes = model.prototypes[cls_idx].detach()[valid].to(device)
        was_training = model.training
        model.eval()
        score_chunks = []
        with torch.no_grad():
            for start in range(0, len(coords), chunk_size):
                coord_chunk = coords[start:start + chunk_size]
                images = patch_tensor_from_coords(hsi_array, coord_chunk, args.hsi_patch_size).to(device)
                _, features, _ = forward_for_classes(
                    model,
                    images,
                    active_class_ids=active_class_ids,
                    return_features=True,
                )
                scores = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).t()
                score_chunks.append(scores.detach().cpu())
        if was_training:
            model.train()

        if not score_chunks:
            return None
        score_matrix = torch.cat(score_chunks, dim=0).numpy()
        keep_count = min(int(class_limit), len(coords))
        if min_keep_ratio < 1.0:
            keep_count = max(1, int(round(keep_count * min_keep_ratio)))
        order = select_replay_indices_from_scores(score_matrix, keep_count, mode=replay_mode, rng=rng)
        return coords[order]

    return sample_filter


def build_spectral_replay_filter(args, model, hsi_array, previous_classes):
    if (
        model is None
        or not bool(getattr(args, "spectral_replay_filter", 0))
        or not previous_classes
        or not hasattr(model, "spectral_prototypes")
    ):
        return None

    previous_class_set = {int(class_id) for class_id in previous_classes}
    min_keep_ratio = float(getattr(args, "spectral_replay_min_keep_ratio", 1.0))
    replay_mode = str(getattr(args, "spectral_replay_mode", "top"))

    def sample_filter(class_id, coords, class_limit):
        class_id = int(class_id)
        if class_id not in previous_class_set or class_limit <= 0:
            return None

        cls_idx = class_id - 1
        counts = model.spectral_prototype_counts[cls_idx].detach().cpu()
        valid = counts > 0
        if not bool(valid.any()):
            return None

        descriptors = spectral_descriptor_from_coords(
            hsi_array,
            coords,
            args.hsi_patch_size,
            mode=args.spectral_descriptor,
        )
        descriptor_tensor = torch.from_numpy(descriptors).float()
        prototypes = model.spectral_prototypes[cls_idx].detach().cpu()[valid]
        scores = F.normalize(descriptor_tensor, dim=1) @ F.normalize(prototypes, dim=1).t()
        keep_count = min(int(class_limit), len(coords))
        if min_keep_ratio < 1.0:
            keep_count = max(1, int(round(keep_count * min_keep_ratio)))

        if replay_mode == "diverse" and prototypes.size(0) > 1:
            score_matrix = scores.numpy()
            assignments = score_matrix.argmax(axis=1)
            max_scores = score_matrix.max(axis=1)
            selected = []
            num_clusters = int(prototypes.size(0))
            base_quota = keep_count // num_clusters
            remainder = keep_count % num_clusters
            for cluster_idx in range(num_clusters):
                cluster_indices = np.where(assignments == cluster_idx)[0]
                if len(cluster_indices) == 0:
                    continue
                quota = base_quota + (1 if cluster_idx < remainder else 0)
                quota = min(quota, len(cluster_indices))
                ranked = cluster_indices[np.argsort(max_scores[cluster_indices])[::-1]]
                selected.extend(ranked[:quota].tolist())

            if len(selected) < keep_count:
                selected_set = set(selected)
                remaining = [idx for idx in np.argsort(max_scores)[::-1].tolist() if idx not in selected_set]
                selected.extend(remaining[: keep_count - len(selected)])
            order = np.asarray(selected[:keep_count], dtype=np.int64)
        else:
            max_scores = scores.max(dim=1).values.numpy()
            order = np.argsort(max_scores)[::-1][:keep_count]
        return coords[order]

    return sample_filter


def build_class_weights(dataset, num_classes, device, enabled=True, gamma=1.0):
    weights = torch.ones(num_classes, dtype=torch.float32)
    if not enabled:
        return weights.to(device)

    counts = dataset.class_counts()
    nonzero_counts = [count for count in counts.values() if count > 0]
    if not nonzero_counts:
        return weights.to(device)

    total = float(sum(nonzero_counts))
    num_seen = float(len(nonzero_counts))
    for class_id, count in counts.items():
        if count > 0:
            weights[int(class_id) - 1] = total / (num_seen * float(count))
    gamma = float(gamma)
    if gamma != 1.0:
        weights = weights.pow(gamma)
    return weights.to(device)


def mask_logits_to_classes(logits, class_ids):
    indices = torch.tensor([int(class_id) - 1 for class_id in class_ids], device=logits.device, dtype=torch.long)
    masked_logits = torch.full_like(logits, -1e9)
    masked_logits.index_copy_(1, indices, logits.index_select(1, indices))
    return masked_logits


def forward_for_classes(model, images, active_class_ids, return_features=False):
    if hasattr(model, "forward_for_session"):
        return model.forward_for_session(images, active_class_ids=active_class_ids, return_features=return_features)
    logits = model(images)
    if return_features:
        return logits, None, logits.new_tensor(0.0)
    return logits, logits.new_tensor(0.0)


def confusion_margin_loss(
    logits,
    labels,
    topk=1,
    margin=0.2,
    target_class_ids=None,
    mode="fixed",
    adaptive_min_rate=0.05,
    adaptive_gamma=1.0,
    adaptive_min_weight=1.0,
):
    """Push each target logit above its most confusing negative logits."""
    if topk <= 0 or margin <= 0:
        return logits.new_tensor(0.0)

    row_mask = torch.ones_like(labels, dtype=torch.bool)
    if target_class_ids is not None:
        target_indices = torch.tensor(
            [int(class_id) - 1 for class_id in target_class_ids],
            device=labels.device,
            dtype=labels.dtype,
        )
        row_mask = (labels.unsqueeze(1) == target_indices.unsqueeze(0)).any(dim=1)
        if not bool(row_mask.any()):
            return logits.new_tensor(0.0)

    valid = torch.isfinite(logits)
    target_logits = logits.gather(1, labels.unsqueeze(1))
    negative_logits = logits.clone()
    negative_logits.scatter_(1, labels.unsqueeze(1), -1e9)
    negative_logits = negative_logits.masked_fill(~valid, -1e9)
    available_negatives = (negative_logits > -1e8).sum(dim=1)
    if not bool((available_negatives > 0).any()):
        return logits.new_tensor(0.0)

    k = min(int(topk), int(available_negatives.max().item()))
    hard_negatives = negative_logits.topk(k=k, dim=1).values
    raw_loss = F.relu(hard_negatives - target_logits + float(margin))
    valid_rows = (available_negatives > 0) & row_mask
    if not bool(valid_rows.any()):
        return logits.new_tensor(0.0)

    if mode == "adaptive_pair":
        hard_indices = negative_logits.topk(k=k, dim=1).indices
        labels_expanded = labels.unsqueeze(1).expand_as(hard_indices)
        valid_pairs = valid_rows.unsqueeze(1) & (hard_negatives > -1e8)
        if not bool(valid_pairs.any()):
            return logits.new_tensor(0.0)

        num_classes = logits.size(1)
        pair_ids = labels_expanded[valid_pairs] * num_classes + hard_indices[valid_pairs]
        pair_counts = torch.bincount(pair_ids, minlength=num_classes * num_classes).float()
        pair_counts = pair_counts.view(num_classes, num_classes)
        target_counts = torch.bincount(labels[valid_rows], minlength=num_classes).float().clamp_min(1.0)

        flat_labels = labels_expanded.reshape(-1)
        flat_hard_indices = hard_indices.reshape(-1)
        flat_pair_rates = pair_counts[flat_labels, flat_hard_indices] / target_counts[flat_labels]
        pair_rates = flat_pair_rates.view_as(hard_indices)
        selected_pairs = valid_pairs & (pair_rates >= float(adaptive_min_rate))
        if not bool(selected_pairs.any()):
            return logits.new_tensor(0.0)

        if float(adaptive_gamma) > 0:
            weights = pair_rates.clamp(0.0, 1.0).pow(float(adaptive_gamma))
        else:
            weights = torch.ones_like(pair_rates)
        min_weight = float(adaptive_min_weight)
        weights = min_weight + (1.0 - min_weight) * weights
        return (raw_loss * weights)[selected_pairs].mean()

    return raw_loss[valid_rows].mean()


def class_prototype_similarities(vectors, prototypes, counts, temperature=0.2, pooling="max"):
    valid_prototypes = counts > 0
    valid_classes = valid_prototypes.any(dim=1)
    if not bool(valid_classes.any()):
        return None, valid_classes

    batch_size = vectors.size(0)
    num_classes = prototypes.size(0)
    num_prototypes = prototypes.size(1)
    flat_prototypes = prototypes.view(num_classes * num_prototypes, -1)
    similarities = F.normalize(vectors, dim=1) @ F.normalize(flat_prototypes, dim=1).t()
    similarities = similarities.view(batch_size, num_classes, num_prototypes)
    similarities = similarities / max(float(temperature), 1e-6)

    mask = valid_prototypes.unsqueeze(0)
    masked_similarities = similarities.masked_fill(~mask, -1e9)
    if pooling == "mean":
        safe_counts = valid_prototypes.sum(dim=1).clamp_min(1).float().unsqueeze(0)
        class_similarities = (similarities * mask.float()).sum(dim=2) / safe_counts
    elif pooling == "logsumexp":
        class_similarities = torch.logsumexp(masked_similarities, dim=2)
    elif pooling == "logmeanexp":
        safe_counts = valid_prototypes.sum(dim=1).clamp_min(1).float().unsqueeze(0)
        class_similarities = torch.logsumexp(masked_similarities, dim=2) - safe_counts.log()
    elif pooling == "softmax":
        weights = torch.softmax(masked_similarities, dim=2)
        class_similarities = (weights * similarities).sum(dim=2)
    else:
        class_similarities = masked_similarities.max(dim=2).values
    return class_similarities, valid_classes


def true_class_prototype_scores(
    model,
    features,
    labels,
    class_ids,
    temperature=0.2,
    pooling="logsumexp",
    calibrated=False,
    score_type="positive",
):
    if features is None or not hasattr(model, "prototypes"):
        return None, None

    indices = torch.tensor(
        [int(class_id) - 1 for class_id in class_ids],
        device=features.device,
        dtype=torch.long,
    )
    if indices.numel() == 0:
        return None, None

    if hasattr(model, "feature_prototypes_for_indices"):
        prototypes = model.feature_prototypes_for_indices(
            indices,
            device=features.device,
            calibrated=bool(calibrated),
        )
    else:
        prototypes = model.prototypes.index_select(0, indices).to(features.device)
    counts = model.prototype_counts.index_select(0, indices).to(features.device)
    class_scores, valid_classes = class_prototype_similarities(
        features,
        prototypes,
        counts,
        temperature=temperature,
        pooling=pooling,
    )
    if class_scores is None:
        return None, None

    label_to_column = torch.full(
        (int(getattr(model, "num_classes", class_scores.size(1))),),
        -1,
        device=features.device,
        dtype=torch.long,
    )
    label_to_column[indices] = torch.arange(indices.numel(), device=features.device)
    valid_labels = (labels >= 0) & (labels < label_to_column.numel())
    columns = torch.full_like(labels, -1)
    columns[valid_labels] = label_to_column[labels[valid_labels]]
    valid = columns >= 0
    if bool(valid.any()):
        valid_columns = torch.zeros_like(valid)
        valid_columns[valid] = valid_classes[columns[valid]]
        valid = valid & valid_columns

    scores = features.new_full((labels.size(0),), float("nan"))
    if bool(valid.any()):
        true_scores = class_scores[valid, columns[valid]]
        if str(score_type).lower() == "margin":
            masked_scores = class_scores.clone()
            masked_scores = masked_scores.masked_fill(~valid_classes.unsqueeze(0), -1e9)
            row_indices = torch.arange(labels.size(0), device=features.device)[valid]
            masked_scores[row_indices, columns[valid]] = -1e9
            negative_scores = masked_scores[valid].max(dim=1).values
            has_negative = negative_scores > -1e8
            true_scores = torch.where(has_negative, true_scores - negative_scores, true_scores)
        scores[valid] = true_scores
    return scores, valid


def one_dimensional_kmeans_threshold(values, iters=20):
    values = values.float().view(-1)
    if values.numel() <= 1:
        return values.mean(), values.mean(), values.mean()

    low_center = torch.quantile(values, 0.25)
    high_center = torch.quantile(values, 0.75)
    if torch.isclose(low_center, high_center):
        low_center = values.min()
        high_center = values.max()
    if torch.isclose(low_center, high_center):
        return values.mean(), values.mean(), values.mean()

    for _ in range(max(1, int(iters))):
        low_distance = (values - low_center).abs()
        high_distance = (values - high_center).abs()
        high_mask = high_distance < low_distance
        if bool((~high_mask).any()):
            low_center = values[~high_mask].mean()
        if bool(high_mask.any()):
            high_center = values[high_mask].mean()

    if low_center > high_center:
        low_center, high_center = high_center, low_center
    threshold = 0.5 * (low_center + high_center)
    return threshold, low_center, high_center


@torch.no_grad()
def build_capst_thresholds(
    model,
    loader,
    device,
    class_ids,
    temperature=0.2,
    pooling="logsumexp",
    mode="quantile",
    score_type="positive",
    quantile=0.2,
    beta=1.0,
    use_calibrated_prototypes=False,
):
    if not hasattr(model, "prototypes"):
        return None, None

    model.eval()
    score_buckets = {int(class_id): [] for class_id in class_ids}
    for images, labels in tqdm(loader, desc="capst", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, features, _ = forward_for_classes(model, images, class_ids, return_features=True)
        scores, valid = true_class_prototype_scores(
            model,
            features,
            labels,
            class_ids,
            temperature=temperature,
            pooling=pooling,
            calibrated=bool(use_calibrated_prototypes),
            score_type=score_type,
        )
        if scores is None or valid is None:
            continue
        for class_id in class_ids:
            cls_idx = int(class_id) - 1
            mask = valid & (labels == cls_idx)
            if bool(mask.any()):
                score_buckets[int(class_id)].append(scores[mask].detach().cpu())

    num_classes = int(getattr(model, "num_classes", len(HOUSTON2013_CLASS_NAMES)))
    thresholds = torch.full((num_classes,), -1e9, dtype=torch.float32, device=device)
    class_stats = {}
    q = min(1.0, max(0.0, float(quantile)))
    for class_id, chunks in score_buckets.items():
        if not chunks:
            continue
        values = torch.cat(chunks).float()
        if values.numel() == 0:
            continue
        if mode == "mean_std":
            threshold = values.mean() - float(beta) * values.std(unbiased=False)
            low_center = values.mean()
            high_center = values.mean()
        elif mode == "kmeans":
            threshold, low_center, high_center = one_dimensional_kmeans_threshold(values)
        else:
            sorted_values = torch.sort(values).values
            rank = int(round((sorted_values.numel() - 1) * q))
            threshold = sorted_values[min(max(rank, 0), sorted_values.numel() - 1)]
            low_center = values.min()
            high_center = values.max()
        thresholds[int(class_id) - 1] = threshold.to(device)
        class_stats[int(class_id)] = {
            "threshold": float(threshold),
            "mean": float(values.mean()),
            "std": float(values.std(unbiased=False)),
            "low_center": float(low_center),
            "high_center": float(high_center),
            "samples": int(values.numel()),
        }

    valid_thresholds = thresholds > -1e8
    if not bool(valid_thresholds.any()):
        return None, None
    values = thresholds[valid_thresholds].detach().cpu()
    stats = {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "classes": int(valid_thresholds.sum().item()),
        "per_class": class_stats,
    }
    return thresholds, stats


def capst_sample_weights(
    model,
    features,
    labels,
    class_ids,
    thresholds,
    temperature=0.05,
    min_weight=0.2,
    prototype_temperature=0.2,
    prototype_pooling="logsumexp",
    use_calibrated_prototypes=False,
    weight_class_ids=None,
    score_type="positive",
):
    if thresholds is None:
        return None
    scores, valid = true_class_prototype_scores(
        model,
        features,
        labels,
        class_ids,
        temperature=prototype_temperature,
        pooling=prototype_pooling,
        calibrated=bool(use_calibrated_prototypes),
        score_type=score_type,
    )
    if scores is None or valid is None:
        return None

    thresholds = thresholds.to(features.device)
    safe_labels = labels.clamp(0, thresholds.numel() - 1)
    class_thresholds = thresholds[safe_labels]
    valid = valid & (class_thresholds > -1e8)
    if weight_class_ids is not None:
        weight_indices = torch.tensor(
            [int(class_id) - 1 for class_id in weight_class_ids],
            device=features.device,
            dtype=torch.long,
        )
        if weight_indices.numel() == 0:
            valid = torch.zeros_like(valid)
        else:
            valid = valid & (labels.unsqueeze(1) == weight_indices.unsqueeze(0)).any(dim=1)
    weights = features.new_ones(labels.size(0))
    if bool(valid.any()):
        raw_weights = torch.sigmoid((scores[valid] - class_thresholds[valid]) / max(float(temperature), 1e-6))
        floor = min(1.0, max(0.0, float(min_weight)))
        weights[valid] = floor + (1.0 - floor) * raw_weights
    return weights


def spectral_consistency_loss(model, images, features, labels, class_ids, temperature=0.2, pooling="max"):
    if (
        features is None
        or not hasattr(model, "prototypes")
        or not hasattr(model, "spectral_prototypes")
    ):
        return images.new_tensor(0.0)

    indices = torch.tensor([int(class_id) - 1 for class_id in class_ids], device=features.device, dtype=torch.long)
    if indices.numel() <= 1:
        return features.new_tensor(0.0)

    feature_prototypes = model.prototypes.index_select(0, indices).to(features.device)
    feature_counts = model.prototype_counts.index_select(0, indices).to(features.device)
    spectral_prototypes = model.spectral_prototypes.index_select(0, indices).to(features.device)
    spectral_counts = model.spectral_prototype_counts.index_select(0, indices).to(features.device)
    valid_classes = (feature_counts > 0).any(dim=1) & (spectral_counts > 0).any(dim=1)
    if int(valid_classes.sum().item()) <= 1:
        return features.new_tensor(0.0)

    valid_label_indices = indices[valid_classes]
    row_mask = (labels.unsqueeze(1) == valid_label_indices.unsqueeze(0)).any(dim=1)
    if not bool(row_mask.any()):
        return features.new_tensor(0.0)

    feature_sims, feature_valid = class_prototype_similarities(
        features,
        feature_prototypes,
        feature_counts,
        temperature=temperature,
        pooling=pooling,
    )
    descriptors = spectral_descriptor_from_images(
        images,
        mode=getattr(model, "spectral_descriptor_mode", "mean_grad"),
    )
    spectral_sims, spectral_valid = class_prototype_similarities(
        descriptors,
        spectral_prototypes,
        spectral_counts,
        temperature=temperature,
        pooling=pooling,
    )
    valid_classes = valid_classes & feature_valid & spectral_valid
    if int(valid_classes.sum().item()) <= 1:
        return features.new_tensor(0.0)

    feature_sims = feature_sims[row_mask][:, valid_classes]
    spectral_sims = spectral_sims[row_mask][:, valid_classes].detach()
    teacher = F.softmax(spectral_sims, dim=1)
    student = F.log_softmax(feature_sims, dim=1)
    return F.kl_div(student, teacher, reduction="batchmean")


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    epoch,
    class_weights,
    train_class_ids,
    prompt_loss_weight,
    confusion_margin_weight=0.0,
    confusion_margin=0.2,
    confusion_topk=1,
    confusion_margin_target_class_ids=None,
    confusion_margin_mode="fixed",
    adaptive_confusion_min_rate=0.05,
    adaptive_confusion_gamma=1.0,
    adaptive_confusion_min_weight=1.0,
    spectral_consistency_weight=0.0,
    spectral_consistency_temperature=0.2,
    spectral_consistency_pooling="max",
    teacher_model=None,
    distill_class_ids=None,
    old_logit_distill_weight=0.0,
    old_logit_distill_temperature=2.0,
    old_logit_distill_scope="old",
    freeze_backbone_stats=False,
):
    model.train()
    if freeze_backbone_stats and hasattr(model, "features"):
        model.features.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        need_features = spectral_consistency_weight > 0
        if need_features:
            raw_logits, features, prompt_loss = forward_for_classes(
                model,
                images,
                train_class_ids,
                return_features=True,
            )
        else:
            raw_logits, prompt_loss = forward_for_classes(model, images, train_class_ids, return_features=False)
            features = None
        logits = mask_logits_to_classes(raw_logits, train_class_ids)
        loss = F.cross_entropy(logits, labels, weight=class_weights)
        loss = loss + prompt_loss_weight * prompt_loss
        if (
            teacher_model is not None
            and float(old_logit_distill_weight) > 0
            and distill_class_ids
        ):
            old_indices = torch.tensor(
                [int(class_id) - 1 for class_id in distill_class_ids],
                device=labels.device,
                dtype=torch.long,
            )
            if old_logit_distill_scope == "all":
                distill_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                distill_mask = (labels.unsqueeze(1) == old_indices.unsqueeze(0)).any(dim=1)
            if bool(distill_mask.any()):
                with torch.no_grad():
                    teacher_logits, _ = forward_for_classes(
                        teacher_model,
                        images,
                        distill_class_ids,
                        return_features=False,
                    )
                temp = max(float(old_logit_distill_temperature), 1e-6)
                student_old = raw_logits[distill_mask].index_select(1, old_indices) / temp
                teacher_old = teacher_logits[distill_mask].index_select(1, old_indices) / temp
                distill_loss = F.kl_div(
                    F.log_softmax(student_old, dim=1),
                    F.softmax(teacher_old, dim=1),
                    reduction="batchmean",
                ) * (temp ** 2)
                loss = loss + float(old_logit_distill_weight) * distill_loss
        if confusion_margin_weight > 0:
            margin_loss = confusion_margin_loss(
                logits,
                labels,
                topk=confusion_topk,
                margin=confusion_margin,
                target_class_ids=confusion_margin_target_class_ids,
                mode=confusion_margin_mode,
                adaptive_min_rate=adaptive_confusion_min_rate,
                adaptive_gamma=adaptive_confusion_gamma,
                adaptive_min_weight=adaptive_confusion_min_weight,
            )
            loss = loss + confusion_margin_weight * margin_loss
        if spectral_consistency_weight > 0:
            consistency_loss = spectral_consistency_loss(
                model,
                images,
                features,
                labels,
                train_class_ids,
                temperature=spectral_consistency_temperature,
                pooling=spectral_consistency_pooling,
            )
            loss = loss + spectral_consistency_weight * consistency_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total += int(labels.size(0))
        progress.set_postfix(loss=total_loss / max(total, 1), acc=total_correct / max(total, 1))
    return total_loss / max(total, 1), total_correct / max(total, 1)


def calibrate_prototype_offsets(
    model,
    loader,
    device,
    class_ids,
    class_weights,
    epochs=20,
    lr=1e-3,
    l2_weight=0.01,
    prototype_alpha=0.7,
    prototype_temperature=0.2,
    prototype_pooling="logsumexp",
    use_prototype_margin_gate=False,
    prototype_margin_gate_threshold=0.2,
    prototype_margin_gate_temperature=0.5,
    prototype_margin_gate_min=0.5,
    gate_l2_weight=0.001,
    capst_thresholds=None,
    capst_temperature=0.05,
    capst_min_weight=0.2,
    capst_use_calibrated_prototypes=False,
    capst_weight_class_ids=None,
    capst_score_type="positive",
    use_role_aware_offset_l2=False,
    old_class_ids=None,
    new_class_ids=None,
    old_l2_weight=0.1,
    new_l2_weight=0.02,
    use_balanced_offset_loss=False,
    lambda_offset_new=0.5,
    lambda_offset_old_stability=0.2,
    offset_stability_temperature=1.0,
):
    if (
        not hasattr(model, "prototype_offsets")
        or model.prototype_offsets is None
        or not model.prototype_offsets_active()
        or int(epochs) <= 0
    ):
        return None

    offset_indices = torch.tensor(
        [int(class_id) - 1 for class_id in class_ids],
        device=model.prototype_offsets.device,
        dtype=torch.long,
    )
    if offset_indices.numel() == 0:
        return None

    role_l2_weights = None
    if bool(use_role_aware_offset_l2):
        role_l2_weights = torch.full(
            (offset_indices.numel(), 1, 1),
            float(l2_weight),
            device=model.prototype_offsets.device,
            dtype=model.prototype_offsets.dtype,
        )
        if old_class_ids:
            old_indices = torch.tensor(
                [int(class_id) - 1 for class_id in old_class_ids],
                device=model.prototype_offsets.device,
                dtype=torch.long,
            )
            old_mask = (offset_indices.unsqueeze(1) == old_indices.unsqueeze(0)).any(dim=1)
            role_l2_weights[old_mask] = float(old_l2_weight)
        if new_class_ids:
            new_indices = torch.tensor(
                [int(class_id) - 1 for class_id in new_class_ids],
                device=model.prototype_offsets.device,
                dtype=torch.long,
            )
            new_mask = (offset_indices.unsqueeze(1) == new_indices.unsqueeze(0)).any(dim=1)
            role_l2_weights[new_mask] = float(new_l2_weight)

    previous_requires_grad = [(param, param.requires_grad) for param in model.parameters()]
    for param in model.parameters():
        param.requires_grad = False
    model.prototype_offsets.requires_grad = True
    optimizer_params = [model.prototype_offsets]
    if getattr(model, "prototype_offset_gates", None) is not None:
        model.prototype_offset_gates.requires_grad = True
        optimizer_params.append(model.prototype_offset_gates)

    optimizer = torch.optim.AdamW(optimizer_params, lr=float(lr), weight_decay=0.0)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    capst_weight_sum = 0.0
    capst_weight_count = 0
    capst_weight_min = None
    capst_weight_max = None

    try:
        for epoch in range(1, int(epochs) + 1):
            total_loss = 0.0
            total_correct = 0
            total = 0
            progress = tqdm(loader, desc=f"offset {epoch}", leave=False)
            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.no_grad():
                    base_logits, features, _ = forward_for_classes(
                        model,
                        images,
                        active_class_ids=class_ids,
                        return_features=True,
                    )
                    teacher_logits = None
                    if (
                        bool(use_balanced_offset_loss)
                        and float(lambda_offset_old_stability) > 0
                        and old_class_ids
                    ):
                        teacher_logits = add_prototype_logits(
                            model,
                            base_logits.detach(),
                            features.detach(),
                            class_ids,
                            alpha=prototype_alpha,
                            temperature=prototype_temperature,
                            pooling=prototype_pooling,
                            calibrated=False,
                            use_margin_gate=use_prototype_margin_gate,
                            margin_gate_threshold=prototype_margin_gate_threshold,
                            margin_gate_temperature=prototype_margin_gate_temperature,
                            margin_gate_min=prototype_margin_gate_min,
                        )
                        teacher_logits = mask_logits_to_classes(teacher_logits, class_ids).detach()
                logits = add_prototype_logits(
                    model,
                    base_logits,
                    features,
                    class_ids,
                    alpha=prototype_alpha,
                    temperature=prototype_temperature,
                    pooling=prototype_pooling,
                    calibrated=True,
                    use_margin_gate=use_prototype_margin_gate,
                    margin_gate_threshold=prototype_margin_gate_threshold,
                    margin_gate_temperature=prototype_margin_gate_temperature,
                    margin_gate_min=prototype_margin_gate_min,
                )
                logits = mask_logits_to_classes(logits, class_ids)
                if capst_thresholds is not None:
                    sample_weights = capst_sample_weights(
                        model,
                        features,
                        labels,
                        class_ids,
                        capst_thresholds,
                        temperature=capst_temperature,
                        min_weight=capst_min_weight,
                        prototype_temperature=prototype_temperature,
                        prototype_pooling=prototype_pooling,
                        use_calibrated_prototypes=capst_use_calibrated_prototypes,
                        weight_class_ids=capst_weight_class_ids,
                        score_type=capst_score_type,
                    )
                    loss_values = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
                    if sample_weights is not None:
                        loss = (loss_values * sample_weights).sum() / sample_weights.sum().clamp_min(1e-6)
                        detached_weights = sample_weights.detach()
                        capst_weight_sum += float(detached_weights.sum().item())
                        capst_weight_count += int(detached_weights.numel())
                        current_min = float(detached_weights.min().item())
                        current_max = float(detached_weights.max().item())
                        capst_weight_min = current_min if capst_weight_min is None else min(capst_weight_min, current_min)
                        capst_weight_max = current_max if capst_weight_max is None else max(capst_weight_max, current_max)
                    else:
                        loss = loss_values.mean()
                else:
                    loss = F.cross_entropy(logits, labels, weight=class_weights)
                if bool(use_balanced_offset_loss):
                    if float(lambda_offset_new) > 0 and new_class_ids:
                        new_indices = torch.tensor(
                            [int(class_id) - 1 for class_id in new_class_ids],
                            device=labels.device,
                            dtype=torch.long,
                        )
                        new_mask = (labels.unsqueeze(1) == new_indices.unsqueeze(0)).any(dim=1)
                        if bool(new_mask.any()):
                            new_loss = F.cross_entropy(
                                logits[new_mask],
                                labels[new_mask],
                                weight=class_weights,
                            )
                            loss = loss + float(lambda_offset_new) * new_loss
                    if (
                        float(lambda_offset_old_stability) > 0
                        and teacher_logits is not None
                        and old_class_ids
                    ):
                        old_indices = torch.tensor(
                            [int(class_id) - 1 for class_id in old_class_ids],
                            device=labels.device,
                            dtype=torch.long,
                        )
                        old_mask = (labels.unsqueeze(1) == old_indices.unsqueeze(0)).any(dim=1)
                        if bool(old_mask.any()):
                            temp = max(float(offset_stability_temperature), 1e-6)
                            student_old = logits[old_mask].index_select(1, old_indices) / temp
                            teacher_old = teacher_logits[old_mask].index_select(1, old_indices) / temp
                            stability_loss = F.kl_div(
                                F.log_softmax(student_old, dim=1),
                                F.softmax(teacher_old, dim=1),
                                reduction="batchmean",
                            ) * (temp ** 2)
                            loss = loss + float(lambda_offset_old_stability) * stability_loss
                if float(l2_weight) > 0:
                    active_offsets = model.prototype_offsets.index_select(0, offset_indices)
                    if role_l2_weights is not None:
                        loss = loss + (role_l2_weights.to(active_offsets.dtype) * active_offsets.pow(2)).mean()
                    else:
                        loss = loss + float(l2_weight) * active_offsets.pow(2).mean()
                if float(gate_l2_weight) > 0 and getattr(model, "prototype_offset_gates", None) is not None:
                    active_gates = model.prototype_offset_gates.index_select(0, offset_indices)
                    gate_center = float(getattr(model, "prototype_offset_gate_init", 0.0))
                    loss = loss + float(gate_l2_weight) * (active_gates - gate_center).pow(2).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item()) * labels.size(0)
                preds = logits.argmax(dim=1)
                total_correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
                progress.set_postfix(loss=total_loss / max(total, 1), acc=total_correct / max(total, 1))
    finally:
        for param, requires_grad in previous_requires_grad:
            param.requires_grad = requires_grad
        if was_training:
            model.train()

    capst_stats = None
    if capst_weight_count > 0:
        capst_stats = {
            "mean": capst_weight_sum / max(capst_weight_count, 1),
            "min": capst_weight_min,
            "max": capst_weight_max,
        }
    return total_loss / max(total, 1), total_correct / max(total, 1), capst_stats


@torch.no_grad()
def compute_feature_prototypes(features, num_prototypes=1, iters=10):
    features = F.normalize(features.detach(), dim=1)
    num_samples, feature_dim = features.shape
    num_active = min(max(1, int(num_prototypes)), num_samples)

    if num_active == 1:
        center = F.normalize(features.mean(dim=0, keepdim=True), dim=1)
        counts = torch.tensor([float(num_samples)], device=features.device)
        return center, counts

    mean_feature = features.mean(dim=0, keepdim=True)
    first_idx = torch.argmin((features - mean_feature).pow(2).sum(dim=1))
    centers = [features[first_idx]]
    min_distances = 1.0 - features @ centers[0]
    for _ in range(1, num_active):
        next_idx = torch.argmax(min_distances)
        centers.append(features[next_idx])
        next_distances = 1.0 - features @ centers[-1]
        min_distances = torch.minimum(min_distances, next_distances)
    centers = torch.stack(centers, dim=0)

    assignments = torch.zeros(num_samples, dtype=torch.long, device=features.device)
    for _ in range(max(1, int(iters))):
        distances = torch.cdist(features, centers)
        assignments = distances.argmin(dim=1)
        new_centers = []
        for proto_idx in range(num_active):
            mask = assignments == proto_idx
            if bool(mask.any()):
                new_centers.append(features[mask].mean(dim=0))
            else:
                new_centers.append(centers[proto_idx])
        centers = F.normalize(torch.stack(new_centers, dim=0), dim=1)

    counts = torch.bincount(assignments, minlength=num_active).float()
    return centers, counts


def feature_dispersion_score(features):
    features = F.normalize(features.detach(), dim=1)
    center = F.normalize(features.mean(dim=0, keepdim=True), dim=1)
    return float((1.0 - features @ center.t()).mean().item())


def prototype_assignment_error(features, centers):
    features = F.normalize(features.detach(), dim=1)
    centers = F.normalize(centers.detach(), dim=1)
    similarities = features @ centers.t()
    return float((1.0 - similarities.max(dim=1).values).mean().item())


def choose_adaptive_num_prototypes(
    features,
    max_prototypes,
    min_prototypes=1,
    dispersion_low=0.08,
    dispersion_high=0.20,
    mode="dispersion",
    compactness_target=0.08,
    elbow_min_gain=0.03,
    kmeans_iters=10,
):
    max_prototypes = max(1, int(max_prototypes))
    min_prototypes = max(1, min(int(min_prototypes), max_prototypes))
    if max_prototypes <= min_prototypes:
        return min(max_prototypes, int(features.size(0)))

    mode = str(mode).lower()
    candidate_max = min(max_prototypes, int(features.size(0)))
    if mode in {"compactness", "elbow"}:
        errors = []
        for candidate_k in range(min_prototypes, candidate_max + 1):
            centers, _ = compute_feature_prototypes(
                features,
                num_prototypes=candidate_k,
                iters=kmeans_iters,
            )
            errors.append((candidate_k, prototype_assignment_error(features, centers)))

        if not errors:
            return 1

        if mode == "compactness":
            target = float(compactness_target)
            for candidate_k, error in errors:
                if error <= target:
                    return candidate_k
            return errors[-1][0]

        previous_error = errors[0][1]
        for candidate_k, error in errors[1:]:
            relative_gain = (previous_error - error) / max(abs(previous_error), 1e-6)
            if relative_gain < float(elbow_min_gain):
                return max(min_prototypes, candidate_k - 1)
            previous_error = error
        return errors[-1][0]

    dispersion = feature_dispersion_score(features)
    low = float(dispersion_low)
    high = max(float(dispersion_high), low + 1e-6)
    ratio = (dispersion - low) / (high - low)
    ratio = min(1.0, max(0.0, ratio))
    adaptive_k = min_prototypes + int(round(ratio * (max_prototypes - min_prototypes)))
    return min(max(1, adaptive_k), int(features.size(0)))


def add_prototype_logits(
    model,
    logits,
    features,
    class_ids,
    alpha=1.0,
    temperature=0.2,
    pooling="max",
    calibrated=True,
    use_margin_gate=False,
    margin_gate_threshold=0.2,
    margin_gate_temperature=0.5,
    margin_gate_min=0.5,
):
    if features is None or not hasattr(model, "prototypes") or alpha <= 0:
        return logits

    indices = torch.tensor([int(class_id) - 1 for class_id in class_ids], device=logits.device, dtype=torch.long)
    if hasattr(model, "feature_prototypes_for_indices"):
        prototypes = model.feature_prototypes_for_indices(
            indices,
            device=logits.device,
            calibrated=bool(calibrated),
        )
    else:
        prototypes = model.prototypes.index_select(0, indices).to(logits.device)
    counts = model.prototype_counts.index_select(0, indices).to(logits.device)

    if prototypes.dim() == 2:
        valid = counts > 0
        if not bool(valid.any()):
            return logits
        valid_indices = indices[valid]
        prototypes = prototypes[valid]
        similarities = F.normalize(features, dim=1) @ F.normalize(prototypes, dim=1).t()
        similarities = similarities / max(float(temperature), 1e-6)
        if bool(use_margin_gate) and similarities.size(1) > 1:
            top2 = similarities.topk(k=2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
            raw_gate = torch.sigmoid(
                (margins - float(margin_gate_threshold)) / max(float(margin_gate_temperature), 1e-6)
            )
            floor = min(1.0, max(0.0, float(margin_gate_min)))
            gates = floor + (1.0 - floor) * raw_gate
            similarities = similarities * gates.unsqueeze(1)
        updated_logits = logits.clone()
        updated_logits[:, valid_indices] = updated_logits[:, valid_indices] + float(alpha) * similarities
        return updated_logits

    valid_prototypes = counts > 0
    valid_classes = valid_prototypes.any(dim=1)
    if not bool(valid_classes.any()):
        return logits

    batch_size = features.size(0)
    num_classes = prototypes.size(0)
    num_prototypes = prototypes.size(1)
    flat_prototypes = prototypes.view(num_classes * num_prototypes, -1)
    similarities = F.normalize(features, dim=1) @ F.normalize(flat_prototypes, dim=1).t()
    similarities = similarities.view(batch_size, num_classes, num_prototypes)
    similarities = similarities / max(float(temperature), 1e-6)

    mask = valid_prototypes.unsqueeze(0)
    masked_similarities = similarities.masked_fill(~mask, -1e9)
    if pooling == "mean":
        safe_counts = valid_prototypes.sum(dim=1).clamp_min(1).float().unsqueeze(0)
        class_similarities = (similarities * mask.float()).sum(dim=2) / safe_counts
    elif pooling == "logsumexp":
        class_similarities = torch.logsumexp(masked_similarities, dim=2)
    elif pooling == "logmeanexp":
        safe_counts = valid_prototypes.sum(dim=1).clamp_min(1).float().unsqueeze(0)
        class_similarities = torch.logsumexp(masked_similarities, dim=2) - safe_counts.log()
    elif pooling == "softmax":
        weights = torch.softmax(masked_similarities, dim=2)
        class_similarities = (weights * similarities).sum(dim=2)
    else:
        class_similarities = masked_similarities.max(dim=2).values

    if bool(use_margin_gate) and int(valid_classes.sum().item()) > 1:
        valid_class_similarities = class_similarities.masked_fill(~valid_classes.unsqueeze(0), -1e9)
        top2 = valid_class_similarities.topk(k=2, dim=1).values
        margins = top2[:, 0] - top2[:, 1]
        raw_gate = torch.sigmoid(
            (margins - float(margin_gate_threshold)) / max(float(margin_gate_temperature), 1e-6)
        )
        floor = min(1.0, max(0.0, float(margin_gate_min)))
        gates = floor + (1.0 - floor) * raw_gate
        class_similarities = class_similarities * gates.unsqueeze(1)

    updated_logits = logits.clone()
    valid_indices = indices[valid_classes]
    updated_logits[:, valid_indices] = updated_logits[:, valid_indices] + float(alpha) * class_similarities[:, valid_classes]
    return updated_logits


def add_spectral_prototype_logits(model, logits, images, class_ids, alpha=0.5, temperature=0.2, pooling="max"):
    if not hasattr(model, "spectral_prototypes") or alpha <= 0:
        return logits

    indices = torch.tensor([int(class_id) - 1 for class_id in class_ids], device=logits.device, dtype=torch.long)
    prototypes = model.spectral_prototypes.index_select(0, indices).to(logits.device)
    counts = model.spectral_prototype_counts.index_select(0, indices).to(logits.device)
    valid_prototypes = counts > 0
    valid_classes = valid_prototypes.any(dim=1)
    if not bool(valid_classes.any()):
        return logits

    descriptors = spectral_descriptor_from_images(
        images,
        mode=getattr(model, "spectral_descriptor_mode", "mean_grad"),
    )
    batch_size = descriptors.size(0)
    num_classes = prototypes.size(0)
    num_prototypes = prototypes.size(1)
    flat_prototypes = prototypes.view(num_classes * num_prototypes, -1)
    similarities = F.normalize(descriptors, dim=1) @ F.normalize(flat_prototypes, dim=1).t()
    similarities = similarities.view(batch_size, num_classes, num_prototypes)
    similarities = similarities / max(float(temperature), 1e-6)

    mask = valid_prototypes.unsqueeze(0)
    masked_similarities = similarities.masked_fill(~mask, -1e9)
    if pooling == "mean":
        safe_counts = valid_prototypes.sum(dim=1).clamp_min(1).float().unsqueeze(0)
        class_similarities = (similarities * mask.float()).sum(dim=2) / safe_counts
    elif pooling == "logsumexp":
        class_similarities = torch.logsumexp(masked_similarities, dim=2)
    elif pooling == "logmeanexp":
        safe_counts = valid_prototypes.sum(dim=1).clamp_min(1).float().unsqueeze(0)
        class_similarities = torch.logsumexp(masked_similarities, dim=2) - safe_counts.log()
    elif pooling == "softmax":
        weights = torch.softmax(masked_similarities, dim=2)
        class_similarities = (weights * similarities).sum(dim=2)
    else:
        class_similarities = masked_similarities.max(dim=2).values

    updated_logits = logits.clone()
    valid_indices = indices[valid_classes]
    updated_logits[:, valid_indices] = updated_logits[:, valid_indices] + float(alpha) * class_similarities[:, valid_classes]
    return updated_logits


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    allowed_class_ids,
    num_classes=15,
    use_prototypes=False,
    prototype_alpha=1.0,
    prototype_temperature=0.2,
    prototype_pooling="max",
    use_prototype_margin_gate=False,
    prototype_margin_gate_threshold=0.2,
    prototype_margin_gate_temperature=0.5,
    prototype_margin_gate_min=0.5,
    use_spectral_prototypes=False,
    spectral_prototype_alpha=0.5,
    spectral_prototype_temperature=0.2,
    spectral_prototype_pooling="max",
):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device, non_blocking=True)
        logits, features, _ = forward_for_classes(model, images, allowed_class_ids, return_features=True)
        if use_prototypes:
            logits = add_prototype_logits(
                model,
                logits,
                features,
                allowed_class_ids,
                alpha=prototype_alpha,
                temperature=prototype_temperature,
                pooling=prototype_pooling,
                use_margin_gate=use_prototype_margin_gate,
                margin_gate_threshold=prototype_margin_gate_threshold,
                margin_gate_temperature=prototype_margin_gate_temperature,
                margin_gate_min=prototype_margin_gate_min,
            )
        if use_spectral_prototypes:
            logits = add_spectral_prototype_logits(
                model,
                logits,
                images,
                allowed_class_ids,
                alpha=spectral_prototype_alpha,
                temperature=spectral_prototype_temperature,
                pooling=spectral_prototype_pooling,
            )
        logits = mask_logits_to_classes(logits, allowed_class_ids)
        preds = logits.argmax(dim=1).cpu().numpy()
        targets = labels.numpy()
        for target, pred in zip(targets, preds):
            if 0 <= target < num_classes and 0 <= pred < num_classes:
                confusion[int(target), int(pred)] += 1
    return confusion


@torch.no_grad()
def refresh_prototypes(
    model,
    loader,
    device,
    class_ids,
    kmeans_iters=10,
    ema_old_classes=None,
    ema_momentum=0.0,
    adaptive_prototypes=False,
    adaptive_min_prototypes=1,
    adaptive_dispersion_low=0.08,
    adaptive_dispersion_high=0.20,
    adaptive_k_mode="dispersion",
    adaptive_compactness_target=0.08,
    adaptive_elbow_min_gain=0.03,
    adaptive_old_classes=None,
    adaptive_old_elbow_min_gain=None,
    adaptive_new_elbow_min_gain=None,
):
    if not hasattr(model, "prototypes"):
        return {}

    model.eval()
    multi_prototype = model.prototypes.dim() == 3
    feature_buckets = {int(class_id): [] for class_id in class_ids}
    ema_old_classes = {int(class_id) for class_id in (ema_old_classes or [])}
    adaptive_old_classes = {int(class_id) for class_id in (adaptive_old_classes or [])}
    ema_momentum = float(ema_momentum)

    for images, labels in tqdm(loader, desc="prototype", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, features, _ = forward_for_classes(model, images, class_ids, return_features=True)
        for class_id in class_ids:
            cls_idx = int(class_id) - 1
            mask = labels == cls_idx
            if bool(mask.any()):
                feature_buckets[int(class_id)].append(features[mask].detach())

    updated = {}
    for class_id in class_ids:
        cls_idx = int(class_id) - 1
        if not feature_buckets[int(class_id)]:
            continue
        class_features = torch.cat(feature_buckets[int(class_id)], dim=0)
        if multi_prototype:
            num_prototypes = model.prototypes.size(1)
            target_num_prototypes = num_prototypes
            if bool(adaptive_prototypes):
                class_elbow_min_gain = adaptive_elbow_min_gain
                if adaptive_k_mode == "elbow" and (
                    adaptive_old_elbow_min_gain is not None or adaptive_new_elbow_min_gain is not None
                ):
                    if int(class_id) in adaptive_old_classes and adaptive_old_elbow_min_gain is not None:
                        class_elbow_min_gain = adaptive_old_elbow_min_gain
                    elif int(class_id) not in adaptive_old_classes and adaptive_new_elbow_min_gain is not None:
                        class_elbow_min_gain = adaptive_new_elbow_min_gain
                target_num_prototypes = choose_adaptive_num_prototypes(
                    class_features,
                    max_prototypes=num_prototypes,
                    min_prototypes=adaptive_min_prototypes,
                    dispersion_low=adaptive_dispersion_low,
                    dispersion_high=adaptive_dispersion_high,
                    mode=adaptive_k_mode,
                    compactness_target=adaptive_compactness_target,
                    elbow_min_gain=class_elbow_min_gain,
                    kmeans_iters=kmeans_iters,
                )
            centers, counts = compute_feature_prototypes(
                class_features,
                num_prototypes=target_num_prototypes,
                iters=kmeans_iters,
            )
            active = centers.size(0)
            if class_id in ema_old_classes and ema_momentum > 0:
                old_centers = model.prototypes[cls_idx, :active].clone()
                old_counts = model.prototype_counts[cls_idx, :active].clone()
                old_valid = old_counts > 0
                model.prototypes[cls_idx].zero_()
                model.prototype_counts[cls_idx].zero_()
                fused_centers = centers.clone()
                if bool(old_valid.any()):
                    fused_centers[old_valid] = F.normalize(
                        ema_momentum * old_centers[old_valid] + (1.0 - ema_momentum) * centers[old_valid],
                        dim=1,
                    )
                fused_counts = torch.maximum(old_counts, counts)
                model.prototypes[cls_idx, :active] = fused_centers
                model.prototype_counts[cls_idx, :active] = fused_counts
            else:
                model.prototypes[cls_idx].zero_()
                model.prototype_counts[cls_idx].zero_()
                model.prototypes[cls_idx, :active] = centers
                model.prototype_counts[cls_idx, :active] = counts
            updated[int(class_id)] = [int(value) for value in counts.detach().cpu().tolist()]
        else:
            center = F.normalize(class_features.mean(dim=0), dim=0)
            count = float(class_features.size(0))
            if class_id in ema_old_classes and ema_momentum > 0 and model.prototype_counts[cls_idx] > 0:
                center = F.normalize(
                    ema_momentum * model.prototypes[cls_idx] + (1.0 - ema_momentum) * center,
                    dim=0,
                )
                count = max(float(model.prototype_counts[cls_idx].item()), count)
            model.prototypes[cls_idx] = center
            model.prototype_counts[cls_idx] = count
            updated[int(class_id)] = int(class_features.size(0))
    return updated


@torch.no_grad()
def refresh_spectral_prototypes(model, loader, device, class_ids, kmeans_iters=10):
    if not hasattr(model, "spectral_prototypes"):
        return {}

    descriptor_buckets = {int(class_id): [] for class_id in class_ids}
    descriptor_mode = getattr(model, "spectral_descriptor_mode", "mean_grad")

    for images, labels in tqdm(loader, desc="spectral prototype", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        descriptors = spectral_descriptor_from_images(images, mode=descriptor_mode)
        for class_id in class_ids:
            cls_idx = int(class_id) - 1
            mask = labels == cls_idx
            if bool(mask.any()):
                descriptor_buckets[int(class_id)].append(descriptors[mask].detach())

    updated = {}
    num_prototypes = model.spectral_prototypes.size(1)
    for class_id in class_ids:
        cls_idx = int(class_id) - 1
        if not descriptor_buckets[int(class_id)]:
            continue
        class_descriptors = torch.cat(descriptor_buckets[int(class_id)], dim=0)
        centers, counts = compute_feature_prototypes(
            class_descriptors,
            num_prototypes=num_prototypes,
            iters=kmeans_iters,
        )
        model.spectral_prototypes[cls_idx].zero_()
        model.spectral_prototype_counts[cls_idx].zero_()
        active = centers.size(0)
        model.spectral_prototypes[cls_idx, :active] = centers
        model.spectral_prototype_counts[cls_idx, :active] = counts
        updated[int(class_id)] = [int(value) for value in counts.detach().cpu().tolist()]
    return updated


@torch.no_grad()
def active_prototype_k_stats(model, class_ids):
    if not hasattr(model, "prototype_counts") or model.prototype_counts.dim() != 2:
        return {
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "by_class": "{}",
        }

    values = {}
    for class_id in class_ids:
        cls_idx = int(class_id) - 1
        if cls_idx < 0 or cls_idx >= model.prototype_counts.size(0):
            continue
        active_k = int((model.prototype_counts[cls_idx] > 0).sum().item())
        values[int(class_id)] = active_k

    if not values:
        return {
            "mean": 0.0,
            "min": 0,
            "max": 0,
            "by_class": "{}",
        }

    counts = list(values.values())
    return {
        "mean": float(sum(counts) / len(counts)),
        "min": int(min(counts)),
        "max": int(max(counts)),
        "by_class": json.dumps(values, ensure_ascii=False),
    }


def configure_trainable_parameters(model, args, session_idx):
    for param in model.parameters():
        param.requires_grad = True

    if args.model == "pdp" and getattr(model, "prototype_offsets", None) is not None:
        model.prototype_offsets.requires_grad = False
    if args.model == "pdp" and getattr(model, "prototype_offset_gates", None) is not None:
        model.prototype_offset_gates.requires_grad = False

    if args.model == "pdp" and getattr(args, "prompt_fusion", "") not in {"tri_pool", "joint_tri_pool"}:
        model.prompt_pool.old_private_keys.requires_grad = False
        model.prompt_pool.old_private_prompts.requires_grad = False
        model.prompt_pool.new_private_keys.requires_grad = False
        model.prompt_pool.new_private_prompts.requires_grad = False
        if getattr(model, "tri_pool_logit_scale", None) is not None:
            model.tri_pool_logit_scale.requires_grad = False

    if session_idx > 1:
        if bool(args.freeze_backbone_after_base):
            for param in model.features.parameters():
                param.requires_grad = False
        if args.model == "pdp" and bool(args.freeze_shared_prompts_after_base):
            model.prompt_pool.shared_keys.requires_grad = False
            model.prompt_pool.shared_prompts.requires_grad = False
            for param in model.prompt_pool.norm.parameters():
                param.requires_grad = False
        if (
            args.model == "pdp"
            and getattr(args, "prompt_fusion", "") in {"tri_pool", "joint_tri_pool"}
            and bool(getattr(args, "freeze_old_prompts_after_promotion", 1))
        ):
            model.prompt_pool.old_private_keys.requires_grad = False
            model.prompt_pool.old_private_prompts.requires_grad = False

    return [param for param in model.parameters() if param.requires_grad]


def print_metrics(title, metrics):
    print(title)
    print(f"  OA    = {metrics['oa']:.4f}")
    print(f"  AA    = {metrics['aa']:.4f}")
    print(f"  Kappa = {metrics['kappa']:.4f}")
    print(f"  Correct / Total = {metrics['correct']} / {metrics['total']}")
    if metrics.get("per_class_acc"):
        per_class_text = ", ".join(
            f"{class_id}:{acc:.4f}" for class_id, acc in sorted(metrics["per_class_acc"].items())
        )
        print(f"  Per-class acc = {per_class_text}")


def print_confusion_details(confusion, class_ids, topk=3):
    print("  Top confusions:")
    for class_id in class_ids:
        row_idx = int(class_id) - 1
        row = confusion[row_idx].astype(np.int64).copy()
        total = int(row.sum())
        if total <= 0:
            continue
        correct = int(row[row_idx])
        row[row_idx] = 0
        mistake_total = int(row.sum())
        if mistake_total <= 0:
            print(f"    class {class_id}: no mistakes ({correct}/{total})")
            continue
        top_indices = np.argsort(row)[::-1][:topk]
        pieces = []
        for pred_idx in top_indices:
            count = int(row[pred_idx])
            if count <= 0:
                continue
            pieces.append(f"to {pred_idx + 1}: {count} ({count / total:.3f})")
        detail = ", ".join(pieces) if pieces else "no dominant target"
        print(f"    class {class_id}: correct {correct}/{total}, {detail}")


def save_metrics_row(csv_path, fieldnames, row):
    exists = csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def get_prompt_param_snapshot(model):
    if not hasattr(model, "prompt_pool"):
        return None
    snapshot = {}
    for name, param in model.prompt_pool.named_parameters():
        if name.startswith("norm."):
            continue
        snapshot[name] = param.detach().clone()
    return snapshot


@torch.no_grad()
def select_topk_percent(tensor, percent):
    percent = float(percent)
    if percent <= 0:
        return torch.zeros_like(tensor, dtype=torch.bool)
    if percent >= 1:
        return torch.ones_like(tensor, dtype=torch.bool)
    threshold = torch.quantile(tensor.flatten(), 1.0 - percent)
    return tensor >= threshold


@torch.no_grad()
def fuse_prompt_param(ori_param, new_param, init_param, topk_percent_ori, topk_percent_new):
    vector_ori_init = ori_param - init_param
    vector_new_ori = new_param - ori_param
    vector_new_init = new_param - init_param

    ori_important = select_topk_percent(vector_ori_init.abs(), topk_percent_ori)
    new_important = select_topk_percent(vector_new_ori.abs(), topk_percent_new)
    replace_mask = new_important & ~ori_important
    not_important = ~(ori_important | new_important)

    ori_direction = torch.sign(vector_ori_init)
    new_direction = torch.sign(vector_new_init)
    same_direction = (ori_direction * new_direction) >= 0
    average_mask = not_important & same_direction

    fused = torch.where(replace_mask, new_param, ori_param)
    fused = torch.where(average_mask, (ori_param + new_param) * 0.5, fused)
    return fused, replace_mask, average_mask


@torch.no_grad()
def fuse_prompt_pool_parameters(model, ori_snapshot, init_snapshot, topk_percent_ori, topk_percent_new):
    if ori_snapshot is None or init_snapshot is None or not hasattr(model, "prompt_pool"):
        return None

    stats = {}
    for name, param in model.prompt_pool.named_parameters():
        if name not in ori_snapshot or name not in init_snapshot:
            continue
        fused, replace_mask, average_mask = fuse_prompt_param(
            ori_snapshot[name].to(param.device),
            param.detach(),
            init_snapshot[name].to(param.device),
            topk_percent_ori,
            topk_percent_new,
        )
        param.copy_(fused)
        stats[name] = {
            "replace": float(replace_mask.float().mean().item()),
            "average": float(average_mask.float().mean().item()),
        }
    return stats


def main(args):
    seed_everything(args.seed)
    resolve_paths(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    hsi_array = load_houston_hsi(
        args.hsi_file,
        hsi_key=args.hsi_key,
        band_indices=args.hsi_band_indices,
        normalize_per_band=bool(args.hsi_normalize_per_band),
    )
    if bool(args.use_lidar):
        lidar_array = load_houston_lidar(
            args.lidar_file,
            lidar_key=args.lidar_key,
            normalize_per_band=bool(args.lidar_normalize_per_band),
        )
        hsi_array = fuse_hsi_lidar(hsi_array, lidar_array)
        print(f"Loaded LiDAR: shape={lidar_array.shape}, fused HSI+LiDAR shape={hsi_array.shape}")
    num_channels = int(hsi_array.shape[2])
    modality = "HSI+LiDAR" if bool(args.use_lidar) else "HSI"
    print(f"Loaded {modality}: shape={hsi_array.shape}, channels={num_channels}")

    sessions = parse_class_sessions(args.sessions)
    num_classes = int(args.num_classes) if int(args.num_classes) > 0 else max(
        class_id for session_classes in sessions for class_id in session_classes
    )
    if args.mode == "all":
        sessions = [list(range(1, num_classes + 1))]
    class_first_session = {}
    for first_session_idx, session_classes in enumerate(sessions, start=1):
        for class_id in session_classes:
            class_first_session[int(class_id)] = first_session_idx

    if args.model == "pdp":
        model = HoustonPDPClassifier(
            num_channels,
            num_classes=num_classes,
            dropout=args.dropout,
            prompt_shared_size=args.prompt_shared_size,
            prompt_scale=args.prompt_scale,
            prompt_temperature=args.prompt_temperature,
            prompt_ortho_weight=args.prompt_ortho_weight,
            num_prototypes_per_class=args.num_prototypes_per_class,
            prompt_fusion=args.prompt_fusion,
            prototype_gate_hidden=args.prototype_gate_hidden,
            prototype_gate_temperature=args.prototype_gate_temperature,
            prototype_gate_alpha=args.prototype_gate_alpha,
            spectral_descriptor=args.spectral_descriptor,
            use_prototype_offsets=args.use_prototype_offsets,
            prototype_offset_scale=args.prototype_offset_scale,
            prototype_offset_start_session=args.prototype_offset_start_session,
            use_role_aware_offsets=args.use_role_aware_offsets,
            prototype_offset_old_scale=args.prototype_offset_old_scale,
            prototype_offset_new_scale=args.prototype_offset_new_scale,
            use_prototype_offset_gates=args.use_prototype_offset_gates,
            prototype_offset_gate_init=args.prototype_offset_gate_init,
            use_prototype_reliability=args.use_prototype_reliability,
            prototype_reliability_power=args.prototype_reliability_power,
            prototype_reliability_min=args.prototype_reliability_min,
            prototype_reliability_max=args.prototype_reliability_max,
            prototype_reliability_normalize=args.prototype_reliability_normalize,
            use_confusion_aware_offsets=args.use_confusion_aware_offsets,
            confusion_offset_strength=args.confusion_offset_strength,
            confusion_offset_min=args.confusion_offset_min,
            confusion_offset_max=args.confusion_offset_max,
            confusion_offset_normalize=args.confusion_offset_normalize,
            tri_pool_similarity=args.tri_pool_similarity,
            tri_pool_alpha=args.tri_pool_alpha,
            tri_pool_routing=args.tri_pool_routing,
            tri_pool_proto_weight=args.tri_pool_proto_weight,
            tri_pool_consistency_weight=args.tri_pool_consistency_weight,
            tri_pool_learnable_scale=args.tri_pool_learnable_scale,
            tri_pool_scale_init=args.tri_pool_scale_init,
            tri_pool_start_session=args.tri_pool_start_session,
            use_spectral_prompt=args.use_spectral_prompt,
            spectral_prompt_alpha=args.spectral_prompt_alpha,
            spectral_prompt_mode=args.spectral_prompt_mode,
            spectral_prompt_start_session=args.spectral_prompt_start_session,
            use_prototype_aware_spectral_prompt=args.use_prototype_aware_spectral_prompt,
            spectral_prompt_gate_threshold=args.spectral_prompt_gate_threshold,
            spectral_prompt_gate_temperature=args.spectral_prompt_gate_temperature,
            spectral_prompt_gate_min=args.spectral_prompt_gate_min,
        ).to(device)
    else:
        model = HoustonPatchCNN(num_channels, num_classes=num_classes, dropout=args.dropout).to(device)
    print(f"Using model: {args.model}")
    initial_prompt_snapshot = None
    if args.model == "pdp" and bool(args.prompt_param_fusion):
        initial_prompt_snapshot = get_prompt_param_snapshot(model)
        print("Using P2IOD-style prompt parameter fusion")

    metrics_csv = output_dir / "metrics.csv"
    fieldnames = [
        "session",
        "train_classes",
        "eval_classes",
        "train_samples",
        "test_samples",
        "model",
        "input_modality",
        "input_channels",
        "use_lidar",
        "lidar_file",
        "lidar_key",
        "class_weight",
        "class_weight_gamma",
        "prompt_fusion",
        "tri_pool_similarity",
        "tri_pool_alpha",
        "tri_pool_routing",
        "tri_pool_proto_weight",
        "tri_pool_consistency_weight",
        "tri_pool_learnable_scale",
        "tri_pool_scale_init",
        "tri_pool_start_session",
        "freeze_old_prompts_after_promotion",
        "prototype_gate_hidden",
        "prototype_gate_temperature",
        "prototype_gate_alpha",
        "prompt_param_fusion",
        "prompt_fuse_topk_ori",
        "prompt_fuse_topk_new",
        "lambda_confusion_margin",
        "confusion_margin",
        "confusion_topk",
        "confusion_margin_scope",
        "confusion_margin_mode",
        "adaptive_confusion_min_rate",
        "adaptive_confusion_gamma",
        "adaptive_confusion_min_weight",
        "lambda_old_logit_distill",
        "old_logit_distill_temperature",
        "old_logit_distill_scope",
        "num_prototypes_per_class",
        "use_prototype_margin_gate",
        "prototype_margin_gate_threshold",
        "prototype_margin_gate_temperature",
        "prototype_margin_gate_min",
        "adaptive_prototypes",
        "adaptive_min_prototypes",
        "adaptive_dispersion_low",
        "adaptive_dispersion_high",
        "adaptive_k_mode",
        "adaptive_compactness_target",
        "adaptive_elbow_min_gain",
        "use_role_aware_adaptive_k",
        "adaptive_old_elbow_min_gain",
        "adaptive_new_elbow_min_gain",
        "adaptive_old_min_age",
        "active_prototype_k_mean",
        "active_prototype_k_min",
        "active_prototype_k_max",
        "active_prototype_k_by_class",
        "prototype_pooling",
        "prototype_ema",
        "prototype_ema_momentum",
        "use_prototype_offsets",
        "prototype_offset_scale",
        "prototype_offset_start_session",
        "use_role_aware_offsets",
        "prototype_offset_old_scale",
        "prototype_offset_new_scale",
        "prototype_offset_epochs",
        "prototype_offset_lr",
        "prototype_offset_l2",
        "use_role_aware_offset_l2",
        "prototype_offset_old_l2",
        "prototype_offset_new_l2",
        "use_balanced_offset_loss",
        "lambda_offset_new",
        "lambda_offset_old_stability",
        "offset_stability_temperature",
        "use_prototype_offset_gates",
        "prototype_offset_gate_init",
        "prototype_offset_gate_l2",
        "use_prototype_reliability",
        "prototype_reliability_power",
        "prototype_reliability_min",
        "prototype_reliability_max",
        "prototype_reliability_normalize",
        "use_confusion_aware_offsets",
        "confusion_offset_strength",
        "confusion_offset_min",
        "confusion_offset_max",
        "confusion_offset_normalize",
        "use_capst",
        "capst_start_session",
        "capst_mode",
        "capst_score_type",
        "capst_quantile",
        "capst_beta",
        "capst_temperature",
        "capst_min_weight",
        "capst_use_calibrated_prototypes",
        "capst_scope",
        "use_spectral_prototypes",
        "spectral_descriptor",
        "use_spectral_prompt",
        "spectral_prompt_alpha",
        "spectral_prompt_mode",
        "spectral_prompt_start_session",
        "use_prototype_aware_spectral_prompt",
        "spectral_prompt_gate_threshold",
        "spectral_prompt_gate_temperature",
        "spectral_prompt_gate_min",
        "spectral_prototype_alpha",
        "spectral_prototype_temperature",
        "spectral_prototype_pooling",
        "lambda_spectral_consistency",
        "spectral_consistency_temperature",
        "spectral_consistency_pooling",
        "spectral_replay_filter",
        "spectral_replay_mode",
        "spectral_replay_min_keep_ratio",
        "feature_replay_filter",
        "feature_replay_mode",
        "feature_replay_min_keep_ratio",
        "feature_replay_chunk_size",
        "oa",
        "aa",
        "kappa",
        "current_oa",
        "current_aa",
        "current_kappa",
        "base_oa",
        "apd_base_oa_raw",
        "apd_base_forgetting",
        "per_class_acc",
    ]

    seen_classes = []
    base_classes = None
    base_oa_at_base_session = None
    all_results = []

    for session_idx, current_classes in enumerate(sessions, start=1):
        previous_classes = list(seen_classes)
        seen_classes = sorted(set(seen_classes + current_classes))
        if base_classes is None:
            base_classes = list(current_classes)

        max_samples_by_class = None
        if args.mode == "all" or args.incremental_train == "cumulative":
            train_classes = seen_classes
        elif args.incremental_train == "replay" and session_idx > 1:
            train_classes = seen_classes
            if args.replay_old_samples_per_class > 0:
                max_samples_by_class = {
                    class_id: args.replay_old_samples_per_class
                    for class_id in previous_classes
                }
        else:
            train_classes = current_classes

        teacher_model = None
        if session_idx > 1 and previous_classes and float(args.lambda_old_logit_distill) > 0:
            teacher_model = copy.deepcopy(model).to(device)
            teacher_model.eval()
            for parameter in teacher_model.parameters():
                parameter.requires_grad_(False)

        promoted_prompt_classes = []
        if args.model == "pdp":
            if args.prompt_fusion in {"tri_pool", "joint_tri_pool"}:
                promoted_prompt_classes = model.promote_new_prompts_to_old(previous_classes)
                if hasattr(model, "initialize_current_prompts_from_joint"):
                    model.initialize_current_prompts_from_joint(current_classes)
            model.set_prompt_task_context(previous_classes, current_classes, session_idx=session_idx)

        print("")
        print(f"Session {session_idx}")
        print(f"  Train classes: {train_classes}")
        print(f"  Eval classes : {seen_classes}")
        if teacher_model is not None:
            print(f"  Old-logit distillation classes: {previous_classes}")
        if args.model == "pdp" and args.prompt_fusion in {"tri_pool", "joint_tri_pool"}:
            print(f"  Tri-pool old classes: {previous_classes}")
            print(f"  Tri-pool new classes: {current_classes}")
            if hasattr(model, "effective_tri_pool_alpha"):
                print(f"  Tri-pool effective alpha: {model.effective_tri_pool_alpha():.6f}")
            if promoted_prompt_classes:
                print(f"  Promoted new prompts to old pool: {promoted_prompt_classes}")
        args.sample_filter = build_feature_replay_filter(
            args,
            model if args.model == "pdp" else None,
            hsi_array,
            previous_classes,
            device,
        )
        if args.sample_filter is not None:
            print("  Using feature-prototype replay sample filtering")
        if args.sample_filter is None:
            args.sample_filter = build_spectral_replay_filter(
                args,
                model if args.model == "pdp" else None,
                hsi_array,
                previous_classes,
            )
            if args.sample_filter is not None:
                print("  Using spectral-guided replay sample filtering")

        train_dataset, train_loader = make_loader(
            args,
            hsi_array,
            args.train_label_file,
            train_classes,
            train=True,
            max_samples_by_class=max_samples_by_class,
        )
        args.sample_filter = None
        test_dataset, test_loader = make_loader(args, hsi_array, args.test_label_file, seen_classes, train=False)
        print(f"  Train samples: {len(train_dataset)}, counts={train_dataset.class_counts()}")
        print(f"  Test samples : {len(test_dataset)}, counts={test_dataset.class_counts()}")

        trainable_params = configure_trainable_parameters(model, args, session_idx)
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
        print(f"  Trainable tensors: {len(trainable_params)}")
        prompt_snapshot_before = None
        if args.model == "pdp" and bool(args.prompt_param_fusion) and session_idx > 1:
            prompt_snapshot_before = get_prompt_param_snapshot(model)

        class_weights = build_class_weights(
            train_dataset,
            num_classes=num_classes,
            device=device,
            enabled=args.class_weight == "balanced",
            gamma=args.class_weight_gamma,
        )
        for epoch in range(1, args.epochs + 1):
            loss, acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                epoch,
                class_weights,
                train_class_ids=train_classes,
                prompt_loss_weight=args.lambda_prompt_ortho,
                confusion_margin_weight=args.lambda_confusion_margin,
                confusion_margin=args.confusion_margin,
                confusion_topk=args.confusion_topk,
                confusion_margin_target_class_ids=(
                    current_classes if args.confusion_margin_scope == "current" else None
                ),
                confusion_margin_mode=args.confusion_margin_mode,
                adaptive_confusion_min_rate=args.adaptive_confusion_min_rate,
                adaptive_confusion_gamma=args.adaptive_confusion_gamma,
                adaptive_confusion_min_weight=args.adaptive_confusion_min_weight,
                spectral_consistency_weight=args.lambda_spectral_consistency,
                spectral_consistency_temperature=args.spectral_consistency_temperature,
                spectral_consistency_pooling=args.spectral_consistency_pooling,
                teacher_model=teacher_model,
                distill_class_ids=previous_classes,
                old_logit_distill_weight=args.lambda_old_logit_distill,
                old_logit_distill_temperature=args.old_logit_distill_temperature,
                old_logit_distill_scope=args.old_logit_distill_scope,
                freeze_backbone_stats=(
                    args.model == "pdp"
                    and session_idx > 1
                    and bool(args.freeze_backbone_after_base)
                ),
            )
            if epoch == 1 or epoch == args.epochs or epoch % 10 == 0:
                print(f"  epoch {epoch:03d}: train_loss={loss:.4f}, train_acc={acc:.4f}")

        if args.model == "pdp" and bool(args.prompt_param_fusion) and session_idx > 1:
            fusion_stats = fuse_prompt_pool_parameters(
                model,
                prompt_snapshot_before,
                initial_prompt_snapshot,
                args.prompt_fuse_topk_ori,
                args.prompt_fuse_topk_new,
            )
            if fusion_stats:
                stat_text = ", ".join(
                    f"{name}:replace={values['replace']:.3f},avg={values['average']:.3f}"
                    for name, values in fusion_stats.items()
                )
                print(f"  Prompt parameter fusion: {stat_text}")

        adaptive_old_classes = None
        if bool(args.use_role_aware_adaptive_k):
            min_age = max(1, int(args.adaptive_old_min_age))
            adaptive_old_classes = [
                class_id
                for class_id in previous_classes
                if session_idx - class_first_session.get(int(class_id), session_idx) >= min_age
            ]

        updated_prototypes = refresh_prototypes(
            model,
            train_loader,
            device,
            train_classes,
            kmeans_iters=args.prototype_kmeans_iters,
            ema_old_classes=previous_classes if bool(args.prototype_ema) else None,
            ema_momentum=args.prototype_ema_momentum,
            adaptive_prototypes=bool(args.adaptive_prototypes),
            adaptive_min_prototypes=args.adaptive_min_prototypes,
            adaptive_dispersion_low=args.adaptive_dispersion_low,
            adaptive_dispersion_high=args.adaptive_dispersion_high,
            adaptive_k_mode=args.adaptive_k_mode,
            adaptive_compactness_target=args.adaptive_compactness_target,
            adaptive_elbow_min_gain=args.adaptive_elbow_min_gain,
            adaptive_old_classes=adaptive_old_classes,
            adaptive_old_elbow_min_gain=args.adaptive_old_elbow_min_gain if bool(args.use_role_aware_adaptive_k) else None,
            adaptive_new_elbow_min_gain=args.adaptive_new_elbow_min_gain if bool(args.use_role_aware_adaptive_k) else None,
        )
        if updated_prototypes:
            print(f"  Updated prototypes: {updated_prototypes}")
        if args.model == "pdp" and bool(args.use_prototype_offsets):
            if hasattr(model, "reset_prototype_offsets"):
                model.reset_prototype_offsets(train_classes)
            if bool(getattr(args, "use_confusion_aware_offsets", 0)) and hasattr(model, "set_confusion_offset_factors"):
                train_confusion_for_offsets = evaluate(
                    model,
                    train_loader,
                    device,
                    allowed_class_ids=train_classes,
                    num_classes=num_classes,
                    use_prototypes=bool(args.use_prototypes),
                    prototype_alpha=args.prototype_alpha,
                    prototype_temperature=args.prototype_temperature,
                    prototype_pooling=args.prototype_pooling,
                    use_prototype_margin_gate=bool(args.use_prototype_margin_gate),
                    prototype_margin_gate_threshold=args.prototype_margin_gate_threshold,
                    prototype_margin_gate_temperature=args.prototype_margin_gate_temperature,
                    prototype_margin_gate_min=args.prototype_margin_gate_min,
                    use_spectral_prototypes=False,
                )
                confusion_factor_stats = model.set_confusion_offset_factors(
                    train_confusion_for_offsets,
                    train_classes,
                )
                if confusion_factor_stats is not None:
                    print(
                        "  Prototype confusion factor: "
                        f"mean={confusion_factor_stats['mean']:.4f}, "
                        f"min={confusion_factor_stats['min']:.4f}, "
                        f"max={confusion_factor_stats['max']:.4f}"
                    )
            capst_thresholds = None
            if bool(args.use_capst) and session_idx >= int(args.capst_start_session):
                capst_thresholds, capst_stats = build_capst_thresholds(
                    model,
                    train_loader,
                    device,
                    train_classes,
                    temperature=args.prototype_temperature,
                    pooling=args.prototype_pooling,
                    mode=args.capst_mode,
                    score_type=args.capst_score_type,
                    quantile=args.capst_quantile,
                    beta=args.capst_beta,
                    use_calibrated_prototypes=bool(args.capst_use_calibrated_prototypes),
                )
                if capst_stats is not None:
                    print(
                        "  CAPST threshold: "
                        f"classes={capst_stats['classes']}, "
                        f"mean={capst_stats['mean']:.4f}, "
                        f"min={capst_stats['min']:.4f}, "
                        f"max={capst_stats['max']:.4f}"
                    )
            capst_weight_class_ids = None
            if bool(args.use_capst):
                if args.capst_scope == "old":
                    capst_weight_class_ids = previous_classes
                elif args.capst_scope == "current":
                    capst_weight_class_ids = current_classes
            offset_stats = calibrate_prototype_offsets(
                model,
                train_loader,
                device,
                train_classes,
                class_weights,
                epochs=args.prototype_offset_epochs,
                lr=args.prototype_offset_lr,
                l2_weight=args.prototype_offset_l2,
                prototype_alpha=args.prototype_alpha,
                prototype_temperature=args.prototype_temperature,
                prototype_pooling=args.prototype_pooling,
                use_prototype_margin_gate=bool(args.use_prototype_margin_gate),
                prototype_margin_gate_threshold=args.prototype_margin_gate_threshold,
                prototype_margin_gate_temperature=args.prototype_margin_gate_temperature,
                prototype_margin_gate_min=args.prototype_margin_gate_min,
                gate_l2_weight=args.prototype_offset_gate_l2,
                capst_thresholds=(
                    capst_thresholds
                    if bool(args.use_capst) and session_idx >= int(args.capst_start_session)
                    else None
                ),
                capst_temperature=args.capst_temperature,
                capst_min_weight=args.capst_min_weight,
                capst_use_calibrated_prototypes=bool(args.capst_use_calibrated_prototypes),
                capst_weight_class_ids=capst_weight_class_ids,
                capst_score_type=args.capst_score_type,
                use_role_aware_offset_l2=bool(args.use_role_aware_offset_l2),
                old_class_ids=previous_classes,
                new_class_ids=current_classes,
                old_l2_weight=args.prototype_offset_old_l2,
                new_l2_weight=args.prototype_offset_new_l2,
                use_balanced_offset_loss=bool(args.use_balanced_offset_loss),
                lambda_offset_new=args.lambda_offset_new,
                lambda_offset_old_stability=args.lambda_offset_old_stability,
                offset_stability_temperature=args.offset_stability_temperature,
            )
            if offset_stats is not None:
                offset_loss, offset_acc = offset_stats[:2]
                offset_capst_stats = offset_stats[2] if len(offset_stats) > 2 else None
                print(f"  Prototype offset calibration: loss={offset_loss:.4f}, acc={offset_acc:.4f}")
                if offset_capst_stats is not None:
                    print(
                        "  CAPST sample weight: "
                        f"mean={offset_capst_stats['mean']:.4f}, "
                        f"min={offset_capst_stats['min']:.4f}, "
                        f"max={offset_capst_stats['max']:.4f}"
                    )
                if hasattr(model, "prototype_offset_gate_stats"):
                    gate_stats = model.prototype_offset_gate_stats(train_classes)
                    if gate_stats is not None:
                        print(
                            "  Prototype offset gate factor: "
                            f"mean={gate_stats['mean']:.4f}, "
                            f"min={gate_stats['min']:.4f}, "
                            f"max={gate_stats['max']:.4f}"
                        )
                if hasattr(model, "prototype_offset_scale_stats"):
                    scale_stats = model.prototype_offset_scale_stats(train_classes)
                    if scale_stats is not None:
                        print(
                            "  Role-aware offset scale: "
                            f"mean={scale_stats['mean']:.4f}, "
                            f"min={scale_stats['min']:.4f}, "
                            f"max={scale_stats['max']:.4f}"
                        )
                if bool(args.use_role_aware_offset_l2):
                    print(
                        "  Role-aware offset L2: "
                        f"old={args.prototype_offset_old_l2:.4f}, "
                        f"new={args.prototype_offset_new_l2:.4f}"
                    )
                if bool(args.use_balanced_offset_loss):
                    print(
                        "  Balanced offset loss: "
                        f"lambda_new={args.lambda_offset_new:.4f}, "
                        f"lambda_old_stability={args.lambda_offset_old_stability:.4f}, "
                        f"T={args.offset_stability_temperature:.4f}"
                    )
                if hasattr(model, "prototype_reliability_stats"):
                    reliability_stats = model.prototype_reliability_stats(train_classes)
                    if reliability_stats is not None:
                        print(
                            "  Prototype reliability factor: "
                            f"mean={reliability_stats['mean']:.4f}, "
                            f"min={reliability_stats['min']:.4f}, "
                            f"max={reliability_stats['max']:.4f}"
                        )
                if hasattr(model, "prototype_confusion_factor_stats"):
                    confusion_factor_stats = model.prototype_confusion_factor_stats(train_classes)
                    if confusion_factor_stats is not None:
                        print(
                            "  Active prototype confusion factor: "
                            f"mean={confusion_factor_stats['mean']:.4f}, "
                            f"min={confusion_factor_stats['min']:.4f}, "
                            f"max={confusion_factor_stats['max']:.4f}"
                        )
        if (
            bool(args.use_spectral_prototypes)
            or args.lambda_spectral_consistency > 0
            or bool(args.spectral_replay_filter)
        ):
            updated_spectral_prototypes = refresh_spectral_prototypes(
                model,
                train_loader,
                device,
                train_classes,
                kmeans_iters=args.prototype_kmeans_iters,
            )
            if updated_spectral_prototypes:
                print(f"  Updated spectral prototypes: {updated_spectral_prototypes}")

        confusion = evaluate(
            model,
            test_loader,
            device,
            allowed_class_ids=seen_classes,
            num_classes=num_classes,
            use_prototypes=bool(args.use_prototypes),
            prototype_alpha=args.prototype_alpha,
            prototype_temperature=args.prototype_temperature,
            prototype_pooling=args.prototype_pooling,
            use_prototype_margin_gate=bool(args.use_prototype_margin_gate),
            prototype_margin_gate_threshold=args.prototype_margin_gate_threshold,
            prototype_margin_gate_temperature=args.prototype_margin_gate_temperature,
            prototype_margin_gate_min=args.prototype_margin_gate_min,
            use_spectral_prototypes=bool(args.use_spectral_prototypes),
            spectral_prototype_alpha=args.spectral_prototype_alpha,
            spectral_prototype_temperature=args.spectral_prototype_temperature,
            spectral_prototype_pooling=args.spectral_prototype_pooling,
        )
        metrics = confusion_to_metrics(confusion, seen_classes)
        base_metrics = confusion_to_metrics(confusion, base_classes)
        current_metrics = confusion_to_metrics(confusion, current_classes)
        if session_idx == 1:
            base_oa_at_base_session = base_metrics["oa"]
        apd_base_oa = base_oa_at_base_session - base_metrics["oa"]
        apd_base_forgetting = max(0.0, apd_base_oa)

        print_metrics(f"Session {session_idx} metrics on seen classes", metrics)
        print_metrics(f"Session {session_idx} metrics on current classes", current_metrics)
        print_confusion_details(confusion, current_classes)
        print(f"  Base-class OA now = {base_metrics['oa']:.4f}")
        print(f"  APD(Base OA raw) = {apd_base_oa:.4f}")
        print(f"  Base forgetting  = {apd_base_forgetting:.4f}")

        active_k_stats = active_prototype_k_stats(model, seen_classes)
        row = {
            "session": session_idx,
            "train_classes": " ".join(map(str, train_classes)),
            "eval_classes": " ".join(map(str, seen_classes)),
            "train_samples": len(train_dataset),
            "test_samples": len(test_dataset),
            "model": args.model,
            "input_modality": "HSI+LiDAR" if bool(args.use_lidar) else "HSI",
            "input_channels": num_channels,
            "use_lidar": args.use_lidar,
            "lidar_file": args.lidar_file,
            "lidar_key": args.lidar_key,
            "class_weight": args.class_weight,
            "class_weight_gamma": args.class_weight_gamma,
            "prompt_fusion": args.prompt_fusion,
            "tri_pool_similarity": args.tri_pool_similarity,
            "tri_pool_alpha": args.tri_pool_alpha,
            "tri_pool_routing": args.tri_pool_routing,
            "tri_pool_proto_weight": args.tri_pool_proto_weight,
            "tri_pool_consistency_weight": args.tri_pool_consistency_weight,
            "tri_pool_learnable_scale": args.tri_pool_learnable_scale,
            "tri_pool_scale_init": args.tri_pool_scale_init,
            "tri_pool_start_session": args.tri_pool_start_session,
            "freeze_old_prompts_after_promotion": args.freeze_old_prompts_after_promotion,
            "prototype_gate_hidden": args.prototype_gate_hidden,
            "prototype_gate_temperature": args.prototype_gate_temperature,
            "prototype_gate_alpha": args.prototype_gate_alpha,
            "prompt_param_fusion": args.prompt_param_fusion,
            "prompt_fuse_topk_ori": args.prompt_fuse_topk_ori,
            "prompt_fuse_topk_new": args.prompt_fuse_topk_new,
            "lambda_confusion_margin": args.lambda_confusion_margin,
            "confusion_margin": args.confusion_margin,
            "confusion_topk": args.confusion_topk,
            "confusion_margin_scope": args.confusion_margin_scope,
            "confusion_margin_mode": args.confusion_margin_mode,
            "adaptive_confusion_min_rate": args.adaptive_confusion_min_rate,
            "adaptive_confusion_gamma": args.adaptive_confusion_gamma,
            "adaptive_confusion_min_weight": args.adaptive_confusion_min_weight,
            "lambda_old_logit_distill": args.lambda_old_logit_distill,
            "old_logit_distill_temperature": args.old_logit_distill_temperature,
            "old_logit_distill_scope": args.old_logit_distill_scope,
            "num_prototypes_per_class": args.num_prototypes_per_class,
            "use_prototype_margin_gate": args.use_prototype_margin_gate,
            "prototype_margin_gate_threshold": args.prototype_margin_gate_threshold,
            "prototype_margin_gate_temperature": args.prototype_margin_gate_temperature,
            "prototype_margin_gate_min": args.prototype_margin_gate_min,
            "adaptive_prototypes": args.adaptive_prototypes,
            "adaptive_min_prototypes": args.adaptive_min_prototypes,
            "adaptive_dispersion_low": args.adaptive_dispersion_low,
            "adaptive_dispersion_high": args.adaptive_dispersion_high,
            "adaptive_k_mode": args.adaptive_k_mode,
            "adaptive_compactness_target": args.adaptive_compactness_target,
            "adaptive_elbow_min_gain": args.adaptive_elbow_min_gain,
            "use_role_aware_adaptive_k": args.use_role_aware_adaptive_k,
            "adaptive_old_elbow_min_gain": args.adaptive_old_elbow_min_gain,
            "adaptive_new_elbow_min_gain": args.adaptive_new_elbow_min_gain,
            "adaptive_old_min_age": args.adaptive_old_min_age,
            "active_prototype_k_mean": active_k_stats["mean"],
            "active_prototype_k_min": active_k_stats["min"],
            "active_prototype_k_max": active_k_stats["max"],
            "active_prototype_k_by_class": active_k_stats["by_class"],
            "prototype_pooling": args.prototype_pooling,
            "prototype_ema": args.prototype_ema,
            "prototype_ema_momentum": args.prototype_ema_momentum,
            "use_prototype_offsets": args.use_prototype_offsets,
            "prototype_offset_scale": args.prototype_offset_scale,
            "prototype_offset_start_session": args.prototype_offset_start_session,
            "use_role_aware_offsets": args.use_role_aware_offsets,
            "prototype_offset_old_scale": args.prototype_offset_old_scale,
            "prototype_offset_new_scale": args.prototype_offset_new_scale,
            "prototype_offset_epochs": args.prototype_offset_epochs,
            "prototype_offset_lr": args.prototype_offset_lr,
            "prototype_offset_l2": args.prototype_offset_l2,
            "use_role_aware_offset_l2": args.use_role_aware_offset_l2,
            "prototype_offset_old_l2": args.prototype_offset_old_l2,
            "prototype_offset_new_l2": args.prototype_offset_new_l2,
            "use_balanced_offset_loss": args.use_balanced_offset_loss,
            "lambda_offset_new": args.lambda_offset_new,
            "lambda_offset_old_stability": args.lambda_offset_old_stability,
            "offset_stability_temperature": args.offset_stability_temperature,
            "use_prototype_offset_gates": args.use_prototype_offset_gates,
            "prototype_offset_gate_init": args.prototype_offset_gate_init,
            "prototype_offset_gate_l2": args.prototype_offset_gate_l2,
            "use_prototype_reliability": args.use_prototype_reliability,
            "prototype_reliability_power": args.prototype_reliability_power,
            "prototype_reliability_min": args.prototype_reliability_min,
            "prototype_reliability_max": args.prototype_reliability_max,
            "prototype_reliability_normalize": args.prototype_reliability_normalize,
            "use_confusion_aware_offsets": args.use_confusion_aware_offsets,
            "confusion_offset_strength": args.confusion_offset_strength,
            "confusion_offset_min": args.confusion_offset_min,
            "confusion_offset_max": args.confusion_offset_max,
            "confusion_offset_normalize": args.confusion_offset_normalize,
            "use_capst": args.use_capst,
            "capst_start_session": args.capst_start_session,
            "capst_mode": args.capst_mode,
            "capst_score_type": args.capst_score_type,
            "capst_quantile": args.capst_quantile,
            "capst_beta": args.capst_beta,
            "capst_temperature": args.capst_temperature,
            "capst_min_weight": args.capst_min_weight,
            "capst_use_calibrated_prototypes": args.capst_use_calibrated_prototypes,
            "capst_scope": args.capst_scope,
            "use_spectral_prototypes": args.use_spectral_prototypes,
            "spectral_descriptor": args.spectral_descriptor,
            "use_spectral_prompt": args.use_spectral_prompt,
            "spectral_prompt_alpha": args.spectral_prompt_alpha,
            "spectral_prompt_mode": args.spectral_prompt_mode,
            "spectral_prompt_start_session": args.spectral_prompt_start_session,
            "use_prototype_aware_spectral_prompt": args.use_prototype_aware_spectral_prompt,
            "spectral_prompt_gate_threshold": args.spectral_prompt_gate_threshold,
            "spectral_prompt_gate_temperature": args.spectral_prompt_gate_temperature,
            "spectral_prompt_gate_min": args.spectral_prompt_gate_min,
            "spectral_prototype_alpha": args.spectral_prototype_alpha,
            "spectral_prototype_temperature": args.spectral_prototype_temperature,
            "spectral_prototype_pooling": args.spectral_prototype_pooling,
            "lambda_spectral_consistency": args.lambda_spectral_consistency,
            "spectral_consistency_temperature": args.spectral_consistency_temperature,
            "spectral_consistency_pooling": args.spectral_consistency_pooling,
            "spectral_replay_filter": args.spectral_replay_filter,
            "spectral_replay_mode": args.spectral_replay_mode,
            "spectral_replay_min_keep_ratio": args.spectral_replay_min_keep_ratio,
            "feature_replay_filter": args.feature_replay_filter,
            "feature_replay_mode": args.feature_replay_mode,
            "feature_replay_min_keep_ratio": args.feature_replay_min_keep_ratio,
            "feature_replay_chunk_size": args.feature_replay_chunk_size,
            "oa": metrics["oa"],
            "aa": metrics["aa"],
            "kappa": metrics["kappa"],
            "current_oa": current_metrics["oa"],
            "current_aa": current_metrics["aa"],
            "current_kappa": current_metrics["kappa"],
            "base_oa": base_metrics["oa"],
            "apd_base_oa_raw": apd_base_oa,
            "apd_base_forgetting": apd_base_forgetting,
            "per_class_acc": json.dumps(metrics["per_class_acc"], sort_keys=True),
        }
        save_metrics_row(metrics_csv, fieldnames, row)
        all_results.append(row)

        ckpt_path = output_dir / f"session_{session_idx}.pth"
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "session": session_idx,
                "seen_classes": seen_classes,
                "metrics": row,
            },
            ckpt_path,
        )
        np.save(output_dir / f"confusion_session_{session_idx}.npy", confusion)

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(all_results, f, indent=2)
    print("")
    print(f"Saved metrics to {metrics_csv}")
    print(f"Saved checkpoints to {output_dir}")


if __name__ == "__main__":
    parser = get_args_parser()
    main(parser.parse_args())
