"""Transformer Lightning model for MALDI mixture identification.

The implementation is adapted to polymicrobial multi-label identification and
to the preprocessing and evaluation utilities in this codebase.
"""

from pathlib import Path
import re

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from bio_attention.attention import TransformerEncoder
from bio_attention.embed import ContinuousEmbedding
from benchmarking.core.metrics import global_prauc, threshold_subset_metrics
from lightning import LightningModule


class TransformerEmbedder(nn.Module):
    """Continuous-value Transformer encoder used for mixture-identification experiments."""

    def __init__(
        self,
        depth: int,
        dim: int,
        n_heads: int = 8,
        dropout: float = 0.2,
        cls: bool = True,
        reduce: str = "cls",
        output_head_dim: int = 256,
    ):
        super().__init__()
        self.embed = ContinuousEmbedding(
            dim,
            depth=1,
            norm=False,
            cls=cls,
            init_cls_as_bias=True,
            init_mask_as_bias=True,
        )
        self.transformer = TransformerEncoder(
            depth=depth,
            dim=dim,
            nh=n_heads,
            attentiontype="vanilla",
            attention_args={"dropout": dropout},
            plugintype="sinusoidal",
            plugin_args={"dim": dim, "divide": 10},
            only_apply_plugin_at_first=True,
            dropout=dropout,
            glu_ff=True,
            activation="gelu",
        )
        self.reduce = str(reduce)
        self.output_head = nn.Linear(dim, output_head_dim)

    def forward(self, spectrum):
        z = self.embed(spectrum["intensity"])
        z = self.transformer(z, pos=spectrum["mz"])
        if self.reduce == "mean":
            return self.output_head(z.sum(1))
        if self.reduce == "max":
            return self.output_head(z.max(1).values)
        if self.reduce == "cls":
            return self.output_head(z[:, 0, :])
        raise ValueError("Transformer reduce must be 'cls', 'mean', or 'max'")


class MixtureTransformer(LightningModule):
    """Rank species presence from a MALDI spectrum using only the Transformer."""

    def __init__(
        self,
        n_species: int,
        transformer: dict,
        head: dict,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        lr_decay_factor: float = 1.0,
        warmup_steps: int = 30,
        dropout: float = 0.2,
        ranking_margin: float = 0.5,
        ranking_max_negatives: int = 20,
        ranking_negative_mode: str = "hard",
        score_temperature: float = 0.5,
        presence_threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.n_species = int(n_species)
        self.register_buffer(
            "selected_presence_threshold",
            torch.tensor(float(presence_threshold), dtype=torch.float32),
            persistent=True,
        )

        if score_temperature <= 0:
            raise ValueError("score_temperature must be greater than zero.")
        self.ranking_negative_mode = str(ranking_negative_mode).lower()
        if self.ranking_negative_mode not in {"hard", "all"}:
            raise ValueError("ranking_negative_mode must be 'hard' or 'all'")

        transformer_kwargs = dict(transformer)
        self.spectrum_embedder = TransformerEmbedder(**transformer_kwargs)
        emb_dim = int(transformer_kwargs["output_head_dim"])

        self.shared = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head_type = str(head.get("type", "prototype")).lower()
        if self.head_type not in {"linear", "prototype"}:
            raise ValueError("head.type must be 'linear' or 'prototype'")
        self.presence_head = nn.Linear(emb_dim, self.n_species)

        self.prototype_similarity = None
        if self.head_type == "prototype":
            proto_cfg = head.get("prototype", {}) or {}
            proto_dim = int(proto_cfg.get("dim") or emb_dim)
            self.prototype_similarity = str(proto_cfg.get("similarity", "dot")).lower()
            if self.prototype_similarity not in {"cosine", "dot"}:
                raise ValueError("head.prototype.similarity must be 'cosine' or 'dot'")
            self.prototype_projector = nn.Sequential(
                nn.LayerNorm(emb_dim),
                nn.Linear(emb_dim, proto_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.LayerNorm(proto_dim),
            )
            self.presence_prototypes = nn.Parameter(torch.empty(self.n_species, proto_dim))
            self.presence_proto_bias = nn.Parameter(torch.zeros(self.n_species))
            init_scale = float(proto_cfg.get("logit_scale_init", 0.5))
            self.prototype_logit_scale = nn.Parameter(torch.tensor(init_scale).log())
            nn.init.normal_(self.presence_prototypes, std=1.0 / proto_dim**0.5)

        self._validation_truth = None
        self._validation_scores = None
        self._validation_truth_by_dataset = None
        self._validation_scores_by_dataset = None
        self._validation_loss_sum = None
        self._validation_loss_n = None
        self._validation_loss_sums_by_dataset = None
        self._validation_loss_ns_by_dataset = None

    def trainable_parameter_summary(self):
        total = sum(param.numel() for param in self.parameters())
        trainable = sum(param.numel() for param in self.parameters() if param.requires_grad)
        return {
            "total": int(total),
            "trainable": int(trainable),
            "frozen": int(total - trainable),
            "trainable_percent": float(100.0 * trainable / total if total else 0.0),
        }

    def encode(self, batch):
        emb_batch = {"mz": batch["mz"], "intensity": batch["intensity"]}
        return self.shared(self.spectrum_embedder(emb_batch))

    def forward(self, batch):
        z = self.encode(batch)
        if self.head_type == "linear":
            return self.presence_head(z)

        proto_z = self.prototype_projector(z)
        prototypes = self.presence_prototypes
        if self.prototype_similarity == "cosine":
            proto_z = F.normalize(proto_z, dim=-1)
            prototypes = F.normalize(prototypes, dim=-1)
        scale = self.prototype_logit_scale.exp().clamp(0.01, 100.0)
        return scale * (proto_z @ prototypes.T) + self.presence_proto_bias

    def _safe_normalize(self, x, eps=1e-8):
        return x / x.sum(dim=-1, keepdim=True).clamp_min(eps)

    def _abundance_distribution(self, score_logits, mask=None):
        if mask is not None:
            very_neg = torch.finfo(score_logits.dtype).min
            masked = score_logits.masked_fill(~mask.bool(), very_neg)
            probs = F.softmax(masked, dim=-1)
            empty_rows = mask.sum(dim=-1) == 0
            if empty_rows.any():
                fallback_probs = F.softmax(score_logits, dim=-1)
                probs = torch.where(empty_rows.unsqueeze(-1), fallback_probs, probs)
            return probs
        return F.softmax(score_logits, dim=-1)

    def _prediction_threshold(self):
        threshold = getattr(self, "selected_presence_threshold", None)
        return 0.5 if threshold is None else float(threshold.detach().cpu())

    def _calibrated_presence_logits(self, presence_logits):
        threshold_prob = self.selected_presence_threshold.to(
            device=presence_logits.device,
            dtype=presence_logits.dtype,
        ).clamp(1e-6, 1.0 - 1e-6)
        threshold_logit = torch.logit(threshold_prob)
        temperature = max(float(getattr(self.hparams, "score_temperature", 1.0)), 1e-6)
        return (presence_logits - threshold_logit) / temperature

    def _presence_probabilities(self, presence_logits, calibrated=True):
        if calibrated:
            presence_logits = self._calibrated_presence_logits(presence_logits)
        return torch.sigmoid(presence_logits)

    def _ranking_loss(self, presence_logits, y_presence):
        margin = float(getattr(self.hparams, "ranking_margin", 0.5))
        max_negatives = int(getattr(self.hparams, "ranking_max_negatives", 20))
        negative_mode = str(
            getattr(self.hparams, "ranking_negative_mode", self.ranking_negative_mode)
        ).lower()
        losses = []
        for logits_row, target_row in zip(presence_logits, y_presence):
            pos_logits = logits_row[target_row > 0]
            neg_logits = logits_row[target_row <= 0]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            if negative_mode == "hard" and max_negatives > 0 and neg_logits.numel() > max_negatives:
                neg_logits = neg_logits.topk(max_negatives).values
            elif negative_mode != "all" and negative_mode != "hard":
                raise ValueError("ranking_negative_mode must be 'hard' or 'all'")
            pair_losses = F.relu(margin - pos_logits.unsqueeze(1) + neg_logits.unsqueeze(0))
            losses.append(pair_losses.mean())
        if not losses:
            return presence_logits.new_zeros(())
        return torch.stack(losses).mean()

    def _compute_loss_and_proportions(self, batch, presence_logits):
        y_presence = batch["multi_species"].to(presence_logits).float()
        ranking_loss = self._ranking_loss(presence_logits, y_presence)
        abundance_mask = self._presence_probabilities(presence_logits, calibrated=True) > 0.5
        pred_prop = self._abundance_distribution(presence_logits, mask=abundance_mask)
        return ranking_loss, pred_prop

    def _compute_set_metrics_from_logits(self, presence_logits, true_multi_hot, threshold=None):
        # Torch equivalent of core.metrics.threshold_subset_metrics, kept on-device
        # for training/validation step logging.
        if threshold is None:
            presence_probs = self._presence_probabilities(presence_logits, calibrated=True)
            pred_bin = (presence_probs > 0.5).int()
        else:
            raw_probs = self._presence_probabilities(presence_logits, calibrated=False)
            pred_bin = (raw_probs > threshold).int()

        true_bin = true_multi_hot.int()
        tp = ((pred_bin == 1) & (true_bin == 1)).sum().float()
        fp = ((pred_bin == 1) & (true_bin == 0)).sum().float()
        fn = ((pred_bin == 0) & (true_bin == 1)).sum().float()
        precision_denom = tp + fp
        recall_denom = tp + fn
        nan = torch.tensor(float("nan"), device=presence_logits.device, dtype=torch.float32)
        precision = torch.where(precision_denom > 0, tp / precision_denom, nan)
        recall = torch.where(recall_denom > 0, tp / recall_denom, nan)
        exact_match = (pred_bin == true_bin).all(dim=1).float()
        return {
            "precision": precision,
            "recall": recall,
            "exact_match": exact_match.mean(),
        }

    def on_validation_epoch_start(self):
        self._validation_truth = []
        self._validation_scores = []
        self._validation_truth_by_dataset = {}
        self._validation_scores_by_dataset = {}
        self._validation_loss_sum = {"loss": torch.zeros((), device=self.device)}
        self._validation_loss_n = torch.zeros((), device=self.device)
        self._validation_loss_sums_by_dataset = {}
        self._validation_loss_ns_by_dataset = {}

    @torch.no_grad()
    def _update_validation_score_buffers(self, presence_logits, true_multi_hot, dataset_key=None):
        if self._validation_truth is None or self._validation_scores is None:
            return
        scores = torch.sigmoid(presence_logits.detach()).float().cpu()
        truth = true_multi_hot.detach().bool().cpu()
        if self._validation_dataset_is_global(dataset_key):
            self._validation_scores.append(scores)
            self._validation_truth.append(truth)
        if dataset_key is None:
            return
        key = int(dataset_key)
        self._validation_scores_by_dataset.setdefault(key, []).append(scores)
        self._validation_truth_by_dataset.setdefault(key, []).append(truth)

    @torch.no_grad()
    def _update_validation_loss_buffer(self, loss, batch_size, dataset_key=None):
        batch_n = int(batch_size)
        weighted = loss.detach().float() * batch_n
        if self._validation_dataset_is_global(dataset_key):
            self._validation_loss_sum["loss"] = self._validation_loss_sum["loss"] + weighted
            self._validation_loss_n = self._validation_loss_n + self._validation_loss_n.new_tensor(float(batch_n))
        if dataset_key is None:
            return
        key = int(dataset_key)
        self._validation_loss_sums_by_dataset.setdefault(
            key, {"loss": torch.zeros((), device=loss.device)}
        )
        self._validation_loss_ns_by_dataset.setdefault(key, self._validation_loss_n.new_zeros(()))
        self._validation_loss_sums_by_dataset[key]["loss"] += weighted
        self._validation_loss_ns_by_dataset[key] += self._validation_loss_n.new_tensor(float(batch_n))

    def _validation_dataset_is_global(self, dataset_key):
        if dataset_key is None:
            return True
        global_count = getattr(getattr(self.trainer, "datamodule", None), "val_global_h5_count", None)
        return global_count is None or int(dataset_key) < int(global_count)

    def _validation_epoch_losses(self, loss_sum=None, loss_n=None):
        if loss_sum is None:
            loss_sum = self._validation_loss_sum
        if loss_n is None:
            loss_n = self._validation_loss_n
        if not loss_sum or loss_n is None:
            return None
        if self.trainer.world_size > 1:
            loss_sum = {name: self.all_gather(value).sum() for name, value in loss_sum.items()}
            loss_n = self.all_gather(loss_n).sum()
        if loss_n <= 0:
            return None
        return {name: value / loss_n.clamp_min(1.0) for name, value in loss_sum.items()}

    def _gather_numpy_validation_buffers(self, truth_parts, score_parts):
        if not truth_parts or not score_parts:
            return None, None
        truth = torch.cat(truth_parts, dim=0).numpy()
        scores = torch.cat(score_parts, dim=0).numpy()
        world_size = int(getattr(getattr(self, "_trainer", None), "world_size", 1) or 1)
        if world_size > 1 and dist.is_available() and dist.is_initialized():
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, (truth, scores))
            truth = np.concatenate([item[0] for item in gathered if item is not None], axis=0)
            scores = np.concatenate([item[1] for item in gathered if item is not None], axis=0)
        return truth, scores

    @staticmethod
    def _benchmark_global_prauc(truth, scores):
        if truth is None or scores is None:
            return float("nan")
        return global_prauc(truth, scores)

    @staticmethod
    def _set_metrics_at_threshold(truth, scores, threshold):
        pred = np.asarray(scores > threshold, dtype=bool)
        return threshold_subset_metrics(truth, pred)

    def _best_exact_match_threshold_metrics(self, truth_parts, score_parts):
        truth, scores = self._gather_numpy_validation_buffers(truth_parts, score_parts)
        if truth is None or scores is None:
            return None
        scores = np.asarray(scores, dtype=float)
        truth = np.asarray(truth, dtype=bool)
        finite = np.isfinite(scores)
        if not finite.any():
            return None
        safe_scores = np.where(finite, scores, -np.inf)
        candidates = np.unique(safe_scores[finite])
        candidates = np.unique(np.clip(np.concatenate(([0.0], candidates, [1.0])), 0.0, 1.0))
        reference_threshold = self._prediction_threshold()
        best_threshold = float(reference_threshold)
        best_metrics = self._set_metrics_at_threshold(truth, safe_scores, best_threshold)
        best_key = (
            best_metrics["exact_match"],
            best_metrics["precision"] + best_metrics["recall"],
            -abs(best_threshold - reference_threshold),
        )
        for threshold in candidates:
            threshold = float(threshold)
            metrics = self._set_metrics_at_threshold(truth, safe_scores, threshold)
            key = (
                metrics["exact_match"],
                metrics["precision"] + metrics["recall"],
                -abs(threshold - reference_threshold),
            )
            if key > best_key:
                best_threshold = threshold
                best_metrics = metrics
                best_key = key
        return {"threshold": best_threshold, **best_metrics}

    @staticmethod
    def _metric_slug(value):
        return re.sub(r"[^0-9A-Za-z]+", "_", str(value).strip().lower()).strip("_") or "unknown"

    def _validation_dataset_metric_prefix(self, dataset_key):
        names = getattr(getattr(self.trainer, "datamodule", None), "val_dataset_names", None)
        if not names or int(dataset_key) >= len(names):
            return None
        return f"val_{self._metric_slug(Path(str(names[int(dataset_key)])).stem)}"

    def _validation_pr_auc_tensor(self, truth_parts, score_parts):
        truth_np, scores_np = self._gather_numpy_validation_buffers(truth_parts, score_parts)
        return torch.tensor(
            self._benchmark_global_prauc(truth_np, scores_np),
            device=self.device,
            dtype=torch.float32,
        )

    def on_validation_epoch_end(self):
        pr_auc = self._validation_pr_auc_tensor(self._validation_truth, self._validation_scores)
        self.log("val_pr_auc", pr_auc, prog_bar=True, sync_dist=True)
        best_threshold_metrics = self._best_exact_match_threshold_metrics(
            self._validation_truth,
            self._validation_scores,
        )
        if best_threshold_metrics is not None:
            self.selected_presence_threshold.fill_(float(best_threshold_metrics["threshold"]))
            for name in ("threshold", "exact_match", "precision", "recall"):
                metric_name = "val_best_threshold" if name == "threshold" else f"val_best_{name}"
                self.log(
                    metric_name,
                    torch.tensor(best_threshold_metrics[name], device=self.device, dtype=torch.float32),
                    prog_bar=name == "exact_match",
                    sync_dist=True,
                )
        val_losses = self._validation_epoch_losses()
        if val_losses:
            self.log("val_loss", val_losses["loss"], prog_bar=True, sync_dist=True)
            self.log("val_ranking_loss", val_losses["loss"], sync_dist=True)
            self.log("val_pr_auc_loss_tiebreak", pr_auc * 1_000_000.0 - val_losses["loss"], logger=False, sync_dist=True)
            self.log("val_loss_pr_auc_tiebreak", -val_losses["loss"] * 1_000_000.0 + pr_auc, logger=False, sync_dist=True)

        if getattr(getattr(self.trainer, "datamodule", None), "val_dataset_names", None):
            for dataset_key in sorted(self._validation_scores_by_dataset or {}):
                prefix = self._validation_dataset_metric_prefix(dataset_key)
                if prefix is None:
                    continue
                truth_np, scores_np = self._gather_numpy_validation_buffers(
                    self._validation_truth_by_dataset.get(dataset_key, []),
                    self._validation_scores_by_dataset.get(dataset_key, []),
                )
                self.log(
                    f"{prefix}_pr_auc",
                    torch.tensor(self._benchmark_global_prauc(truth_np, scores_np), device=self.device, dtype=torch.float32),
                    sync_dist=True,
                )
                if truth_np is not None and scores_np is not None:
                    metrics = self._set_metrics_at_threshold(truth_np, scores_np, self._prediction_threshold())
                    for name in ("precision", "recall", "f1", "exact_match"):
                        value = metrics.get(name)
                        if value is None:
                            continue
                        self.log(f"{prefix}_{name}", torch.tensor(value, device=self.device, dtype=torch.float32), sync_dist=True)
                dataset_losses = self._validation_epoch_losses(
                    self._validation_loss_sums_by_dataset.get(dataset_key),
                    self._validation_loss_ns_by_dataset.get(dataset_key),
                )
                if dataset_losses:
                    self.log(f"{prefix}_loss", dataset_losses["loss"], sync_dist=True)

        self._validation_truth = None
        self._validation_scores = None
        self._validation_truth_by_dataset = None
        self._validation_scores_by_dataset = None
        self._validation_loss_sum = None
        self._validation_loss_n = None
        self._validation_loss_sums_by_dataset = None
        self._validation_loss_ns_by_dataset = None

    def training_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        presence_logits = self(batch)
        loss, _ = self._compute_loss_and_proportions(batch, presence_logits)
        self.log("train_loss", loss, prog_bar=True, batch_size=len(batch["mz"]))
        self.log("train_ranking_loss", loss, batch_size=len(batch["mz"]))
        return loss

    def validation_step(self, batch, batch_idx, *extra_args):
        dataset_key = int(extra_args[0]) if extra_args else 0
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        presence_logits = self(batch)
        loss, _ = self._compute_loss_and_proportions(batch, presence_logits)
        y_presence = batch["multi_species"].to(presence_logits)
        self._update_validation_score_buffers(presence_logits, y_presence, dataset_key=dataset_key)
        self._update_validation_loss_buffer(loss, len(batch["mz"]), dataset_key=dataset_key)

    def test_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        presence_logits = self(batch)
        loss, pred_prop = self._compute_loss_and_proportions(batch, presence_logits)
        y_presence = batch["multi_species"].to(presence_logits)
        y_prop = batch["proportions"].to(pred_prop)
        set_metrics = self._compute_set_metrics_from_logits(presence_logits, y_presence)
        self.log("test_loss", loss, batch_size=len(batch["mz"]))
        self.log("test_ranking_loss", loss, batch_size=len(batch["mz"]))
        self.log("test_selected_threshold", self.selected_presence_threshold.to(self.device), batch_size=len(batch["mz"]))
        self.log("test_precision", set_metrics["precision"], batch_size=len(batch["mz"]))
        self.log("test_recall", set_metrics["recall"], batch_size=len(batch["mz"]))
        self.log("test_exact_match", set_metrics["exact_match"], batch_size=len(batch["mz"]))
        self.log("test_abundance_mae", (pred_prop - y_prop).abs().mean(), batch_size=len(batch["mz"]))

    def predict_step(self, batch, batch_idx):
        batch["intensity"] = batch["intensity"].to(self.dtype)
        batch["mz"] = batch["mz"].to(self.dtype)
        presence_logits = self(batch)
        raw_presence_probs = self._presence_probabilities(presence_logits, calibrated=False)
        presence_probs = self._presence_probabilities(presence_logits, calibrated=True)
        presence_pred = (presence_probs > 0.5).long()
        pred_prop = self._abundance_distribution(presence_logits, mask=presence_pred.bool())
        return {
            "dataset_index": batch["dataset_index"].detach().cpu(),
            "presence_logits": presence_logits.detach().cpu(),
            "presence_probs": presence_probs.detach().cpu(),
            "presence_probs_raw": raw_presence_probs.detach().cpu(),
            "presence_pred": presence_pred.detach().cpu(),
            "proportion_logits": presence_logits.detach().cpu(),
            "proportions_pred": pred_prop.detach().cpu(),
            "true_presence": batch["multi_species"].detach().cpu(),
            "true_proportions": batch["proportions"].detach().cpu(),
            "num_species": batch["num_species"].detach().cpu(),
            "avg_max_similarity": batch["avg_max_similarity"].detach().cpu(),
        }

    def configure_optimizers(self):
        trainable_params = [param for param in self.parameters() if param.requires_grad]
        if not trainable_params:
            raise RuntimeError("No trainable parameters are available for optimization")
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(
            optimizer,
            lr_lambda=lambda epoch: self.hparams.lr_decay_factor,
        )
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        if self.trainer.global_step < self.hparams.warmup_steps:
            lr_scale = min(1.0, (self.trainer.global_step + 1) / self.hparams.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr_scale * self.hparams.lr
        optimizer.step(closure=optimizer_closure)
