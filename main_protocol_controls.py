"""Separate supervised CE competition from sample/prompt classes.

The original engine is preserved. This entry point replaces only its epoch
function and extends its parser, leaving data loading/evaluation unchanged.
The explicit epoch body mirrors the original; CE is the only changed loss term.
"""
import functools
import torch
import torch.nn.functional as F
from tqdm import tqdm
import main_houston_cls as engine

ORIGINAL_PARSER = engine.get_args_parser


def get_args_parser():
    parser = ORIGINAL_PARSER()
    parser.add_argument("--supervised_class_scope", choices=("sampled", "seen"), default="sampled")
    return parser


def supervised_classes(sample_classes, previous_classes, scope):
    if scope == "sampled":
        return list(sample_classes)
    if scope == "seen":
        return sorted(set(sample_classes) | set(previous_classes or []))
    raise ValueError(f"Unknown supervised scope: {scope}")


def train_one_epoch(model, loader, optimizer, device, epoch, class_weights,
                    train_class_ids, prompt_loss_weight, confusion_margin_weight=0.0,
                    confusion_margin=0.2, confusion_topk=1,
                    confusion_margin_target_class_ids=None, confusion_margin_mode="fixed",
                    adaptive_confusion_min_rate=0.05, adaptive_confusion_gamma=1.0,
                    adaptive_confusion_min_weight=1.0, spectral_consistency_weight=0.0,
                    spectral_consistency_temperature=0.2, spectral_consistency_pooling="max",
                    teacher_model=None, distill_class_ids=None, old_logit_distill_weight=0.0,
                    old_logit_distill_temperature=2.0, old_logit_distill_scope="old",
                    freeze_backbone_stats=False, supervised_class_scope="sampled"):
    model.train()
    if freeze_backbone_stats and hasattr(model, "features"):
        model.features.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    ce_ids = supervised_classes(train_class_ids, distill_class_ids, supervised_class_scope)
    progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
    for images, labels in progress:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if spectral_consistency_weight > 0:
            raw_logits, features, prompt_loss = engine.forward_for_classes(
                model, images, train_class_ids, return_features=True)
        else:
            raw_logits, prompt_loss = engine.forward_for_classes(
                model, images, train_class_ids, return_features=False)
            features = None
        # Original mask stays attached to margin loss and training accuracy.
        # The separately masked CE does not change prompt context or data access.
        logits = engine.mask_logits_to_classes(raw_logits, train_class_ids)
        ce_logits = engine.mask_logits_to_classes(raw_logits, ce_ids)
        loss = F.cross_entropy(ce_logits, labels, weight=class_weights)
        loss = loss + prompt_loss_weight * prompt_loss
        if teacher_model is not None and float(old_logit_distill_weight) > 0 and distill_class_ids:
            old_indices = torch.tensor([int(c)-1 for c in distill_class_ids],
                                       device=labels.device, dtype=torch.long)
            if old_logit_distill_scope == "all":
                distill_mask = torch.ones_like(labels, dtype=torch.bool)
            else:
                distill_mask = (labels.unsqueeze(1) == old_indices.unsqueeze(0)).any(dim=1)
            if bool(distill_mask.any()):
                with torch.no_grad():
                    teacher_logits, _ = engine.forward_for_classes(
                        teacher_model, images, distill_class_ids, return_features=False)
                temp = max(float(old_logit_distill_temperature), 1e-6)
                student_old = raw_logits[distill_mask].index_select(1, old_indices)/temp
                teacher_old = teacher_logits[distill_mask].index_select(1, old_indices)/temp
                loss = loss + float(old_logit_distill_weight) * F.kl_div(
                    F.log_softmax(student_old, dim=1), F.softmax(teacher_old, dim=1),
                    reduction="batchmean") * temp**2
        if confusion_margin_weight > 0:
            loss = loss + confusion_margin_weight * engine.confusion_margin_loss(
                logits, labels, topk=confusion_topk, margin=confusion_margin,
                target_class_ids=confusion_margin_target_class_ids, mode=confusion_margin_mode,
                adaptive_min_rate=adaptive_confusion_min_rate, adaptive_gamma=adaptive_confusion_gamma,
                adaptive_min_weight=adaptive_confusion_min_weight)
        if spectral_consistency_weight > 0:
            loss = loss + spectral_consistency_weight * engine.spectral_consistency_loss(
                model, images, features, labels, train_class_ids,
                temperature=spectral_consistency_temperature, pooling=spectral_consistency_pooling)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.size(0))
        progress.set_postfix(loss=total_loss/max(total, 1), acc=total_correct/max(total, 1))
    return total_loss/max(total, 1), total_correct/max(total, 1)


def main():
    args = get_args_parser().parse_args()
    engine.train_one_epoch = functools.partial(
        train_one_epoch, supervised_class_scope=args.supervised_class_scope)
    print(f"Protocol control: CE={args.supervised_class_scope}; sample/prompt/margin classes unchanged", flush=True)
    engine.main(args)


if __name__ == "__main__":
    main()
