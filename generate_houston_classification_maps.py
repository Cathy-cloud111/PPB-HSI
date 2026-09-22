"""Generate audited Houston2013 session-wise classification maps.

The script reconstructs predictions from saved checkpoints and refuses to draw
the figure unless every reconstructed confusion matrix exactly matches the
matrix saved during the corresponding formal run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

import main_houston_cls as engine
import main_persistent_controls as persistent
from datasets.houston_cls import HoustonPatchClassification, load_mat_array, parse_class_sessions


CLASS_NAMES = [
    "Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil",
    "Water", "Residential", "Commercial", "Road", "Highway", "Railway",
    "Parking lot 1", "Parking lot 2", "Tennis court", "Running track",
]

# High-contrast, color-blind-conscious palette; class 0 is black background.
COLORS = np.asarray([
    (0, 0, 0),
    (230, 25, 75), (60, 180, 75), (0, 70, 200), (245, 220, 35),
    (70, 240, 240), (240, 50, 230), (170, 170, 170), (128, 128, 0),
    (145, 30, 30), (128, 128, 0), (0, 130, 70), (75, 0, 130),
    (0, 130, 130), (0, 35, 100), (245, 130, 0),
], dtype=np.uint8)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--er-dir", type=Path, required=True)
    p.add_argument("--full-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("classification_map_outputs"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=0,
                   help="Inference batch size; 0 reuses the checkpoint value")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--hsi-file", type=Path)
    p.add_argument("--test-label-file", type=Path)
    return p


def font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def checkpoint_args(checkpoint: dict, cli) -> SimpleNamespace:
    values = dict(checkpoint["args"])
    values["device"] = cli.device
    if cli.hsi_file is not None:
        values["hsi_file"] = str(cli.hsi_file.resolve())
    if cli.test_label_file is not None:
        values["test_label_file"] = str(cli.test_label_file.resolve())
    return SimpleNamespace(**values)


@torch.no_grad()
def predict_checkpoint(folder: Path, session: int, cli):
    checkpoint_path = folder / f"session_{session}.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    args = checkpoint_args(checkpoint, cli)
    sessions = parse_class_sessions(args.sessions)
    if session != int(checkpoint["session"]):
        raise ValueError(f"Session mismatch in {checkpoint_path}")
    seen = [int(c) for c in checkpoint["seen_classes"]]
    expected_seen = sorted(c for block in sessions[:session] for c in block)
    if seen != expected_seen:
        raise ValueError(f"Seen-class mismatch in {checkpoint_path}")

    hsi = engine.load_houston_hsi(
        args.hsi_file,
        args.hsi_key,
        band_indices=args.hsi_band_indices,
        normalize_per_band=bool(args.hsi_normalize_per_band),
    )
    model = persistent.build_model(args, hsi.shape[2])
    model.load_state_dict(checkpoint["model"], strict=True)
    current = list(sessions[session - 1])
    old = sorted(c for block in sessions[:session - 1] for c in block)
    model.set_prompt_task_context(old, current, session_idx=session)
    model.eval()

    dataset = HoustonPatchClassification(
        hsi_file=args.hsi_file,
        hsi_array=hsi,
        hsi_key=args.hsi_key,
        label_file=args.test_label_file,
        label_key=args.hsi_label_key,
        class_ids=seen,
        patch_size=args.hsi_patch_size,
        max_samples_per_class=0,
        seed=args.seed,
        band_indices=args.hsi_band_indices,
        normalize_per_band=bool(args.hsi_normalize_per_band),
    )
    inference_batch_size = int(cli.batch_size) if int(cli.batch_size) > 0 else int(args.batch_size)
    loader = DataLoader(
        dataset,
        batch_size=inference_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=cli.device.startswith("cuda"),
    )

    predictions = []
    for images, _ in loader:
        images = images.to(cli.device, non_blocking=True)
        logits, features, _ = engine.forward_for_classes(
            model, images, seen, return_features=True
        )
        if bool(args.use_prototypes):
            logits = engine.add_prototype_logits(
                model,
                logits,
                features,
                seen,
                alpha=args.prototype_alpha,
                temperature=args.prototype_temperature,
                pooling=args.prototype_pooling,
                use_margin_gate=bool(args.use_prototype_margin_gate),
                margin_gate_threshold=args.prototype_margin_gate_threshold,
                margin_gate_temperature=args.prototype_margin_gate_temperature,
                margin_gate_min=args.prototype_margin_gate_min,
            )
        logits = engine.mask_logits_to_classes(logits, seen)
        predictions.extend((logits.argmax(dim=1) + 1).cpu().tolist())

    if len(predictions) != len(dataset.samples):
        raise RuntimeError("Prediction count differs from evaluated test centers")
    shape = load_mat_array(args.test_label_file, args.hsi_label_key, ndim=2).shape
    prediction_map = np.zeros(shape, dtype=np.uint8)
    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    for (y, x, true_class), pred_class in zip(dataset.samples, predictions):
        prediction_map[y, x] = pred_class
        confusion[true_class - 1, pred_class - 1] += 1

    saved_confusion = np.load(folder / f"confusion_session_{session}.npy", allow_pickle=False)
    if not np.array_equal(confusion, saved_confusion):
        delta = int(np.abs(confusion - saved_confusion).sum())
        raise RuntimeError(
            f"Prediction audit failed for {folder.name} session {session}; "
            f"confusion L1 difference={delta}"
        )
    return prediction_map, args, seen, int(len(dataset.samples))


def colorize(label_map: np.ndarray) -> Image.Image:
    if label_map.min() < 0 or label_map.max() >= len(COLORS):
        raise ValueError("Map contains an unknown class id")
    return Image.fromarray(COLORS[label_map], mode="RGB")


def fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.resize((width, height), resample=Image.Resampling.NEAREST)


def compose(gt_maps, method_maps, out_png: Path):
    panel_w, panel_h = 1000, 183
    left, gap_x, gap_y = 240, 20, 14
    header_h, legend_h = 58, 108
    rows = ["Ground truth", "ER", "Full"]
    width = left + 3 * panel_w + 2 * gap_x
    height = header_h + 3 * panel_h + 2 * gap_y + legend_h
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, row_font, legend_font = font(24, True), font(22, True), font(16)

    for col in range(3):
        x = left + col * (panel_w + gap_x)
        text = f"Session {col + 1}"
        box = draw.textbbox((0, 0), text, font=title_font)
        draw.text((x + (panel_w - (box[2] - box[0])) / 2, 13), text,
                  fill="black", font=title_font)

    grid = [gt_maps, method_maps["ER"], method_maps["Full"]]
    for row, (label, maps) in enumerate(zip(rows, grid)):
        y = header_h + row * (panel_h + gap_y)
        box = draw.textbbox((0, 0), label, font=row_font)
        draw.text((left - 18 - (box[2] - box[0]), y + (panel_h - (box[3] - box[1])) / 2),
                  label, fill="black", font=row_font)
        for col, label_map in enumerate(maps):
            x = left + col * (panel_w + gap_x)
            canvas.paste(fit_panel(colorize(label_map), panel_w, panel_h), (x, y))
            draw.rectangle((x, y, x + panel_w - 1, y + panel_h - 1),
                           outline=(185, 185, 185), width=1)

    legend_y = header_h + 3 * panel_h + 2 * gap_y + 16
    cell_w = (width - left) // 8
    for idx, name in enumerate(CLASS_NAMES, start=1):
        row, col = divmod(idx - 1, 8)
        x = left + col * cell_w
        y = legend_y + row * 37
        draw.rectangle((x, y + 2, x + 23, y + 25), fill=tuple(COLORS[idx]),
                       outline=(60, 60, 60), width=1)
        draw.text((x + 31, y + 2), name, fill="black", font=legend_font)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, dpi=(300, 300), optimize=True)
    canvas.save(out_png.with_suffix(".pdf"), "PDF", resolution=300.0)


def main() -> None:
    cli = parser().parse_args()
    if cli.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    torch.set_num_threads(cli.threads)
    cli.output_dir.mkdir(parents=True, exist_ok=True)

    method_dirs = {"ER": cli.er_dir.resolve(), "Full": cli.full_dir.resolve()}
    method_maps = {name: [] for name in method_dirs}
    audits = []
    gt_maps = []
    reference_test_map = None

    for session in (1, 2, 3):
        reference_args = None
        for name, folder in method_dirs.items():
            pred_map, args, seen, count = predict_checkpoint(folder, session, cli)
            method_maps[name].append(pred_map)
            reference_args = args if reference_args is None else reference_args
            audits.append({
                "method": name,
                "session": session,
                "checkpoint": str(folder / f"session_{session}.pth"),
                "seen_classes": seen,
                "test_samples": count,
                "confusion_match": True,
            })
        test_map = load_mat_array(
            reference_args.test_label_file, reference_args.hsi_label_key, ndim=2
        ).astype(np.uint8)
        if reference_test_map is None:
            reference_test_map = test_map
        elif not np.array_equal(reference_test_map, test_map):
            raise ValueError("Methods do not use the same Houston test map")
        visible = np.isin(test_map, np.asarray(audits[-1]["seen_classes"]))
        gt_maps.append(np.where(visible, test_map, 0).astype(np.uint8))

    np.savez_compressed(
        cli.output_dir / "houston_classification_maps.npz",
        gt=np.stack(gt_maps),
        er=np.stack(method_maps["ER"]),
        full=np.stack(method_maps["Full"]),
    )
    (cli.output_dir / "audit_manifest.json").write_text(
        json.dumps({"audits": audits, "class_names": CLASS_NAMES}, indent=2),
        encoding="utf-8",
    )
    compose(gt_maps, method_maps, cli.output_dir / "houston_classification_maps.png")
    print(f"Saved audited maps to {cli.output_dir.resolve()}")


if __name__ == "__main__":
    main()
