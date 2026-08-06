"""Candidate-only CUDA global linear-light photometric calibration.

This module is deliberately separate from :mod:`video_photometric`, whose
NumPy/OpenCV implementation remains a legacy experiment bridge.  It consumes
only resident *real-source* colour tensors and caller-supplied common safe
background evidence.  It neither chooses an owner nor changes a pose, and a
rejected solve exposes CUDA-resident identity corrections rather than a
partial colour adjustment.

The model solves, in one anchored system, for corrections satisfying
``g_i * I_i + b_i ~= g_j * I_j + b_j``.  Every pair contributes spatially
disjoint train and held-out tiles.  The held-out pixels never enter the
normal equations and all gates are evaluated before a correction may be
applied to a real source tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence


class CudaPhotometricError(ValueError):
    """The C7 CUDA evidence or correction contract is not satisfied."""


@dataclass(frozen=True)
class CudaPhotometricConfig:
    """Fail-closed bounds for candidate C7 linear RGB gain/bias fitting."""

    minimum_training_pixels: int = 128
    minimum_held_out_pixels: int = 64
    held_out_tile_side_pixels: int = 16
    held_out_tile_modulus: int = 5
    held_out_tile_remainder: int = 0
    maximum_training_samples_per_pair: int = 32768
    # The merged plan's early-stop limit is |gain| <= 1.50.  Keep that
    # declared envelope here rather than silently introducing a tighter,
    # unrecorded candidate-selection criterion.
    gain_minimum: float = 1.0 / 1.50
    gain_maximum: float = 1.50
    bias_absolute_maximum: float = 0.08
    maximum_training_error_p95: float = 0.035
    maximum_training_error_max: float = 0.12
    maximum_held_out_error_p95: float = 0.035
    maximum_held_out_error_max: float = 0.12
    regularization: float = 1.0e-6

    def validated(self) -> "CudaPhotometricConfig":
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (
                self.minimum_training_pixels,
                self.minimum_held_out_pixels,
                self.held_out_tile_side_pixels,
                self.held_out_tile_modulus,
                self.held_out_tile_remainder,
                self.maximum_training_samples_per_pair,
            )
        ):
            raise CudaPhotometricError("C7 integer configuration values must be integers")
        if self.minimum_training_pixels < 32 or self.minimum_held_out_pixels < 32:
            raise CudaPhotometricError("C7 train and held-out support must each be at least 32 pixels")
        if self.held_out_tile_side_pixels < 4 or self.held_out_tile_modulus < 2:
            raise CudaPhotometricError("C7 held-out spatial tile configuration is invalid")
        if not 0 <= self.held_out_tile_remainder < self.held_out_tile_modulus:
            raise CudaPhotometricError("C7 held-out tile remainder is invalid")
        if self.maximum_training_samples_per_pair < self.minimum_training_pixels:
            raise CudaPhotometricError("C7 sample cap cannot be below minimum training support")
        values = (
            self.gain_minimum,
            self.gain_maximum,
            self.bias_absolute_maximum,
            self.maximum_training_error_p95,
            self.maximum_training_error_max,
            self.maximum_held_out_error_p95,
            self.maximum_held_out_error_max,
            self.regularization,
        )
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise CudaPhotometricError("C7 floating-point configuration must be finite")
        if not 0.0 < self.gain_minimum <= self.gain_maximum:
            raise CudaPhotometricError("C7 gain bounds are invalid")
        if min(
            self.bias_absolute_maximum,
            self.maximum_training_error_p95,
            self.maximum_training_error_max,
            self.maximum_held_out_error_p95,
            self.maximum_held_out_error_max,
            self.regularization,
        ) <= 0.0:
            raise CudaPhotometricError("C7 positive gates are invalid")
        if (
            self.maximum_training_error_p95 > self.maximum_training_error_max
            or self.maximum_held_out_error_p95 > self.maximum_held_out_error_max
        ):
            raise CudaPhotometricError("C7 p95 error gate cannot exceed its maximum gate")
        return self


@dataclass(frozen=True)
class CudaPhotometricOverlap:
    """Adjacent genuine-source safe overlap evidence on one CUDA device.

    RGB tensors are CHW sRGB (``uint8`` or finite floating point in ``[0, 1]``).
    The three masks must have already excluded foreground, depth edges,
    occlusions, and seam-risk pixels.  The solver refuses an overlap whose
    safe mask touches the separately supplied protection/risk domains.
    """

    left_frame_id: int
    right_frame_id: int
    left_bgr_srgb: Any
    right_bgr_srgb: Any
    left_valid_mask: Any
    right_valid_mask: Any
    safe_background_mask: Any
    protected_mask: Any
    risk_mask: Any
    # C7 accepts adjacent and genuine skip-one calibrated overlaps.  The
    # value is audit-only; all colour evidence remains the supplied sources.
    edge_kind: str = "adjacent"


@dataclass(frozen=True)
class CudaPhotometricCorrection:
    """One device-resident correction for a real source frame."""

    frame_id: int
    gain_bgr: Any
    bias_bgr: Any


@dataclass(frozen=True)
class CudaGlobalPhotometricResult:
    """Accepted global correction or fail-closed CUDA identity corrections."""

    accepted: bool
    corrections: tuple[CudaPhotometricCorrection, ...]
    audit: dict[str, object]

    def correction_for_frame(self, frame_id: int) -> CudaPhotometricCorrection:
        for correction in self.corrections:
            if correction.frame_id == frame_id:
                return correction
        raise CudaPhotometricError("C7 result has no correction for this real source frame")


def _validate_source_ids(source_frame_ids: Sequence[int]) -> tuple[int, ...]:
    ids = tuple(source_frame_ids)
    if len(ids) < 2:
        raise CudaPhotometricError("C7 needs at least two genuine source frames")
    if not all(isinstance(frame_id, int) and not isinstance(frame_id, bool) and frame_id >= 0 for frame_id in ids):
        raise CudaPhotometricError("C7 source identifiers must be non-negative integers")
    if len(set(ids)) != len(ids) or tuple(sorted(ids)) != ids:
        raise CudaPhotometricError("C7 source identifiers must be unique and chronological")
    return ids


def _require_mask(value: Any, *, shape: tuple[int, int], device: Any, label: str) -> Any:
    if tuple(getattr(value, "shape", ())) != shape or getattr(value, "device", None) != device:
        raise CudaPhotometricError(f"{label} must be an HxW tensor on the source CUDA device")
    return value.bool()


def _linear_rgb(torch: Any, image: Any, *, label: str) -> Any:
    if getattr(image, "ndim", None) != 3 or int(image.shape[0]) != 3:
        raise CudaPhotometricError(f"{label} must be a CHW BGR tensor")
    if not getattr(image, "is_cuda", False):
        raise CudaPhotometricError(f"{label} must remain CUDA-resident")
    if image.dtype == torch.uint8:
        encoded = image.to(dtype=torch.float32).div(255.0)
    elif getattr(image.dtype, "is_floating_point", False):
        if not bool(torch.isfinite(image).all().item()) or bool(((image < 0.0) | (image > 1.0)).any().item()):
            raise CudaPhotometricError(f"{label} floating sRGB values must be finite in [0, 1]")
        encoded = image.to(dtype=torch.float32)
    else:
        raise CudaPhotometricError(f"{label} must use uint8 or floating sRGB")
    return torch.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055).pow(2.4),
    )


def _srgb_from_linear(torch: Any, image: Any) -> Any:
    values = image.clamp(0.0, 1.0)
    return torch.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * values.pow(1.0 / 2.4) - 0.055,
    )


def _identity_result(
    torch: Any,
    *,
    source_frame_ids: tuple[int, ...],
    device: Any,
    config: CudaPhotometricConfig,
    reason: str,
    pairs: Sequence[dict[str, object]] = (),
) -> CudaGlobalPhotometricResult:
    corrections = tuple(
        CudaPhotometricCorrection(
            frame_id=frame_id,
            gain_bgr=torch.ones(3, dtype=torch.float32, device=device),
            bias_bgr=torch.zeros(3, dtype=torch.float32, device=device),
        )
        for frame_id in source_frame_ids
    )
    return CudaGlobalPhotometricResult(
        accepted=False,
        corrections=corrections,
        audit={
            "schema": "gemini305-video-cuda-global-photometric/v1",
            "candidate_only": True,
            "accepted": False,
            "fail_closed_identity": True,
            "rejection_reason": reason,
            "source_frame_ids": list(source_frame_ids),
            "pair_count": len(pairs),
            "pairs": list(pairs),
            "linear_light": True,
            "output_residency": "device_tensors",
            "dense_host_transfer_count": 0,
            "scalar_audit_only": True,
            "maximum_gain": float(config.gain_maximum),
            "maximum_absolute_bias": float(config.bias_absolute_maximum),
            "creates_owner": False,
            "creates_pose": False,
        },
    )


@dataclass(frozen=True)
class _PreparedPair:
    left_index: int
    right_index: int
    left_linear: Any
    right_linear: Any
    train_mask: Any
    held_out_mask: Any
    audit: dict[str, object]


def _prepare_pair(
    torch: Any,
    *,
    overlap: CudaPhotometricOverlap,
    left_index: int,
    right_index: int,
    config: CudaPhotometricConfig,
    device: Any | None,
) -> tuple[_PreparedPair | None, Any, str | None]:
    if not isinstance(overlap.edge_kind, str) or overlap.edge_kind not in {"adjacent", "skip_one_overlap"}:
        raise CudaPhotometricError("C7 overlap edge kind must be adjacent or skip_one_overlap")
    if left_index < 0 or right_index <= left_index:
        raise CudaPhotometricError("C7 overlaps must connect chronological genuine source pairs")
    left_linear = _linear_rgb(torch, overlap.left_bgr_srgb, label="left real source colour")
    right_linear = _linear_rgb(torch, overlap.right_bgr_srgb, label="right real source colour")
    if tuple(left_linear.shape) != tuple(right_linear.shape):
        raise CudaPhotometricError("C7 adjacent source colours must have identical CHW dimensions")
    if right_linear.device != left_linear.device:
        raise CudaPhotometricError("C7 adjacent source colours must remain on one CUDA device")
    if device is not None and left_linear.device != device:
        raise CudaPhotometricError("C7 cannot mix CUDA devices within one global solve")
    height, width = int(left_linear.shape[1]), int(left_linear.shape[2])
    if height < 4 or width < 4:
        raise CudaPhotometricError("C7 source overlap is too small")
    shape = (height, width)
    left_valid = _require_mask(overlap.left_valid_mask, shape=shape, device=left_linear.device, label="left_valid_mask")
    right_valid = _require_mask(overlap.right_valid_mask, shape=shape, device=left_linear.device, label="right_valid_mask")
    safe = _require_mask(overlap.safe_background_mask, shape=shape, device=left_linear.device, label="safe_background_mask")
    protected = _require_mask(overlap.protected_mask, shape=shape, device=left_linear.device, label="protected_mask")
    risk = _require_mask(overlap.risk_mask, shape=shape, device=left_linear.device, label="risk_mask")
    if bool((safe & (protected | risk)).any().item()):
        raise CudaPhotometricError("C7 safe background overlaps protected or risk evidence")
    common = left_valid & right_valid
    if bool((safe & ~common).any().item()):
        raise CudaPhotometricError("C7 safe background must be jointly valid in both genuine sources")
    # A colour at the encoding rails cannot establish an affine exposure
    # relation.  This does not invalidate black RGB for rendering; it merely
    # excludes unidentifiable samples from C7's evidence population.
    left_encoded = overlap.left_bgr_srgb if overlap.left_bgr_srgb.dtype == torch.uint8 else (overlap.left_bgr_srgb * 255.0)
    right_encoded = overlap.right_bgr_srgb if overlap.right_bgr_srgb.dtype == torch.uint8 else (overlap.right_bgr_srgb * 255.0)
    unclipped = (
        (left_encoded.amin(dim=0) >= 8.0)
        & (left_encoded.amax(dim=0) <= 247.0)
        & (right_encoded.amin(dim=0) >= 8.0)
        & (right_encoded.amax(dim=0) <= 247.0)
    )
    stable = safe & common & unclipped
    rows = torch.arange(height, device=left_linear.device).view(height, 1)
    columns = torch.arange(width, device=left_linear.device).view(1, width)
    tile = rows.div(config.held_out_tile_side_pixels, rounding_mode="floor") + columns.div(config.held_out_tile_side_pixels, rounding_mode="floor")
    held_out = stable & (tile.remainder(config.held_out_tile_modulus) == config.held_out_tile_remainder)
    train = stable & ~held_out
    stable_count = int(stable.sum().item())
    train_count = int(train.sum().item())
    held_out_count = int(held_out.sum().item())
    audit = {
        "left_frame_id": int(overlap.left_frame_id),
        "right_frame_id": int(overlap.right_frame_id),
        "edge_kind": overlap.edge_kind,
        "safe_shared_pixel_count": int(safe.sum().item()),
        "stable_shared_pixel_count": stable_count,
        "training_pixel_count": train_count,
        "held_out_pixel_count": held_out_count,
        "held_out_split": {
            "kind": "deterministic_spatial_tiles/v1",
            "tile_side_pixels": int(config.held_out_tile_side_pixels),
            "tile_modulus": int(config.held_out_tile_modulus),
            "tile_remainder": int(config.held_out_tile_remainder),
        },
    }
    if train_count < config.minimum_training_pixels or held_out_count < config.minimum_held_out_pixels:
        audit.update({
            "accepted": False,
            "reason": "insufficient_training_support" if train_count < config.minimum_training_pixels else "insufficient_held_out_support",
        })
        return None, left_linear.device, str(audit["reason"])
    audit.update({"accepted": True, "reason": None})
    return _PreparedPair(
        left_index,
        right_index,
        left_linear,
        right_linear,
        train,
        held_out,
        audit,
    ), left_linear.device, None


def _pair_training_rows(
    torch: Any,
    *,
    pair: _PreparedPair,
    left_index: int,
    right_index: int,
    anchor_index: int,
    source_to_unknown: dict[int, int],
    unknown_count: int,
    channel: int,
    maximum_samples: int,
) -> tuple[Any, Any, int]:
    indices = pair.train_mask.flatten().nonzero(as_tuple=False).flatten()
    if int(indices.numel()) > maximum_samples:
        stride = math.ceil(int(indices.numel()) / maximum_samples)
        indices = indices[::stride]
    left = pair.left_linear[channel].flatten()[indices]
    right = pair.right_linear[channel].flatten()[indices]
    rows = torch.zeros((int(indices.numel()), unknown_count), dtype=torch.float32, device=left.device)
    target = torch.zeros(int(indices.numel()), dtype=torch.float32, device=left.device)

    def gain_column(source_index: int) -> int:
        return 2 * source_to_unknown[source_index]

    def bias_column(source_index: int) -> int:
        return gain_column(source_index) + 1

    if left_index == anchor_index:
        target.copy_(-left)
    else:
        rows[:, gain_column(left_index)] = left
        rows[:, bias_column(left_index)] = 1.0
    if right_index == anchor_index:
        target.add_(right)
    else:
        rows[:, gain_column(right_index)] -= right
        rows[:, bias_column(right_index)] -= 1.0
    return rows, target, int(indices.numel())


def _corrected_pair_error(torch: Any, pair: _PreparedPair, gains: Any, biases: Any, *, held_out: bool) -> Any:
    mask = pair.held_out_mask if held_out else pair.train_mask
    left = pair.left_linear[:, mask]
    right = pair.right_linear[:, mask]
    return (left * gains[pair.left_index].view(3, 1) + biases[pair.left_index].view(3, 1) - right * gains[pair.right_index].view(3, 1) - biases[pair.right_index].view(3, 1)).abs().amax(dim=0)


def solve_cuda_global_photometric(
    torch: Any,
    *,
    source_frame_ids: Sequence[int],
    overlaps: Sequence[CudaPhotometricOverlap],
    config: CudaPhotometricConfig | None = None,
    anchor_frame_id: int | None = None,
    anchor_policy: str = "source_0",
) -> CudaGlobalPhotometricResult:
    """Solve C7 gain/bias globally from safe real-source overlap samples.

    ``overlaps`` must form one connected chronological graph of genuine
    adjacent and/or skip-one overlaps.  The graph is solved in one anchored
    system, never by chaining source-0 pair corrections.
    Dense image, mask, sample and normal-equation tensors never leave CUDA;
    only scalar audit counters/errors are materialised for the returned JSON
    safe audit.  Failure returns identity tensors for every supplied source.
    """

    settings = (config or CudaPhotometricConfig()).validated()
    ids = _validate_source_ids(source_frame_ids)
    if not overlaps:
        raise CudaPhotometricError("C7 needs at least one genuine real-source overlap")
    index_by_id = {frame_id: index for index, frame_id in enumerate(ids)}
    if anchor_frame_id is None:
        anchor_index = 0
    elif anchor_frame_id in index_by_id:
        anchor_index = index_by_id[anchor_frame_id]
    else:
        raise CudaPhotometricError("C7 anchor frame must be a genuine source")
    if anchor_policy not in {"source_0", "median_exposure"}:
        raise CudaPhotometricError("C7 anchor policy is unsupported")
    source_to_unknown = {
        index: unknown_index
        for unknown_index, index in enumerate(index for index in range(len(ids)) if index != anchor_index)
    }
    prepared: list[_PreparedPair] = []
    pair_audits: list[dict[str, object]] = []
    device: Any | None = None
    graph_adjacency = [set() for _ in ids]
    seen_edges: set[tuple[int, int]] = set()
    for overlap in overlaps:
        if overlap.left_frame_id not in index_by_id or overlap.right_frame_id not in index_by_id:
            raise CudaPhotometricError("C7 overlap references a non-source frame")
        left_index = index_by_id[overlap.left_frame_id]
        right_index = index_by_id[overlap.right_frame_id]
        if right_index <= left_index or right_index - left_index > 2:
            raise CudaPhotometricError("C7 overlap must be chronological and span at most one source")
        edge = (left_index, right_index)
        if edge in seen_edges:
            raise CudaPhotometricError("C7 overlap graph cannot contain duplicate source edges")
        seen_edges.add(edge)
        graph_adjacency[left_index].add(right_index)
        graph_adjacency[right_index].add(left_index)
        pair, pair_device, rejection = _prepare_pair(
            torch,
            overlap=overlap,
            left_index=left_index,
            right_index=right_index,
            config=settings,
            device=device,
        )
        device = pair_device
        if pair is None:
            assert device is not None
            pair_audits.append({
                "left_frame_id": int(overlap.left_frame_id),
                "right_frame_id": int(overlap.right_frame_id),
                "edge_kind": overlap.edge_kind,
                "accepted": False,
                "reason": rejection,
            })
            return _identity_result(
                torch,
                source_frame_ids=ids,
                device=device,
                config=settings,
                reason=f"rejected_pair_{overlap.left_frame_id}_{overlap.right_frame_id}:{rejection}",
                pairs=pair_audits,
            )
        prepared.append(pair)
        pair_audits.append(pair.audit)
    assert device is not None
    reached = {anchor_index}
    frontier = [anchor_index]
    while frontier:
        current = frontier.pop()
        for neighbour in graph_adjacency[current]:
            if neighbour not in reached:
                reached.add(neighbour)
                frontier.append(neighbour)
    if len(reached) != len(ids):
        return _identity_result(
            torch, source_frame_ids=ids, device=device, config=settings,
            reason="disconnected_global_photometric_graph", pairs=pair_audits,
        )
    unknown_count = 2 * (len(ids) - 1)
    all_rows: list[Any] = []
    all_targets: list[Any] = []
    sampled_training_count = 0
    try:
        for channel in range(3):
            channel_rows: list[Any] = []
            channel_targets: list[Any] = []
            for pair in prepared:
                rows, target, sampled = _pair_training_rows(
                    torch,
                    pair=pair,
                    left_index=pair.left_index,
                    right_index=pair.right_index,
                    anchor_index=anchor_index,
                    source_to_unknown=source_to_unknown,
                    unknown_count=unknown_count,
                    channel=channel,
                    maximum_samples=settings.maximum_training_samples_per_pair,
                )
                # ``sampled`` is the number of spatial pixels, not the three
                # per-channel rows used by the colour solve.
                if channel == 0:
                    sampled_training_count += sampled
                channel_rows.append(rows)
                channel_targets.append(target)
            all_rows.append(torch.cat(channel_rows, dim=0))
            all_targets.append(torch.cat(channel_targets, dim=0))
        gains = torch.ones((len(ids), 3), dtype=torch.float32, device=device)
        biases = torch.zeros((len(ids), 3), dtype=torch.float32, device=device)
        regularizer = torch.eye(unknown_count, dtype=torch.float32, device=device) * float(settings.regularization)
        for channel in range(3):
            design = all_rows[channel]
            target = all_targets[channel]
            normal = design.transpose(0, 1).matmul(design) + regularizer
            rhs = design.transpose(0, 1).matmul(target)
            solution = torch.linalg.solve(normal, rhs)
            for source_index, unknown_index in source_to_unknown.items():
                gains[source_index, channel] = solution[2 * unknown_index]
                biases[source_index, channel] = solution[2 * unknown_index + 1]
    except RuntimeError:
        return _identity_result(
            torch,
            source_frame_ids=ids,
            device=device,
            config=settings,
            reason="global_cuda_linear_solver_failure",
            pairs=pair_audits,
        )
    if not bool(torch.isfinite(gains).all().item()) or not bool(torch.isfinite(biases).all().item()):
        return _identity_result(torch, source_frame_ids=ids, device=device, config=settings, reason="nonfinite_global_solution", pairs=pair_audits)

    training_errors: list[Any] = []
    held_out_errors: list[Any] = []
    for pair, audit in zip(prepared, pair_audits, strict=True):
        training_error = _corrected_pair_error(torch, pair, gains, biases, held_out=False)
        held_out_error = _corrected_pair_error(torch, pair, gains, biases, held_out=True)
        training_errors.append(training_error)
        held_out_errors.append(held_out_error)
        audit.update({
            "training_error_p95_linear": float(torch.quantile(training_error, 0.95).item()),
            "training_error_max_linear": float(training_error.max().item()),
            "held_out_error_p95_linear": float(torch.quantile(held_out_error, 0.95).item()),
            "held_out_error_max_linear": float(held_out_error.max().item()),
        })
    train_all = torch.cat(training_errors)
    held_all = torch.cat(held_out_errors)
    gain_min = float(gains.min().item())
    gain_max = float(gains.max().item())
    bias_max = float(biases.abs().max().item())
    train_p95 = float(torch.quantile(train_all, 0.95).item())
    train_max = float(train_all.max().item())
    held_p95 = float(torch.quantile(held_all, 0.95).item())
    held_max = float(held_all.max().item())
    accepted = bool(
        gain_min >= settings.gain_minimum
        and gain_max <= settings.gain_maximum
        and bias_max <= settings.bias_absolute_maximum
        and train_p95 <= settings.maximum_training_error_p95
        and train_max <= settings.maximum_training_error_max
        and held_p95 <= settings.maximum_held_out_error_p95
        and held_max <= settings.maximum_held_out_error_max
    )
    if not accepted:
        if gain_min < settings.gain_minimum or gain_max > settings.gain_maximum:
            reason = "global_gain_out_of_bounds"
        elif bias_max > settings.bias_absolute_maximum:
            reason = "global_bias_out_of_bounds"
        elif held_p95 > settings.maximum_held_out_error_p95 or held_max > settings.maximum_held_out_error_max:
            reason = "held_out_error_exceeds_gate"
        else:
            reason = "training_error_exceeds_gate"
        for audit in pair_audits:
            audit["accepted"] = False
            audit["reason"] = reason
        rejected = _identity_result(torch, source_frame_ids=ids, device=device, config=settings, reason=reason, pairs=pair_audits)
        # Preserve scalar-only diagnostics for a rejected graph.  Identity
        # corrections remain mandatory, but the audit must distinguish a
        # real exposure conflict from an implementation/gauge problem.
        rejected.audit.update({
            "anchor_frame_id": int(ids[anchor_index]),
            "anchor_policy": anchor_policy,
            "graph_edge_count": len(prepared),
            "graph_edge_kinds": sorted({str(pair.audit["edge_kind"]) for pair in prepared}),
            "unaccepted_solution_gain_min": gain_min,
            "unaccepted_solution_gain_max": gain_max,
            "unaccepted_solution_bias_absolute_max": bias_max,
            "unaccepted_training_error_p95_linear": train_p95,
            "unaccepted_held_out_error_p95_linear": held_p95,
        })
        return rejected
    corrections = tuple(
        CudaPhotometricCorrection(frame_id=frame_id, gain_bgr=gains[index], bias_bgr=biases[index])
        for index, frame_id in enumerate(ids)
    )
    for audit in pair_audits:
        audit["accepted"] = True
        audit["reason"] = None
    return CudaGlobalPhotometricResult(
        accepted=True,
        corrections=corrections,
        audit={
            "schema": "gemini305-video-cuda-global-photometric/v1",
            "candidate_only": True,
            "accepted": True,
            "fail_closed_identity": False,
            "linear_light": True,
            "source_frame_ids": list(ids),
            "anchor_frame_id": int(ids[anchor_index]),
            "anchor_policy": anchor_policy,
            "pair_count": len(prepared),
            "graph_edge_count": len(prepared),
            "graph_edge_kinds": sorted({str(pair.audit["edge_kind"]) for pair in prepared}),
            "training_sample_count": sampled_training_count,
            "training_error_p95_linear": train_p95,
            "training_error_max_linear": train_max,
            "held_out_error_p95_linear": held_p95,
            "held_out_error_max_linear": held_max,
            "global_gain_min": gain_min,
            "global_gain_max": gain_max,
            "global_bias_absolute_max": bias_max,
            "pairs": pair_audits,
            "output_residency": "device_tensors",
            "dense_host_transfer_count": 0,
            "scalar_audit_only": True,
            "creates_owner": False,
            "creates_pose": False,
            "applies_only_real_source_colours": True,
        },
    )


def apply_cuda_global_photometric_correction(
    torch: Any,
    *,
    real_source_bgr_srgb: Any,
    frame_id: int,
    result: CudaGlobalPhotometricResult,
) -> tuple[Any, dict[str, object]]:
    """Apply an accepted C7 correction to one real source colour tensor.

    The input is never mutated and no owner argument exists: this primitive
    cannot alter provenance.  It returns a same-device BGR tensor in the
    original sRGB dtype plus scalar-only audit evidence.
    """

    if not result.accepted:
        raise CudaPhotometricError("C7 refuses to apply a rejected global photometric result")
    if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 0:
        raise CudaPhotometricError("C7 correction frame identifier must be a real non-negative integer")
    linear = _linear_rgb(torch, real_source_bgr_srgb, label="real source colour")
    correction = result.correction_for_frame(frame_id)
    if correction.gain_bgr.device != linear.device or correction.bias_bgr.device != linear.device:
        raise CudaPhotometricError("C7 correction and real source colour must remain on one CUDA device")
    if tuple(correction.gain_bgr.shape) != (3,) or tuple(correction.bias_bgr.shape) != (3,):
        raise CudaPhotometricError("C7 correction must contain device-resident BGR triplets")
    corrected = _srgb_from_linear(
        torch,
        linear * correction.gain_bgr.view(3, 1, 1) + correction.bias_bgr.view(3, 1, 1),
    )
    if real_source_bgr_srgb.dtype == torch.uint8:
        output = corrected.mul(255.0).round().clamp_(0.0, 255.0).to(dtype=torch.uint8)
    else:
        output = corrected.to(dtype=real_source_bgr_srgb.dtype)
    return output, {
        "schema": "gemini305-video-cuda-photometric-apply/v1",
        "candidate_only": True,
        "frame_id": int(frame_id),
        "output_residency": "device_tensor",
        "dense_host_transfer_count": 0,
        "creates_owner": False,
        "creates_pose": False,
        "applies_only_real_source_colours": True,
        "mutates_source_tensor": False,
        "accepted_global_result_required": True,
    }


__all__ = [
    "CudaGlobalPhotometricResult",
    "CudaPhotometricConfig",
    "CudaPhotometricCorrection",
    "CudaPhotometricError",
    "CudaPhotometricOverlap",
    "apply_cuda_global_photometric_correction",
    "solve_cuda_global_photometric",
]
