"""First-gate panel-native FastSAM whole-object lock diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np

from panorama_demo.fastsam_onnx import (
    FastSAMOnnxConfig,
    FastSAMOnnxRunner,
)
from panorama_demo.inspection_chain_seam import (
    ChainSeamConfig,
    PanelLocalEvidence,
    solve_adjacent_panel_chain,
)
from panorama_demo.inspection_fastsam_track import (
    build_fastsam_rgbd_candidate,
    parse_fastsam_polygons,
    polygon_mask,
    track_fastsam_rgbd_candidates,
)
from panorama_demo.inspection_multiview import (
    InspectionMultiviewConfig,
    InspectionMultiviewLayout,
    VirtualPerspectivePanel,
    _build_depth_mesh_panel_remap,
    _depth_confidence,
    _read_rgbd,
    _reference_panel_inverse_maps,
    _undistortion_maps,
)
from panorama_demo.panel_native_object_lock import (
    PanelNativeLockConfig,
    PanelNativeObservation,
    baseline_pair_costs,
    map_mask_through_existing_inverse,
    mask_overlap_metrics,
    observation_identity_audit,
)
from panorama_demo.session import load_rgbd_session


def _layout_from_report(value: dict[str, object]) -> InspectionMultiviewLayout:
    return InspectionMultiviewLayout(
        width=int(value["width"]),
        height=int(value["height"]),
        reference_depth_mm=float(value["reference_depth_mm"]),
        scan_axis=tuple(float(item) for item in value["scan_axis_world"]),
        down_axis=tuple(float(item) for item in value["down_axis_world"]),
        normal_axis=tuple(float(item) for item in value["normal_axis_world"]),
        panels=tuple(
            VirtualPerspectivePanel(
                panel_index=int(item["panel_index"]),
                anchor_scan_mm=float(item["anchor_scan_mm"]),
                canvas_offset_x=float(item["canvas_offset_x"]),
                center_world_mm=tuple(
                    float(component) for component in item["center_world_mm"]
                ),
            )
            for item in value["panels"]
        ),
        panel_step_mm=float(value["panel_step_mm"]),
        canvas_megapixels=float(value["canvas_megapixels"]),
    )


def _full_baseline(
    formal_output: Path,
    layout: InspectionMultiviewLayout,
    crop: dict[str, object],
    panel_by_frame: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = cv2.imread(
        str(formal_output / "mosaic_inspection.png"), cv2.IMREAD_COLOR
    )
    encoded = cv2.imread(
        str(formal_output / "inspection_owner.png"), cv2.IMREAD_UNCHANGED
    )
    x, y, width, height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    if (
        image is None
        or encoded is None
        or encoded.dtype != np.uint16
        or image.shape != (height, width, 3)
        or encoded.shape != (height, width)
    ):
        raise RuntimeError("Formal inspection baseline is incomplete")
    full_image = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
    full_image[y : y + height, x : x + width] = image
    decoded_frame = encoded.astype(np.int32) - 1
    decoded_panel = np.full(decoded_frame.shape, -1, dtype=np.int16)
    for frame_id, panel_index in panel_by_frame.items():
        decoded_panel[decoded_frame == frame_id] = np.int16(panel_index)
    if np.any((decoded_frame >= 0) & (decoded_panel < 0)):
        values = np.unique(decoded_frame[(decoded_frame >= 0) & (decoded_panel < 0)])
        raise RuntimeError(
            f"Formal owner contains unknown panel source frames: {values.tolist()}"
        )
    full_owner = np.full((layout.height, layout.width), -1, dtype=np.int16)
    full_owner[y : y + height, x : x + width] = decoded_panel
    full_valid = full_owner >= 0
    return full_image, full_owner, full_valid


def _bbox(mask: np.ndarray) -> list[int]:
    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return [0, 0, 0, 0]
    return [
        int(np.min(xx)),
        int(np.min(yy)),
        int(np.max(xx) - np.min(xx) + 1),
        int(np.max(yy) - np.min(yy) + 1),
    ]


def _peer_ambiguity(
    observation: PanelNativeObservation,
    peer_panel_index: int,
    observations_by_panel: list[list[PanelNativeObservation]],
    threshold: float,
) -> tuple[bool, list[dict[str, object]]]:
    matches: list[dict[str, object]] = []
    for peer in observations_by_panel[peer_panel_index]:
        iou, smaller = mask_overlap_metrics(
            observation.target_mask, peer.target_mask
        )
        if smaller >= threshold:
            matches.append(
                {
                    "candidate_id": int(peer.candidate.candidate_id),
                    "target_mask_iou": iou,
                    "target_smaller_mask_coverage": smaller,
                }
            )
    return len(matches) > 1, matches


def _contact_sheet(
    panels: list[dict[str, object]], width: int = 320
) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for panel in panels:
        image = np.asarray(panel["image_bgr"]).copy()
        for observation in panel["observations"]:
            polygon = observation.candidate.polygon_xy
            cv2.polylines(image, [polygon], True, (0, 255, 0), 2)
        cv2.putText(
            image,
            f"P{panel['panel_index']} F{panel['frame_id']}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        tile_height = int(round(image.shape[0] * width / image.shape[1]))
        tiles.append(cv2.resize(image, (width, tile_height)))
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    tile_height = max(tile.shape[0] for tile in tiles)
    sheet = np.zeros((rows * tile_height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = (index // columns) * tile_height
        x = (index % columns) * width
        sheet[y : y + tile.shape[0], x : x + width] = tile
    return sheet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("session", type=Path)
    parser.add_argument("formal_output", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    started = time.perf_counter()
    session_path = arguments.session.expanduser().resolve()
    formal_output = arguments.formal_output.expanduser().resolve()
    labels_path = arguments.labels.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if output == formal_output:
        raise ValueError("Diagnostic output must not be the formal output")

    metadata = json.loads(
        (formal_output / "inspection_meta.json").read_text(encoding="utf-8")
    )
    renderer = metadata["renderer"]
    if renderer["schema"] != "gemini305-inspection-multiview/v1":
        raise ValueError("Panel-native diagnostic requires the v9 renderer")
    transforms = json.loads(
        (formal_output / "transforms.json").read_text(encoding="utf-8")
    )
    layout = _layout_from_report(renderer["layout"])
    formal_config = InspectionMultiviewConfig.from_mapping(renderer["config"])
    lock_config = PanelNativeLockConfig()
    lock_config.validate()
    session = load_rgbd_session(session_path)
    frame_by_id = {int(frame.frame_id): frame for frame in session.frames}
    frame_ids = [int(value) for value in renderer["frame_ids"]]
    frames = [frame_by_id[value] for value in frame_ids]
    pose_by_id = {
        int(item["node_id"]): np.asarray(
            item["camera_to_world"], dtype=np.float64
        )
        for item in transforms["nodes"]
    }
    poses = [pose_by_id[value] for value in frame_ids]
    panel_selection = renderer["selected_panel_sources"]
    if len(panel_selection) < 2:
        raise RuntimeError(
            "Panel-native diagnostic requires at least two formal panel "
            f"sources, got {len(panel_selection)}"
        )
    panel_by_frame = {
        int(item["frame_id"]): int(item["panel_index"])
        for item in panel_selection
    }
    full_image, baseline_owner, full_valid = _full_baseline(
        formal_output, layout, renderer["crop"], panel_by_frame
    )
    source_audit_by_frame = {
        int(item["frame_id"]): item for item in renderer["source_audits"]
    }
    maps = _undistortion_maps(session.calibration)
    candidates_by_panel = []
    observations_by_panel: list[list[PanelNativeObservation]] = []
    observation_audits: list[dict[str, object]] = []
    panel_records: list[dict[str, object]] = []
    candidate_id = 0
    raw_polygon_count = 0
    fastsam = (
        FastSAMOnnxRunner(
            labels_path,
            device_id=0,
            allow_cpu_diagnostic_fallback=False,
            config=FastSAMOnnxConfig(max_detections=80),
        )
        if labels_path.is_file()
        else None
    )
    for expected_panel, panel_payload in enumerate(panel_selection):
        panel_index = int(panel_payload["panel_index"])
        source_position = int(panel_payload["source_position"])
        frame_id = int(panel_payload["frame_id"])
        if panel_index != expected_panel:
            raise RuntimeError("Formal panels are not ordered")
        frame = frames[source_position]
        pose = poses[source_position]
        if int(frame.frame_id) != frame_id:
            raise RuntimeError("Formal panel source frame identity changed")
        image, depth, geometric_valid = _read_rgbd(
            frame, session.calibration, maps
        )
        reliable = (
            geometric_valid
            & np.isfinite(depth)
            & (depth >= formal_config.minimum_depth_mm)
            & (depth <= formal_config.maximum_depth_mm)
        )
        confidence, edge = _depth_confidence(depth, reliable, formal_config)
        foreground_margin = max(
            formal_config.foreground_depth_margin_mm,
            formal_config.foreground_depth_margin_ratio
            * layout.reference_depth_mm,
        )
        geometry_depth_limit = min(
            layout.reference_depth_mm - foreground_margin,
            layout.reference_depth_mm
            * formal_config.foreground_reference_depth_ratio,
        )
        geometry_depth = reliable & (depth < geometry_depth_limit)
        yy, xx = np.indices(depth.shape, dtype=np.float32)
        radius = np.sqrt(
            (
                (xx - session.calibration.cx)
                / max(1.0, session.calibration.width * 0.5)
            )
            ** 2
            + (
                (yy - session.calibration.cy)
                / max(1.0, session.calibration.height * 0.5)
            )
            ** 2
        )
        view_centrality = np.clip(1.0 - radius, 0.0, 1.0)
        confidence[geometry_depth] *= (
            0.35 + 0.65 * view_centrality[geometry_depth]
        )
        projection_valid = (
            geometry_depth & ~edge & (confidence >= np.float32(0.50))
        )
        mesh = _build_depth_mesh_panel_remap(
            source_depth_mm=depth,
            source_solver_valid=projection_valid,
            source_pose=pose,
            panel_index=panel_index,
            layout=layout,
            intrinsics=session.calibration,
            config=formal_config,
        )
        formal_mesh = source_audit_by_frame[frame_id]["depth_mesh"]
        if (
            int(formal_mesh["valid_target_pixel_count"])
            != int(mesh.audit["valid_target_pixel_count"])
            or int(formal_mesh["accepted_cell_count"])
            != int(mesh.audit["accepted_cell_count"])
        ):
            raise RuntimeError(
                f"Existing inverse mesh audit changed for frame {frame_id}"
            )
        (
            ref_x0,
            reference_map_x,
            reference_map_y,
            ref_valid,
            _,
        ) = _reference_panel_inverse_maps(
            source_pose=pose,
            panel_index=panel_index,
            layout=layout,
            intrinsics=session.calibration,
        )
        if ref_x0 != int(round(layout.panels[panel_index].canvas_offset_x)):
            raise RuntimeError("Reference panel footprint changed")
        polygons = (
            [
                np.ascontiguousarray(proposal.polygon_xy, dtype=np.int32)
                for proposal in fastsam.predict(image)
                if 0.001 <= float(proposal.mask.mean()) <= 0.30
            ]
            if fastsam is not None
            else parse_fastsam_polygons(
                labels_path / f"{frame_id:08d}.txt",
                width=session.calibration.width,
                height=session.calibration.height,
            )
        )
        raw_polygon_count += len(polygons)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        panel_candidates = []
        panel_observations: list[PanelNativeObservation] = []
        for polygon in polygons:
            candidate = build_fastsam_rgbd_candidate(
                candidate_id=candidate_id,
                source_index=panel_index,
                frame_id=frame_id,
                polygon_xy=polygon,
                image_bgr=image,
                lab_image=lab,
                depth_mm=depth,
                reliable_depth=reliable,
                camera_to_world=pose,
                intrinsics=session.calibration,
                reference_depth_mm=layout.reference_depth_mm,
            )
            candidate_id += 1
            if candidate is None:
                continue
            panel_candidates.append(candidate)
            observation, observation_audit = (
                map_mask_through_existing_inverse(
                    candidate=candidate,
                    panel_index=panel_index,
                    frame_id=frame_id,
                    source_mask=polygon_mask(candidate, depth.shape),
                    source_image_bgr=image,
                    inverse_map_x=reference_map_x,
                    inverse_map_y=reference_map_y,
                    inverse_valid_mask=ref_valid,
                    corner_x=int(ref_x0),
                    canvas_shape=(layout.height, layout.width),
                    config=lock_config,
                )
            )
            observation_audits.append(observation_audit)
            if observation is not None:
                panel_observations.append(observation)
        candidates_by_panel.append(panel_candidates)
        observations_by_panel.append(panel_observations)
        panel_records.append(
            {
                "panel_index": panel_index,
                "frame_id": frame_id,
                "image_bgr": image,
                "mesh": mesh,
                "reference_valid": ref_valid,
                "candidate_count": len(panel_candidates),
                "observations": panel_observations,
            }
        )
        print(
            f"panel {panel_index + 1}/15 frame {frame_id}: "
            f"{len(polygons)} polygons, {len(panel_candidates)} RGB-D, "
            f"{len(panel_observations)} whole-map",
            flush=True,
        )

    tracks = track_fastsam_rgbd_candidates(
        candidates_by_panel,
        minimum_voxel_overlap_ratio=(
            lock_config.minimum_world_voxel_overlap_ratio
        ),
        maximum_source_gap=2,
    )
    observation_by_candidate = {
        item.candidate.candidate_id: item
        for panel in observations_by_panel
        for item in panel
    }
    proposed: list[dict[str, object]] = []
    rejected_tracks: list[dict[str, object]] = []
    for track in tracks:
        observations = [
            observation_by_candidate[value]
            for value in track.candidate_ids
            if value in observation_by_candidate
        ]
        base = {
            "track_id": int(track.track_id),
            "candidate_ids": [int(value) for value in track.candidate_ids],
            "world_track": track.audit,
            "whole_map_observation_count": len(observations),
        }
        if len({item.panel_index for item in observations}) < 2:
            rejected_tracks.append(
                {
                    **base,
                    "reason": "fewer_than_two_whole_map_panel_views",
                }
            )
            continue
        pair_audits: list[dict[str, object]] = []
        support = Counter()
        for first_index, first in enumerate(observations):
            for second in observations[first_index + 1 :]:
                if first.panel_index == second.panel_index:
                    continue
                pair = observation_identity_audit(
                    first, second, config=lock_config
                )
                first_ambiguous, first_matches = _peer_ambiguity(
                    first,
                    second.panel_index,
                    observations_by_panel,
                    lock_config.merge_split_peer_overlap_ratio,
                )
                second_ambiguous, second_matches = _peer_ambiguity(
                    second,
                    first.panel_index,
                    observations_by_panel,
                    lock_config.merge_split_peer_overlap_ratio,
                )
                pair["first_merge_split_peer_matches"] = first_matches
                pair["second_merge_split_peer_matches"] = second_matches
                pair["merge_split_ambiguous"] = bool(
                    first_ambiguous or second_ambiguous
                )
                pair["pass"] = bool(
                    pair["pass"] and not pair["merge_split_ambiguous"]
                )
                pair_audits.append(pair)
                if pair["pass"]:
                    support[first.candidate.candidate_id] += 1
                    support[second.candidate.candidate_id] += 1
        if not support:
            rejected_tracks.append(
                {
                    **base,
                    "reason": (
                        "no_two_view_identity_pair_after_merge_split_gate"
                    ),
                    "pair_audits": pair_audits,
                }
            )
            continue
        ranked_observations = sorted(
            (
                item
                for item in observations
                if support[item.candidate.candidate_id] > 0
            ),
            key=lambda item: (
                -support[item.candidate.candidate_id],
                -item.clarity,
                -item.centrality,
                -item.inverse_source_coverage_ratio,
                item.frame_id,
            ),
        )
        selected = ranked_observations[0]
        proposed.append(
            {
                **base,
                "selected": selected,
                "ranked_observations": ranked_observations,
                "selected_candidate_id": int(
                    selected.candidate.candidate_id
                ),
                "selected_panel_index": int(selected.panel_index),
                "selected_frame_id": int(selected.frame_id),
                "selected_target_bbox_xywh": _bbox(selected.target_mask),
                "selected_target_pixel_count": int(
                    np.count_nonzero(selected.target_mask)
                ),
                "support_pair_count": int(
                    support[selected.candidate.candidate_id]
                ),
                "pair_audits": pair_audits,
            }
        )

    seams = renderer["background_seam_audit"]["panel_chain_topology"]["seams"]
    nominal_boundaries = [float(item["nominal_x"]) for item in seams]
    chain_config = ChainSeamConfig(
        corridor_width_pixels=int(
            formal_config.chain_seam_corridor_width_pixels
        ),
        maximum_row_step_pixels=int(
            formal_config.chain_seam_maximum_row_step_pixels
        ),
        smoothness_penalty=2.0,
        adaptive_boundary_maximum_shift_pixels=int(
            formal_config.chain_seam_adaptive_boundary_maximum_shift_pixels
        ),
        adaptive_boundary_risk_guard_pixels=int(
            formal_config.chain_seam_adaptive_boundary_risk_guard_pixels
        ),
        adaptive_boundary_minimum_common_coverage_ratio=float(
            formal_config
            .chain_seam_adaptive_boundary_minimum_common_coverage_ratio
        ),
        adaptive_boundary_shift_penalty=float(
            formal_config.chain_seam_adaptive_boundary_shift_penalty
        ),
    )
    panel_valid_evidence = [
        PanelLocalEvidence(
            corner_x=int(round(layout.panels[index].canvas_offset_x)),
            values=np.asarray(panel["reference_valid"], dtype=bool),
            canvas_width=layout.width,
        )
        for index, panel in enumerate(panel_records)
    ]
    pair_costs = baseline_pair_costs(
        baseline_owner,
        nominal_boundaries,
        corridor_width_pixels=chain_config.corridor_width_pixels,
    )
    baseline_chain = solve_adjacent_panel_chain(
        panel_valid_evidence,
        nominal_boundaries,
        pair_costs=pair_costs,
        target_valid_mask=full_valid,
        config=chain_config,
    )
    accepted: list[dict[str, object]] = []
    chain_rejected: list[dict[str, object]] = []
    accepted_mask = np.zeros(full_valid.shape, dtype=bool)
    locked = np.full(full_valid.shape, -1, dtype=np.int16)
    chain_result = baseline_chain
    for proposal in sorted(
        proposed,
        key=lambda item: (
            -int(item["support_pair_count"]),
            -float(item["selected"].clarity),
            -int(item["selected_target_pixel_count"]),
            int(item["track_id"]),
        ),
    ):
        proposal = dict(proposal)
        proposal.pop("selected")
        ranked_observations = proposal.pop("ranked_observations")
        selected = None
        target_mask = None
        trial = None
        trial_result = None
        observation_rejections: list[dict[str, object]] = []
        for observation in ranked_observations:
            candidate_mask = observation.target_mask
            overlap = int(
                np.count_nonzero(candidate_mask & accepted_mask)
            )
            overlap_ratio = float(
                overlap / max(1, np.count_nonzero(candidate_mask))
            )
            if (
                overlap_ratio
                > lock_config.maximum_accepted_target_overlap_ratio
            ):
                observation_rejections.append(
                    {
                        "candidate_id": int(
                            observation.candidate.candidate_id
                        ),
                        "panel_index": int(observation.panel_index),
                        "frame_id": int(observation.frame_id),
                        "reason": "accepted_whole_object_target_overlap",
                        "accepted_target_overlap_ratio": overlap_ratio,
                    }
                )
                continue
            if np.any(candidate_mask & ~full_valid):
                observation_rejections.append(
                    {
                        "candidate_id": int(
                            observation.candidate.candidate_id
                        ),
                        "panel_index": int(observation.panel_index),
                        "frame_id": int(observation.frame_id),
                        "reason": "whole_object_mask_escapes_formal_target",
                    }
                )
                continue
            candidate_trial = locked.copy()
            if np.any(
                candidate_mask
                & (candidate_trial >= 0)
                & (candidate_trial != observation.panel_index)
            ):
                observation_rejections.append(
                    {
                        "candidate_id": int(
                            observation.candidate.candidate_id
                        ),
                        "panel_index": int(observation.panel_index),
                        "frame_id": int(observation.frame_id),
                        "reason": "whole_object_lock_conflicts_with_prior_owner",
                    }
                )
                continue
            candidate_trial[candidate_mask] = np.int16(
                observation.panel_index
            )
            try:
                candidate_result = solve_adjacent_panel_chain(
                    panel_valid_evidence,
                    nominal_boundaries,
                    pair_costs=pair_costs,
                    target_valid_mask=full_valid,
                    locked_owner_panel_index=candidate_trial,
                    config=chain_config,
                )
            except RuntimeError as exc:
                observation_rejections.append(
                    {
                        "candidate_id": int(
                            observation.candidate.candidate_id
                        ),
                        "panel_index": int(observation.panel_index),
                        "frame_id": int(observation.frame_id),
                        "reason": (
                            "closed_monotone_chain_rejected_whole_mask"
                        ),
                        "chain_error": str(exc),
                    }
                )
                continue
            selected = observation
            target_mask = candidate_mask
            trial = candidate_trial
            trial_result = candidate_result
            break
        if (
            selected is None
            or target_mask is None
            or trial is None
            or trial_result is None
        ):
            chain_rejected.append(
                {
                    **proposal,
                    "reason": "no_chain_feasible_real_panel_observation",
                    "observation_rejections": observation_rejections,
                }
            )
            continue
        locked = trial
        chain_result = trial_result
        accepted_mask |= target_mask
        accepted.append(
            {
                **proposal,
                "selected_candidate_id": int(
                    selected.candidate.candidate_id
                ),
                "selected_panel_index": int(selected.panel_index),
                "selected_frame_id": int(selected.frame_id),
                "selected_target_bbox_xywh": _bbox(target_mask),
                "selected_target_pixel_count": int(
                    np.count_nonzero(target_mask)
                ),
                "alternative_observation_rejections": (
                    observation_rejections
                ),
                "selected_observation": selected,
                "single_clear_panel_owner": True,
                "closed_monotone_chain_pass": True,
            }
        )

    diagnostic = full_image.copy()
    label_image = np.zeros(full_valid.shape, dtype=np.uint16)
    serializable_accepted: list[dict[str, object]] = []
    for label, item in enumerate(accepted, start=1):
        observation = item.pop("selected_observation")
        mask = observation.target_mask
        diagnostic[mask] = observation.target_image_bgr[mask]
        label_image[mask] = np.uint16(label)
        serializable_accepted.append(
            {
                **item,
                "label": label,
                "selected_observation": observation.audit,
                "mask_mapping_only_existing_panel_inverse_map": True,
                "panel_native_map": "formal_reference_plane_inverse_map",
                "rgb_sampling": "same_existing_inverse_map_nearest_real_rgb",
            }
        )
    crop = renderer["crop"]
    crop_x, crop_y, crop_width, crop_height = (
        int(crop[name]) for name in ("x", "y", "width", "height")
    )
    baseline_crop = full_image[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    diagnostic_crop = diagnostic[
        crop_y : crop_y + crop_height,
        crop_x : crop_x + crop_width,
    ]
    comparison = np.hstack((baseline_crop, diagnostic_crop))
    for item in serializable_accepted:
        x, y, width, height = item["selected_target_bbox_xywh"]
        x -= crop_x
        y -= crop_y
        if x + width <= 0 or x >= crop_width or y + height <= 0 or y >= crop_height:
            continue
        cv2.rectangle(
            comparison,
            (x + crop_width, y),
            (x + width - 1 + crop_width, y + height - 1),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            comparison,
            (
                f"L{item['label']} P{item['selected_panel_index']} "
                f"F{item['selected_frame_id']}"
            ),
            (x + crop_width, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    diagnostic_path = output / "panel_native_object_lock.png"
    comparison_path = output / "panel_native_object_lock_before_after.png"
    labels_output_path = output / "panel_native_object_lock_labels.png"
    contact_path = output / "selected_panel_fastsam_contact_sheet.jpg"
    audit_path = output / "panel_native_object_lock_audit.json"
    if not cv2.imwrite(str(diagnostic_path), diagnostic_crop):
        raise RuntimeError("Could not write panel-native diagnostic")
    if not cv2.imwrite(str(comparison_path), comparison):
        raise RuntimeError("Could not write panel-native comparison")
    if not cv2.imwrite(
        str(labels_output_path),
        label_image[
            crop_y : crop_y + crop_height,
            crop_x : crop_x + crop_width,
        ],
    ):
        raise RuntimeError("Could not write panel-native label image")
    if not cv2.imwrite(str(contact_path), _contact_sheet(panel_records)):
        raise RuntimeError("Could not write selected-panel contact sheet")

    map_rejections = Counter(
        str(item["rejection_reason"])
        for item in observation_audits
        if not item["accepted"]
    )
    track_rejections = Counter(item["reason"] for item in rejected_tracks)
    chain_rejections = Counter(item["reason"] for item in chain_rejected)
    audit = {
        "schema": "panel-native-fastsam-whole-object-lock-diagnostic/v1",
        "formal_output_modified": False,
        "formal_renderer_connected": False,
        "first_fixed_gate_only": True,
        "panel_source_count": len(panel_records),
        "panel_sources": [
            {
                "panel_index": int(item["panel_index"]),
                "frame_id": int(item["frame_id"]),
                "rgbd_candidate_count": int(item["candidate_count"]),
                "whole_map_observation_count": len(item["observations"]),
                "inverse_mesh_audit_matched_formal": True,
                "panel_native_map": "formal_reference_plane_inverse_map",
            }
            for item in panel_records
        ],
        "raw_polygon_count": raw_polygon_count,
        "rgbd_candidate_count": sum(
            len(value) for value in candidates_by_panel
        ),
        "whole_map_observation_count": len(observation_by_candidate),
        "world_track_count": len(tracks),
        "two_view_track_proposal_count": len(proposed),
        "accepted_whole_object_lock_count": len(serializable_accepted),
        "accepted_whole_object_pixel_count": int(
            np.count_nonzero(accepted_mask)
        ),
        "fixed_map_gate_rejection_counts": dict(map_rejections),
        "identity_merge_split_rejection_counts": dict(track_rejections),
        "chain_rejection_counts": dict(chain_rejections),
        "accepted_whole_object_locks": serializable_accepted,
        "identity_rejected_tracks": rejected_tracks,
        "chain_rejected_tracks": chain_rejected,
        "observation_audits": observation_audits,
        "baseline_chain_audit": baseline_chain.audit,
        "final_chain_audit": chain_result.audit,
        "constraints": {
            **{
                name: getattr(lock_config, name)
                for name in lock_config.__dataclass_fields__
            },
            "selected_panel_sources_only": True,
            "panel_native_map": "formal_reference_plane_inverse_map",
            "minimum_view_count": 2,
            "rgbd_world_role": "identity_and_merge_split_rejection_only",
            "mask_union_used": False,
            "translation_used": False,
            "affine_used": False,
            "new_warp_used": False,
            "pose_interpolation_used": False,
            "hole_fill_used": False,
            "generated_color_used": False,
            "multiband_used": False,
            "manual_roi_used": False,
            "manual_frame_selection_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "files": {
            "diagnostic": diagnostic_path.name,
            "before_after": comparison_path.name,
            "labels": labels_output_path.name,
            "selected_panel_contact_sheet": contact_path.name,
        },
    }
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(audit_path)
    print(
        json.dumps(
            {
                "raw_polygon_count": audit["raw_polygon_count"],
                "rgbd_candidate_count": audit["rgbd_candidate_count"],
                "whole_map_observation_count": audit[
                    "whole_map_observation_count"
                ],
                "world_track_count": audit["world_track_count"],
                "two_view_track_proposal_count": audit[
                    "two_view_track_proposal_count"
                ],
                "accepted_whole_object_lock_count": audit[
                    "accepted_whole_object_lock_count"
                ],
                "elapsed_seconds": audit["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
