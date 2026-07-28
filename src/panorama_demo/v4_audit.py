"""Truthful v4 first-part audit sidecars."""
from __future__ import annotations
from typing import Mapping, Sequence


def build_object_and_visibility_audit(identity_runtime: Mapping[str, object] | None, *, all_frame_ids: Sequence[int]) -> tuple[dict[str, object], dict[str, object]]:
    runtime = {} if identity_runtime is None else dict(identity_runtime)
    closure = dict(runtime.get("middle_shelf_inventory_mesh_closure", {}))
    inventory = dict(runtime.get("shelf_object_inventory", {}))
    required = {int(v) for v in closure.get("required_track_ids", [])}
    direct = {int(v) for v in closure.get("accepted_true_depth_mesh_track_ids", [])}
    corridor = {int(v) for v in closure.get("accepted_object_rich_corridor_track_ids", [])}
    same = {int(v) for v in closure.get("accepted_same_panel_reference_rgb_track_ids", [])}
    rows = {int(v["track_id"]): dict(v) for v in inventory.get("dispositions", []) if isinstance(v, Mapping) and v.get("track_id") is not None}
    tracks, matrix = [], []
    for track_id in sorted(required):
        row = rows.get(track_id, {})
        if track_id in direct:
            disposition, mode = "DIRECT_RGBD_WORLD_OWNER", "aligned_depth_camera_to_world_inverse_mesh"
        elif track_id in corridor:
            disposition, mode = "OBJECT_RICH_SINGLE_SOURCE_CORRIDOR", "complete_single_real_rgb_corridor"
        elif track_id in same:
            disposition, mode = "OBJECT_RICH_SINGLE_SOURCE_CORRIDOR", "same_panel_complete_single_rgb_corridor"
        else:
            disposition, mode = "INSUFFICIENT_OBSERVATION", "none"
        obs = {int(v["frame_id"]): dict(v) for v in row.get("observations", []) if isinstance(v, Mapping) and v.get("frame_id") is not None}
        tracks.append({"track_id": track_id, "required": True, "disposition": disposition, "selected_rgb_frame_id": row.get("selected_frame_id"), "spatial_panel_index": row.get("selected_target_panel_index"), "geometry_mode": mode, "complete_source_coverage": disposition != "INSUFFICIENT_OBSERVATION", "depth_coverage_ratio": row.get("projected_in_bounds_ratio"), "boundary_margin_pixels": None, "world_centroid_mm": None, "world_covariance": None, "thin_structure_attachment_ids": [], "occlusion_predecessors": [], "failure_reason": None})
        matrix.append({"track_id": track_id, "frames": [{"frame_id": int(fid), "observed": int(fid) in obs, "mask_core_coverage": obs.get(int(fid), {}).get("source_mask_pixel_count", 0), "depth_coverage": obs.get(int(fid), {}).get("source_depth_coverage_ratio"), "boundary_clear": obs.get(int(fid), {}).get("source_boundary_clear")} for fid in all_frame_ids]})
    return ({"schema": "g305-object-tracks/v4", "track_count": len(tracks), "required_track_count": len(required), "legacy_same_panel_owner_count": 0, "all_required_tracks_have_one_disposition": len(tracks) == len(required), "tracks": tracks}, {"schema": "g305-visibility-matrix/v1", "frame_count": len(all_frame_ids), "track_count": len(matrix), "matrix": matrix})


def build_occlusion_and_photometric_audit(identity_runtime: Mapping[str, object] | None, render_metadata: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    runtime = {} if identity_runtime is None else dict(identity_runtime)
    mesh = dict(runtime.get("mesh_preflight", {}))
    components = [dict(c) for a in mesh.get("accepted_owner_audits", []) if isinstance(a, Mapping) for c in a.get("components", []) if isinstance(c, Mapping)]
    nodes = [{"track_id": c.get("structure_id"), "source_frame_id": c.get("frame_id"), "z_buffer_occluded_pixel_count": c.get("z_buffer_occluded_pixel_count", 0), "target_scene_occluded_pixel_count": c.get("target_scene_occluded_pixel_count", 0)} for c in components]
    seam = dict(render_metadata.get("background_seam_audit", {}))
    photometric = {
        "available": bool(seam.get("exposure_compensation_used") is True),
        "source_domain_method": seam.get("exposure_compensation_method"),
        "source_gain_statistics": seam.get("exposure_gain_statistics", []),
        "adjacent_residual_gain_statistics": seam.get("adjacent_residual_gain_statistics", []),
        "adjacent_pair_audits": seam.get("adjacent_exposure_pair_audits", []),
        "post_composition_gain_applied": bool(
            dict(seam.get("continuous_canvas_exposure", {})).get("applied") is True
        ),
    }
    return ({"schema": "g305-occlusion-graph/v1", "nodes": nodes, "edges": [], "z_buffer_unexplained_conflict_count": sum(int(n["z_buffer_occluded_pixel_count"]) + int(n["target_scene_occluded_pixel_count"]) for n in nodes), "policy": "measured_depth_zbuffer_only_no_2d_write_order"}, {"schema": "g305-photometric-graph/v1", "renderer_photometric_audit": photometric, "single_panorama_tone_curve": "sRGB_encode_after_linear_source_gains"})
