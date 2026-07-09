from __future__ import annotations

__author__ = "Kramer84"
__all__ = [
    "calculate_graph_layout",
]

import numpy as np
from beartype.typing import Any, Dict, Tuple

def calculate_graph_layout(
    data: Dict[str, Any],
    R_part: float = 15.0,
    r_feat: float = 2.0,
    d_feat: float = 1.5,
    margin: float = 1.5,
    part_spacing: float = 45.0,
    seed: int = 42,
) -> Tuple[Dict[Any, np.ndarray], Dict[str, Tuple[float, float]]]:
    """Compute a hierarchical macro/micro graph layout.

    Positions high-level components (parts) using a force-directed
    layout, initializes sub-features along a radial arc centered on
    their parent part, and iteratively reorders the features by
    orientation to minimize edge crossings.

    Parameters
    ----------
    data : Dict[str, Any]
        A structured dictionary containing part entities, features,
        and their topological interactions.
    R_part : float, optional
        The outer boundary radius of the parent part node
        (the default is 15.0).
    r_feat : float, optional
        The bounding radius allocated for individual child feature
        nodes (the default is 2.0).
    d_feat : float, optional
        The minimum spacing clearance distance between adjacent
        features on the arc (the default is 1.5).
    margin : float, optional
        The offset padding gap subtracted from the `R_part` to
        establish the feature arc (the default is 1.5).
    part_spacing : float, optional
        The scale and expansion modifier configuring the distance
        between part centers (the default is 45.0).
    seed : int, optional
        The seed integer initializing the spring layout random
        number generator (the default is 42).

    Returns
    -------
    part_positions : Dict[Any, np.ndarray]
        A mapping from part identifiers to their calculated (X, Y)
        layout positions.
    feature_positions : Dict[str, Tuple[float, float]]
        A mapping from unique feature identifiers to their optimized
        (X, Y) coordinates.

    Raises
    ------
    ImportError
        If the ``networkx`` graph processing library is not accessible
        in the current environment.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "The 'networkx' library is required to calculate graph layouts."
        )
    parts = data.get("PARTS", {})
    G_macro = nx.Graph()
    for p_id in parts.keys():
        G_macro.add_node(p_id)
    for p_id, features in parts.items():
        for f_id, f_data in features.items():
            for inter in f_data.get("INTERACTIONS", []):
                target_p = inter[1]
                if target_p in parts:
                    if G_macro.has_edge(p_id, target_p):
                        G_macro[p_id][target_p]["weight"] += 0.1
                    else:
                        G_macro.add_edge(p_id, target_p, weight=0.1)
    optimal_dist = 2.5 * R_part / (part_spacing / 2.0)
    part_positions = nx.spring_layout(
        G_macro, k=optimal_dist * 2.5, scale=part_spacing, iterations=100, seed=seed
    )
    R_arc = R_part - r_feat - margin
    chord_len = 2 * r_feat + d_feat
    d_theta = 2 * np.arcsin(chord_len / (2 * R_arc))
    feature_positions = {}
    part_base_angles = {}
    feature_order = {}
    for p_id, features in parts.items():
        px, py = part_positions[p_id]
        interacting_vectors = []
        for neighbor in G_macro.neighbors(p_id):
            nx_pos, ny_pos = part_positions[neighbor]
            interacting_vectors.append([nx_pos - px, ny_pos - py])
        if interacting_vectors:
            avg_vec = np.mean(interacting_vectors, axis=0)
            base_angle = np.arctan2(avg_vec[1], avg_vec[0])
        else:
            base_angle = 0
        part_base_angles[p_id] = base_angle
        f_ids = list(features.keys())
        feature_order[p_id] = f_ids
        n_features = len(f_ids)
        start_angle = base_angle - (n_features - 1) * d_theta / 2
        for idx, f_id in enumerate(f_ids):
            angle = start_angle + idx * d_theta
            fx = px + R_arc * np.cos(angle)
            fy = py + R_arc * np.sin(angle)
            feature_positions[f"P{p_id}{f_id}"] = (fx, fy)
    for _ in range(5):
        for p_id, features in parts.items():
            px, py = part_positions[p_id]
            base_angle = part_base_angles[p_id]
            f_ids = feature_order[p_id]
            target_angles = []
            for f_id in f_ids:
                f_data = features[f_id]
                targets = []
                for inter in f_data.get("INTERACTIONS", []):
                    t_p, t_f = (inter[1], inter[2])
                    t_node = f"P{t_p}{t_f}"
                    if t_node in feature_positions:
                        targets.append(feature_positions[t_node])
                if targets:
                    avg_tx = np.mean([t[0] for t in targets])
                    avg_ty = np.mean([t[1] for t in targets])
                    t_ang = np.arctan2(avg_ty - py, avg_tx - px)
                    rel_ang = np.arctan2(
                        np.sin(t_ang - base_angle), np.cos(t_ang - base_angle)
                    )
                else:
                    rel_ang = 0
                target_angles.append((rel_ang, f_id))
            target_angles.sort(key=lambda x: x[0])
            new_f_ids = [x[1] for x in target_angles]
            feature_order[p_id] = new_f_ids
            n_features = len(new_f_ids)
            start_angle = base_angle - (n_features - 1) * d_theta / 2
            for idx, f_id in enumerate(new_f_ids):
                angle = start_angle + idx * d_theta
                fx = px + R_arc * np.cos(angle)
                fy = py + R_arc * np.sin(angle)
                feature_positions[f"P{p_id}{f_id}"] = (fx, fy)
    return (part_positions, feature_positions)
