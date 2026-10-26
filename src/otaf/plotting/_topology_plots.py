from __future__ import annotations

__author__ = "Kramer84"
__all__ = ["generate_topological_tikz"]
import numpy as np
from beartype.typing import Any, Dict, List, Tuple

from ._color_palettes import color_palette_2


def generate_topological_tikz(
    data: Dict[str, Any],
    part_positions: Dict[Any, np.ndarray],
    feature_positions: Dict[str, Tuple[float, float]],
    color_palette: List[str] = color_palette_2,
    R_part: float = 15,
    r_feat: float = 2,
    scale: float = 0.15,
) -> str:
    """
    Generate the raw TikZ LaTeX code to visualize a system topology diagram.

    Identifies connected feature components to map colors, initializes the TikZ
    canvas settings, draws assembly parts as large circles, embeds child features
    along their outer perimeters, and renders labeled directional loops representing
    geometric constraints between features.

    Parameters
    ----------
    data : Dict[str, Any]
        The structured dictionary tracking part properties, feature nodes, and loop pathways.
    part_positions : Dict[Any, np.ndarray]
        A dictionary mapping part identifiers to their global (X, Y) layout center coordinates.
    feature_positions : Dict[str, Tuple[float, float]]
        A dictionary mapping unique feature codes to their calculated (X, Y) positions.
    color_palette : List[str]
        A list of hex color string specs used to format the visual elements dynamically.
    R_part : float, default=15
        The base physical radius assigned to the primary part boundary circles.
    r_feat : float, default=2
        The base physical radius allocated for individual nested feature nodes.
    scale : float, default=0.15
        A scalar factor used to shrink or expand the coordinate dimensions for LaTeX rendering.

    Returns
    -------
    str
        The uncompiled, multiline raw TikZ string ready to be placed in a LaTeX document.

    Raises
    ------
    ImportError
        If the networkx library is not installed in the execution environment.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "The 'networkx' library is required to calculate graph layouts."
        )
    parts = data.get("PARTS", {})
    loops = data.get("LOOPS", {}).get("COMPATIBILITY", {})
    G_feat = nx.Graph()
    for p_id, features in parts.items():
        for f_id in features.keys():
            G_feat.add_node(f"P{p_id}{f_id}")
    for p_id, features in parts.items():
        for f_id, f_data in features.items():
            start_node = f"P{p_id}{f_id}"
            for inter in f_data.get("INTERACTIONS", []):
                end_node = f"P{inter[1]}{inter[2]}"
                G_feat.add_edge(start_node, end_node)
    feature_color_map = {}
    color_index = 0
    for component in nx.connected_components(G_feat):
        for node in component:
            feature_color_map[node] = color_index
        color_index += 1
    loop_color_offset = color_index
    output = []
    output.append("\\begin{tikzpicture}[>=stealth]")
    output.append("  %% Dynamic Palette Definitions")
    for i in range(min(len(color_palette), color_index + len(loops))):
        hex_val = color_palette[i].lstrip("#").upper()
        output.append(f"  \\definecolor{{pal_{i}}}{{HTML}}{{{hex_val}}}")
    output.append("  \\tikzstyle{part_circle} = [draw, fill=blue!5, thick]")
    scaled_f_size = r_feat * 2 * scale
    output.append(
        "  \\tikzstyle{feat_node} = [circle, draw, inner sep=0pt, minimum size="
        + f"{scaled_f_size:.2f}"
        + "cm, font=\\scriptsize]"
    )
    scaled_R_part = R_part * scale
    for p_id, features in parts.items():
        output.append(f"\n  %% Part {p_id}")
        px, py = part_positions[p_id]
        px_scaled, py_scaled = (px * scale, py * scale)
        output.append(
            f"  \\draw[part_circle] ({px_scaled:.3f}, {py_scaled:.3f}) circle ({scaled_R_part:.3f});"
        )
        output.append(
            f"  \\node[font=\\bfseries\\large, gray] at ({px_scaled:.3f}, {py_scaled:.3f}) {{Part {p_id}}};"
        )
        for f_id, f_data in features.items():
            node_id = f"P{p_id}{f_id}"
            fx, fy = feature_positions[node_id]
            fx_scaled, fy_scaled = (fx * scale, fy * scale)
            is_perfect = "PERFECT" in f_data.get("CONSTRAINTS_D", [])
            border_color = (
                "black"
                if is_perfect
                else "pal_" + str(feature_color_map[node_id] % len(color_palette))
            )
            c_idx = feature_color_map[node_id] % len(color_palette)
            output.append(
                f"  \\node[feat_node, draw={border_color}, fill=pal_{c_idx}!30] ({node_id}) at ({fx_scaled:.3f}, {fy_scaled:.3f}) {{{f_id.upper()}}};"
            )
    interaction_labels = {}
    for p_id, features in parts.items():
        for f_id, f_data in features.items():
            constraint_g = f_data.get("CONSTRAINTS_G", [""])[0]
            label = constraint_g.lower() if constraint_g != "FLOATING" else ""
            if label:
                for inter in f_data.get("INTERACTIONS", []):
                    t_p, t_f = (inter[1], inter[2])
                    start_node = f"P{p_id}{f_id}"
                    end_node = f"P{t_p}{t_f}"
                    pair = tuple(sorted([start_node, end_node]))
                    interaction_labels[pair] = label
    output.append("\n  %% Loops")
    edge_bends = {}
    drawn_labels = set()
    for l_idx, (l_id, path_str) in enumerate(loops.items()):
        c_idx = (loop_color_offset + l_idx) % len(color_palette)
        color = f"pal_{c_idx}"
        nodes = [n[:3] for n in path_str.split(" -> ")]
        if len(nodes) > 1 and nodes[0] != nodes[-1]:
            nodes.append(nodes[0])
        for i in range(len(nodes) - 1):
            u = nodes[i]
            v = nodes[i + 1]
            if u in feature_positions and v in feature_positions:
                pair_dir = (u, v)
                if pair_dir not in edge_bends:
                    edge_bends[pair_dir] = 0
                else:
                    edge_bends[pair_dir] += 10
                part_u = u[1:-1]
                part_v = v[1:-1]
                if part_u == part_v:
                    bend_val = 50 + edge_bends[pair_dir]
                    bend_str = f"bend right={bend_val}, looseness=1.5"
                else:
                    bend_val = 15 + edge_bends[pair_dir]
                    bend_str = f"bend right={bend_val}"
                pair_undir = tuple(sorted([u, v]))
                label = interaction_labels.get(pair_undir, "")
                if label and pair_undir not in drawn_labels:
                    node_str = f" node[midway, sloped, above, font=\\tiny, text=black] {{{label}}}"
                    drawn_labels.add(pair_undir)
                else:
                    node_str = ""
                output.append(
                    f"  \\path[draw, ->, {color}, ultra thick] ({u}) to[{bend_str}] {node_str} ({v});"
                )
    output.append("\\end{tikzpicture}")
    return "\n".join(output)
