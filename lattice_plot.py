"""Plotting utilities for :mod:`lattice`."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from model import Model


FloatArray = NDArray[np.float64]
_FIGURE_DIR = Path(__file__).with_name("figure")
_DPI = 220


def first_brillouin_zone(b1: FloatArray, b2: FloatArray) -> FloatArray:
    """Return the counterclockwise vertices of a two-dimensional first BZ."""

    reduced_b1, reduced_b2 = _gauss_reduce(b1, b2)
    bound = 2.0 * (np.linalg.norm(reduced_b1) + np.linalg.norm(reduced_b2))
    polygon = np.array(
        [
            [-bound, -bound],
            [bound, -bound],
            [bound, bound],
            [-bound, bound],
        ],
        dtype=np.float64,
    )

    reciprocal_vectors = [
        m * reduced_b1 + n * reduced_b2
        for m in range(-2, 3)
        for n in range(-2, 3)
        if m != 0 or n != 0
    ]
    reciprocal_vectors.sort(key=lambda vector: float(vector @ vector))
    for vector in reciprocal_vectors:
        polygon = _clip_half_plane(
            polygon,
            normal=vector,
            limit=0.5 * float(vector @ vector),
        )

    polygon = _remove_duplicate_vertices(polygon)
    if polygon.shape[0] < 3:
        raise RuntimeError("Failed to construct the first Brillouin zone.")
    if _polygon_signed_area(polygon) < 0.0:
        polygon = polygon[::-1].copy()
    polygon = np.ascontiguousarray(polygon)
    polygon.setflags(write=False)
    return polygon


def save_lattice_figures(
    lattice: Model,
    *,
    cells: int,
) -> tuple[Path, Path]:
    """Save real-space lattice and first-BZ figures."""

    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 1:
        raise ValueError("'cells' must be a positive integer.")
    try:
        from matplotlib import colormaps
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError("Plotting requires Matplotlib.") from error

    _FIGURE_DIR.mkdir(exist_ok=True)
    lattice_path = _FIGURE_DIR / f"{lattice.name}_lattice.png"
    bz_path = _FIGURE_DIR / f"{lattice.name}_brillouin_zone.png"
    colors = _sublattice_colors(lattice.n_orbitals, colormaps)

    figure = Figure(figsize=(6.0, 6.0), layout="constrained")
    FigureCanvasAgg(figure)
    _draw_real_space_lattice(figure.subplots(), lattice, cells, colors)
    figure.savefig(lattice_path, dpi=_DPI)
    figure.clear()

    figure = Figure(figsize=(6.0, 6.0), layout="constrained")
    FigureCanvasAgg(figure)
    _draw_brillouin_zone(figure.subplots(), lattice)
    figure.savefig(bz_path, dpi=_DPI)
    figure.clear()

    return lattice_path, bz_path


def _gauss_reduce(vector1: FloatArray, vector2: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Lagrange-reduce a two-dimensional lattice basis."""

    reduced1 = vector1.copy()
    reduced2 = vector2.copy()
    for _ in range(128):
        if reduced2 @ reduced2 < reduced1 @ reduced1:
            reduced1, reduced2 = reduced2, reduced1
        multiple = int(np.rint((reduced1 @ reduced2) / (reduced1 @ reduced1)))
        if multiple == 0:
            return reduced1, reduced2
        reduced2 = reduced2 - multiple * reduced1
    raise RuntimeError("Reciprocal-lattice reduction did not converge.")


def _clip_half_plane(
    polygon: FloatArray,
    *,
    normal: FloatArray,
    limit: float,
) -> FloatArray:
    """Clip a counterclockwise polygon to ``point . normal <= limit``."""

    if polygon.size == 0:
        return polygon
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(limit))
    clipped: list[FloatArray] = []
    previous = polygon[-1]
    previous_value = float(previous @ normal - limit)
    previous_inside = previous_value <= tolerance

    for current in polygon:
        current_value = float(current @ normal - limit)
        current_inside = current_value <= tolerance
        if current_inside != previous_inside:
            fraction = previous_value / (previous_value - current_value)
            clipped.append(previous + fraction * (current - previous))
        if current_inside:
            clipped.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside

    if not clipped:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(clipped, dtype=np.float64)


def _remove_duplicate_vertices(polygon: FloatArray) -> FloatArray:
    if polygon.shape[0] < 2:
        return polygon
    scale = max(1.0, float(np.max(np.linalg.norm(polygon, axis=1))))
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    vertices = [polygon[0]]
    for vertex in polygon[1:]:
        if np.linalg.norm(vertex - vertices[-1]) > tolerance:
            vertices.append(vertex)
    if len(vertices) > 1 and np.linalg.norm(vertices[0] - vertices[-1]) <= tolerance:
        vertices.pop()
    return np.asarray(vertices, dtype=np.float64)


def _polygon_signed_area(polygon: FloatArray) -> float:
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(x @ np.roll(y, -1) - y @ np.roll(x, -1))


def _sublattice_colors(n_orbitals: int, colormaps: object) -> list[object]:
    if n_orbitals <= 10:
        color_map = colormaps["tab10"]
        denominator = 10
    elif n_orbitals <= 20:
        color_map = colormaps["tab20"]
        denominator = 20
    else:
        color_map = colormaps["hsv"]
        denominator = n_orbitals
    return [color_map(index / denominator) for index in range(n_orbitals)]


def _draw_real_space_lattice(
    axes: object,
    lattice: Model,
    cells: int,
    colors: list[object],
) -> None:
    lower = -cells
    upper = cells + 1
    for index in range(lower, upper + 1):
        start = index * lattice.a1 + lower * lattice.a2
        end = index * lattice.a1 + upper * lattice.a2
        axes.plot(*np.vstack((start, end)).T, color="0.78", linewidth=0.7, zorder=1)
        start = lower * lattice.a1 + index * lattice.a2
        end = upper * lattice.a1 + index * lattice.a2
        axes.plot(*np.vstack((start, end)).T, color="0.78", linewidth=0.7, zorder=1)

    indices = np.arange(lower, upper, dtype=np.float64)
    n1, n2 = np.meshgrid(indices, indices, indexing="ij")
    origins = n1.reshape(-1, 1) * lattice.a1 + n2.reshape(-1, 1) * lattice.a2
    for alpha, (tau_alpha, color) in enumerate(zip(lattice.tau_cart, colors)):
        positions = origins + tau_alpha
        axes.scatter(
            positions[:, 0],
            positions[:, 1],
            s=48,
            marker="o",
            color=color,
            edgecolor="0.15",
            linewidth=0.6,
            label=rf"$\alpha={alpha}$",
            zorder=3,
        )

    origin = np.zeros(2)
    _draw_vector(axes, origin, lattice.a1, r"$\mathbf{a}_1$")
    _draw_vector(axes, origin, lattice.a2, r"$\mathbf{a}_2$")
    axes.set_xlabel(r"$x$")
    axes.set_ylabel(r"$y$")
    axes.set_title(f"{lattice.name}: lattice")
    axes.set_aspect("equal", adjustable="datalim")
    axes.legend(frameon=False, ncols=min(4, lattice.n_orbitals))
    axes.margins(0.08)


def _draw_brillouin_zone(axes: object, lattice: Model) -> None:
    polygon = lattice.first_brillouin_zone()
    closed_polygon = np.vstack((polygon, polygon[0]))
    axes.fill(polygon[:, 0], polygon[:, 1], color="0.88", zorder=1)
    axes.plot(
        closed_polygon[:, 0],
        closed_polygon[:, 1],
        color="0.1",
        linewidth=1.5,
        label="Wigner--Seitz BZ",
        zorder=2,
    )

    parallelogram = np.array(
        [np.zeros(2), lattice.b1, lattice.b1 + lattice.b2, lattice.b2, np.zeros(2)]
    )
    axes.plot(
        parallelogram[:, 0],
        parallelogram[:, 1],
        color="C1",
        linestyle="--",
        linewidth=1.6,
        label=r"$\mathbf{b}_1,\mathbf{b}_2$ cell",
        zorder=4,
    )

    reduced_b1, reduced_b2 = _gauss_reduce(lattice.b1, lattice.b2)
    candidates = [
        m * reduced_b1 + n * reduced_b2
        for m in range(-2, 3)
        for n in range(-2, 3)
        if m != 0 or n != 0
    ]
    active_vectors = []
    for vector in candidates:
        boundary_value = 0.5 * float(vector @ vector)
        tolerance = 1.0e-10 * max(1.0, abs(boundary_value))
        if np.min(np.abs(polygon @ vector - boundary_value)) <= tolerance:
            active_vectors.append(vector)
    reciprocal_points = np.asarray(active_vectors, dtype=np.float64)
    axes.scatter(
        reciprocal_points[:, 0],
        reciprocal_points[:, 1],
        s=18,
        color="0.35",
        marker="o",
        zorder=3,
    )
    axes.scatter([0.0], [0.0], s=32, color="0.05", marker="o", zorder=4)
    axes.annotate(r"$\Gamma$", (0.0, 0.0), xytext=(5, 5), textcoords="offset points")
    _draw_vector(axes, np.zeros(2), lattice.b1, r"$\mathbf{b}_1$")
    _draw_vector(axes, np.zeros(2), lattice.b2, r"$\mathbf{b}_2$")
    axes.set_xlabel(r"$k_x$")
    axes.set_ylabel(r"$k_y$")
    axes.set_title(f"{lattice.name}: first Brillouin zone")
    axes.set_aspect("equal", adjustable="datalim")
    axes.legend(frameon=False)
    axes.margins(0.12)


def _draw_vector(axes: object, origin: FloatArray, vector: FloatArray, label: str) -> None:
    axes.annotate(
        "",
        xy=origin + vector,
        xytext=origin,
        arrowprops={"arrowstyle": "->", "color": "0.1", "linewidth": 1.2},
        zorder=5,
    )
    axes.annotate(label, origin + vector, xytext=(4, 4), textcoords="offset points")
