"""Calculate and plot the local model's zero-field bands."""

from __future__ import annotations

from pathlib import Path
import tomllib
from typing import Final

import numpy as np
from numpy.typing import NDArray

from model import Model, load_model, model_path_from_command_line


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_TWO_PI: Final = 2.0 * np.pi
_FIGURE_DIR: Final = Path(__file__).with_name("figure")
_DATA_DIR: Final = Path(__file__).with_name("data")
_DPI: Final = 220


def _load_band_path(path: Path) -> tuple[list[str], FloatArray, int]:
    with path.open("rb") as stream:
        data = tomllib.load(stream)

    text = data.get("band_path")
    if not isinstance(text, str):
        raise ValueError("'band_path' must be a multiline string.")

    labels = []
    vertices = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 3:
            raise ValueError(f"Band-path line {line_number} must contain three columns.")
        try:
            point = [float(fields[1]), float(fields[2])]
        except ValueError as error:
            raise ValueError(f"Invalid coordinate on band-path line {line_number}.") from error
        if not np.all(np.isfinite(point)):
            raise ValueError(f"Non-finite coordinate on band-path line {line_number}.")
        labels.append(fields[0])
        vertices.append(point)

    if len(vertices) < 2:
        raise ValueError("'band_path' must contain at least two points.")

    n_k = data.get("n_k")
    if isinstance(n_k, bool) or not isinstance(n_k, int) or n_k < 2:
        raise ValueError("'n_k' must be an integer greater than one.")

    return labels, np.asarray(vertices, dtype=np.float64), n_k


def _sample_path(
    vertices: FloatArray,
    model: Model,
    n_k: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    reciprocal = np.column_stack((model.b1, model.b2))
    k_parts = []
    distance_parts = []
    tick_positions = [0.0]
    distance = 0.0

    for index, (start, stop) in enumerate(zip(vertices[:-1], vertices[1:])):
        length = float(np.linalg.norm(reciprocal @ (stop - start)))
        if length == 0.0:
            raise ValueError("Consecutive points in 'band_path' must be different.")

        fraction = np.linspace(0.0, 1.0, n_k)
        k_segment = start + fraction[:, None] * (stop - start)
        distance_segment = distance + fraction * length
        if index:
            k_segment = k_segment[1:]
            distance_segment = distance_segment[1:]

        k_parts.append(k_segment)
        distance_parts.append(distance_segment)
        distance += length
        tick_positions.append(distance)

    return (
        np.concatenate(k_parts),
        np.concatenate(distance_parts),
        np.asarray(tick_positions),
    )


def bloch_hamiltonian(model: Model, k_points: FloatArray) -> ComplexArray:
    """Return H0(k) in the article's atomic Bloch convention."""

    # h_ab(ell) = <0, a | H | ell, b>, so the physical displacement is
    # ell_1*a1 + ell_2*a2 + tau_b - tau_a.
    delta = np.array(
        [
            [hopping.ell1, hopping.ell2]
            + model.tau[hopping.alpha2]
            - model.tau[hopping.alpha1]
            for hopping in model.hoppings
        ]
    )
    phases = np.exp(1j * _TWO_PI * (k_points @ delta.T))
    matrices = np.zeros(
        (k_points.shape[0], model.n_orbitals, model.n_orbitals),
        dtype=np.complex128,
    )
    for index, hopping in enumerate(model.hoppings):
        matrices[:, hopping.alpha1, hopping.alpha2] += (
            hopping.amplitude * phases[:, index]
        )
    return matrices


def _save_data(
    model: Model,
    labels: list[str],
    vertices: FloatArray,
    distance: FloatArray,
    k_points: FloatArray,
    energies: FloatArray,
) -> Path:
    _DATA_DIR.mkdir(exist_ok=True)
    path = _DATA_DIR / f"{model.name}_band.dat"
    path_text = " -> ".join(
        f"{label}({point[0]:.10g},{point[1]:.10g})"
        for label, point in zip(labels, vertices)
    )
    columns = "distance k1 k2 " + " ".join(
        f"E{index}" for index in range(model.n_orbitals)
    )
    header = f"k = k1*b1 + k2*b2\npath = {path_text}\ncolumns: {columns}"
    np.savetxt(
        path,
        np.column_stack((distance, k_points, energies)),
        fmt="%.16e",
        header=header,
    )
    return path


def _plot_bands(
    model: Model,
    labels: list[str],
    tick_positions: FloatArray,
    distance: FloatArray,
    energies: FloatArray,
) -> Path:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError("Plotting requires Matplotlib.") from error

    _FIGURE_DIR.mkdir(exist_ok=True)
    path = _FIGURE_DIR / f"{model.name}_band.png"
    figure = Figure(figsize=(6.0, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()

    axes.plot(distance, energies, color="0.12", linewidth=1.1)
    for position in tick_positions:
        axes.axvline(position, color="0.82", linewidth=0.8, zorder=0)
    tick_labels = [r"$\Gamma$" if label.lower() in {"g", "gamma"} else label for label in labels]
    axes.set_xticks(tick_positions, tick_labels)
    axes.set_xlim(distance[0], distance[-1])
    axes.set_ylabel(r"$E$")
    axes.set_title(f"{model.name}: zero-field bands")
    axes.margins(x=0.0)

    figure.savefig(path, dpi=_DPI)
    figure.clear()
    return path


def main() -> None:
    input_path = model_path_from_command_line()
    model = load_model(input_path)
    labels, vertices, n_k = _load_band_path(input_path)
    k_points, distance, tick_positions = _sample_path(vertices, model, n_k)
    energies = np.linalg.eigvalsh(bloch_hamiltonian(model, k_points))
    data_path = _save_data(model, labels, vertices, distance, k_points, energies)
    figure_path = _plot_bands(model, labels, tick_positions, distance, energies)

    print("Real-space Hermiticity check passed.")
    print(f"Saved {data_path}")
    print(f"Saved {figure_path}")


if __name__ == "__main__":
    main()
