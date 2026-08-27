"""Calculate magnetic bands and Hofstadter spectra for the local model."""

from __future__ import annotations

import base64
from contextlib import nullcontext
from dataclasses import dataclass
from fractions import Fraction
from math import ceil, gcd, sqrt
from pathlib import Path
import tomllib
from typing import Final

import numpy as np
from numpy.typing import NDArray

from interactive import save_linked_figure
from model import Model, load_model, model_path_from_command_line


FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
_TWO_PI: Final = 2.0 * np.pi
_FIGURE_DIR: Final = Path(__file__).with_name("figure")
_DATA_DIR: Final = Path(__file__).with_name("data")
_DPI: Final = 220


@dataclass(frozen=True, slots=True)
class BandTask:
    p: int
    q: int
    vertices: FloatArray
    n_k: int
    omp_num_threads: int | None = None


@dataclass(frozen=True, slots=True)
class SpectrumTask:
    flux_min: float
    flux_max: float
    q_max: int
    k_mesh: tuple[int, int]
    k_mesh_q1: tuple[int, int] | None
    energy_window: tuple[float, float] | None = None
    filling_window: tuple[float, float] | None = None
    omp_num_threads: int | None = None


def _integer(
    table: dict[str, object],
    key: str,
    minimum: int | None = None,
) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer.")
    if minimum is not None and value < minimum:
        raise ValueError(f"'{key}' must be an integer >= {minimum}.")
    return value


def _number(table: dict[str, object], key: str) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"'{key}' must be a finite real number.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"'{key}' must be a finite real number.")
    return value


def _optional_window(
    table: dict[str, object],
    lower_key: str,
    upper_key: str,
) -> tuple[float, float] | None:
    """Read an optional pair of finite, strictly ordered bounds."""

    has_lower = lower_key in table
    has_upper = upper_key in table
    if has_lower != has_upper:
        raise ValueError(
            f"'{lower_key}' and '{upper_key}' must be provided together."
        )
    if not has_lower:
        return None

    lower = _number(table, lower_key)
    upper = _number(table, upper_key)
    if upper <= lower:
        raise ValueError(f"Require {lower_key} < {upper_key}.")
    return lower, upper


def _validate_flux(p: int, q: int) -> None:
    if isinstance(p, bool) or not isinstance(p, int):
        raise ValueError("'p' must be an integer.")
    if isinstance(q, bool) or not isinstance(q, int) or q < 1:
        raise ValueError("'q' must be a positive integer.")
    divisor = gcd(abs(p), q)
    if divisor != 1:
        raise ValueError(
            f"p={p} and q={q} must be coprime; use p={p // divisor}, "
            f"q={q // divisor}."
        )


def _parse_path(text: object) -> FloatArray:
    if not isinstance(text, str):
        raise ValueError("'k_path' must be a multiline string.")

    vertices = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 2:
            raise ValueError(f"Hofstadter path line {line_number} needs k1 and k2.")
        try:
            point = [float(fields[0]), float(fields[1])]
        except ValueError as error:
            raise ValueError(
                f"Invalid coordinate on Hofstadter path line {line_number}."
            ) from error
        if not np.all(np.isfinite(point)):
            raise ValueError(
                f"Non-finite coordinate on Hofstadter path line {line_number}."
            )
        vertices.append(point)

    if len(vertices) < 2:
        raise ValueError("'k_path' must contain at least two points.")
    return np.asarray(vertices, dtype=np.float64)


def load_hofstadter_task(path: Path) -> BandTask | SpectrumTask:
    """Read the active ``[hofstadter]`` task from a model file."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)
    root = data.get("hofstadter")
    if not isinstance(root, dict):
        raise ValueError("Missing '[hofstadter]' section.")

    omp_num_threads = (
        _integer(root, "omp_num_threads", 1)
        if "omp_num_threads" in root
        else None
    )
    mode = root.get("mode")
    if mode == "band":
        table = root.get("band")
        if not isinstance(table, dict):
            raise ValueError("Missing '[hofstadter.band]' section.")
        p = _integer(table, "p")
        q = _integer(table, "q", 1)
        _validate_flux(p, q)
        vertices = _parse_path(table.get("k_path"))
        return BandTask(
            p, q, vertices, _integer(table, "n_k", 2),
            omp_num_threads=omp_num_threads,
        )

    if mode == "spectrum":
        table = root.get("spectrum")
        if not isinstance(table, dict):
            raise ValueError("Missing '[hofstadter.spectrum]' section.")
        flux_min = _number(table, "flux_min")
        flux_max = _number(table, "flux_max")
        if flux_max < flux_min:
            raise ValueError("Require flux_min <= flux_max.")

        mesh = table.get("k_mesh")
        if (
            not isinstance(mesh, list)
            or len(mesh) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in mesh
            )
        ):
            raise ValueError("'k_mesh' must contain two positive integers.")

        mesh_q1 = table.get("k_mesh_q1")
        if mesh_q1 is not None:
            if (
                not isinstance(mesh_q1, list)
                or len(mesh_q1) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < mesh[index]
                    for index, value in enumerate(mesh_q1)
                )
            ):
                raise ValueError(
                    "'k_mesh_q1' must contain two integers not smaller than "
                    "the corresponding 'k_mesh' values."
                )
        return SpectrumTask(
            flux_min,
            flux_max,
            _integer(table, "q_max", 1),
            (mesh[0], mesh[1]),
            None if mesh_q1 is None else (mesh_q1[0], mesh_q1[1]),
            _optional_window(table, "energy_min", "energy_max"),
            _optional_window(table, "n_min", "n_max"),
            omp_num_threads=omp_num_threads,
        )

    raise ValueError("'[hofstadter].mode' must be 'band' or 'spectrum'.")


def _as_k_points(k_points: object) -> tuple[FloatArray, bool]:
    points = np.asarray(k_points, dtype=np.float64)
    single_point = points.ndim == 1
    if single_point:
        points = points[None, :]
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] != 2:
        raise ValueError("'k_points' must have shape (2,) or (N_k, 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("'k_points' must contain only finite values.")
    return np.ascontiguousarray(points), single_point


class _MagneticHamiltonianBuilder:
    """Cache the k-independent matrix-element data for one fixed flux."""

    __slots__ = (
        "base_value",
        "column",
        "delta1",
        "delta2",
        "dimension",
        "q",
        "row",
    )

    def __init__(self, model: Model, p: int, q: int) -> None:
        _validate_flux(p, q)
        n_hoppings = model.n_hoppings
        self.dimension = q * model.n_orbitals
        self.q = q
        alpha = np.fromiter(
            (hopping.alpha1 for hopping in model.hoppings),
            dtype=np.intp,
            count=n_hoppings,
        )
        alpha_prime = np.fromiter(
            (hopping.alpha2 for hopping in model.hoppings),
            dtype=np.intp,
            count=n_hoppings,
        )
        ell1 = np.fromiter(
            (hopping.ell1 for hopping in model.hoppings),
            dtype=np.int64,
            count=n_hoppings,
        )
        ell2 = np.fromiter(
            (hopping.ell2 for hopping in model.hoppings),
            dtype=np.int64,
            count=n_hoppings,
        )
        amplitude = np.fromiter(
            (hopping.amplitude for hopping in model.hoppings),
            dtype=np.complex128,
            count=n_hoppings,
        )

        # Wannier90 convention: h_ab(ell) = <0, a | H | ell, b>.
        self.delta1 = (
            ell1 + model.tau[alpha_prime, 0] - model.tau[alpha, 0]
        )
        self.delta2 = (
            ell2 + model.tau[alpha_prime, 1] - model.tau[alpha, 1]
        )
        s_prime = np.arange(q, dtype=np.int64)[None, :]
        s = (s_prime - ell1[:, None]) % q
        self.row = np.asarray(
            s * model.n_orbitals + alpha[:, None],
            dtype=np.int32,
        )
        self.column = np.asarray(
            s_prime * model.n_orbitals + alpha_prime[:, None],
            dtype=np.int32,
        )
        flux = p / q
        translation_phase = np.exp(
            1j
            * _TWO_PI
            * flux
            * (s_prime - s - ell1[:, None])
            * model.tau[alpha_prime, 1][:, None]
        )
        peierls_phase = np.exp(
            1j
            * np.pi
            * flux
            * (
                2.0 * s
                + 2.0 * model.tau[alpha, 0][:, None]
                + self.delta1[:, None]
            )
            * self.delta2[:, None]
        )
        self.base_value = amplitude[:, None] * translation_phase * peierls_phase

    def matrices(self, points: FloatArray) -> ComplexArray:
        """Evaluate all k-dependent matrix elements for one point batch."""

        matrices = np.zeros(
            (points.shape[0], self.dimension, self.dimension),
            dtype=np.complex128,
        )
        k_phase = np.exp(
            1j
            * _TWO_PI
            * (
                points[:, 0, None] * self.delta1[None, :] / self.q
                + points[:, 1, None] * self.delta2[None, :]
            )
        )
        for index in range(self.delta1.size):
            matrices[:, self.row[index], self.column[index]] += (
                k_phase[:, index, None] * self.base_value[index, None, :]
            )
        return matrices


def magnetic_hamiltonian(
    model: Model,
    p: int,
    q: int,
    k_points: object,
) -> ComplexArray:
    """Evaluate the article's explicit oblique-gauge matrix element."""

    points, single_point = _as_k_points(k_points)
    matrices = _MagneticHamiltonianBuilder(model, p, q).matrices(points)
    return matrices[0] if single_point else matrices


def magnetic_energies(
    model: Model,
    p: int,
    q: int,
    k_points: object,
    *,
    batch_size: int = 128,
) -> FloatArray:
    """Return sorted eigenvalues without retaining every magnetic matrix."""

    _validate_flux(p, q)
    points, single_point = _as_k_points(k_points)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("'batch_size' must be a positive integer.")

    dimension = q * model.n_orbitals
    energies = np.empty((points.shape[0], dimension), dtype=np.float64)

    # Bound the temporary stack to roughly 256 MiB.  Previously a fixed batch
    # of 128 could allocate many GiB for a large q before diagonalization began.
    bytes_per_matrix = np.dtype(np.complex128).itemsize * dimension * dimension
    memory_limited_batch = max(1, (256 * 1024**2) // bytes_per_matrix)
    effective_batch = min(batch_size, memory_limited_batch)
    builder = _MagneticHamiltonianBuilder(model, p, q)
    for start in range(0, points.shape[0], effective_batch):
        stop = min(start + effective_batch, points.shape[0])
        energies[start:stop] = np.linalg.eigvalsh(
            builder.matrices(points[start:stop])
        )
    return energies[0] if single_point else energies


def rational_fluxes(
    flux_min: float,
    flux_max: float,
    q_max: int,
) -> list[tuple[int, int]]:
    """List every signed reduced p/q in the closed interval."""

    lower = Fraction(str(flux_min))
    upper = Fraction(str(flux_max))
    fractions = []
    for q in range(1, q_max + 1):
        p_min = (lower.numerator * q + lower.denominator - 1) // lower.denominator
        p_max = (upper.numerator * q) // upper.denominator
        for p in range(p_min, p_max + 1):
            if gcd(p, q) == 1:
                fractions.append((p, q))
    fractions.sort(key=lambda pair: Fraction(pair[0], pair[1]))
    return fractions


def _sample_path(
    vertices: FloatArray,
    reciprocal: FloatArray,
    n_k: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    k_parts = []
    distance_parts = []
    tick_positions = [0.0]
    distance = 0.0

    for index, (start, stop) in enumerate(zip(vertices[:-1], vertices[1:])):
        length = float(np.linalg.norm(reciprocal @ (stop - start)))
        if length == 0.0:
            raise ValueError("Consecutive points in 'k_path' must be different.")
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


def _magnetic_mesh(q: int, shape: tuple[int, int]) -> FloatArray:
    """Sample k1 in [0,1) and the nonredundant k2 interval [0,1/q)."""

    k1 = np.arange(shape[0], dtype=np.float64) / shape[0]
    k2 = np.arange(shape[1], dtype=np.float64) / (q * shape[1])
    mesh1, mesh2 = np.meshgrid(k1, k2, indexing="ij")
    return np.column_stack((mesh1.ravel(), mesh2.ravel()))


def _spectrum_mesh(task: SpectrumTask, q: int) -> tuple[int, int]:
    """Use more momentum points for the broader bands at small q."""

    if task.k_mesh_q1 is None:
        return task.k_mesh
    return tuple(
        max(minimum, ceil(at_q1 / sqrt(q)))
        for minimum, at_q1 in zip(task.k_mesh, task.k_mesh_q1)
    )


def _integer_ticks(lower: float, upper: float, maximum: int = 7) -> FloatArray:
    """Return at most ``maximum`` integer ticks inside a closed interval."""

    first = ceil(lower)
    last = int(np.floor(upper))
    if first > last:
        return np.empty(0, dtype=np.float64)
    step = max(1, ceil((last - first) / max(1, maximum - 1)))
    ticks = np.arange(first, last + 1, step, dtype=np.float64)
    if ticks[-1] != last and ticks.size < maximum:
        ticks = np.append(ticks, float(last))
    return ticks


def _plot_imports() -> tuple[object, object]:
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise ImportError("Plotting requires Matplotlib.") from error
    return Figure, FigureCanvasAgg


def _coordinate_label(point: FloatArray) -> str:
    return rf"$({point[0]:.4g},{point[1]:.4g})$"


def run_band(model: Model, task: BandTask) -> tuple[Path, Path]:
    """Calculate, save, and plot one rational-flux magnetic band structure."""

    reciprocal = np.column_stack((model.b1 / task.q, model.b2))
    k_points, distance, tick_positions = _sample_path(
        task.vertices, reciprocal, task.n_k
    )
    energies = magnetic_energies(model, task.p, task.q, k_points)
    stem = (
        f"{model.name}_hofstadter_p{_number_tag(task.p)}_q{task.q}_band"
    )

    _DATA_DIR.mkdir(exist_ok=True)
    data_path = _DATA_DIR / f"{stem}.dat"
    path_text = " -> ".join(
        f"({point[0]:.10g},{point[1]:.10g})" for point in task.vertices
    )
    columns = "distance k1 k2 " + " ".join(
        f"E{index}" for index in range(task.q * model.n_orbitals)
    )
    header = (
        f"Phi/Phi0 = {task.p}/{task.q}\n"
        "k = k1*P1 + k2*P2, P1 = b1/q, P2 = b2\n"
        f"path = {path_text}\ncolumns: {columns}"
    )
    np.savetxt(
        data_path,
        np.column_stack((distance, k_points, energies)),
        fmt="%.16e",
        header=header,
    )

    Figure, FigureCanvasAgg = _plot_imports()
    _FIGURE_DIR.mkdir(exist_ok=True)
    figure_path = _FIGURE_DIR / f"{stem}.png"
    figure = Figure(figsize=(6.0, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot(distance, energies, color="0.12", linewidth=0.9)
    for position in tick_positions:
        axes.axvline(position, color="0.82", linewidth=0.8, zorder=0)
    axes.set_xticks(
        tick_positions,
        [_coordinate_label(point) for point in task.vertices],
    )
    axes.set_xlim(distance[0], distance[-1])
    axes.set_xlabel(r"$\mathbf{k}=k_1\mathbf{P}_1+k_2\mathbf{P}_2$")
    axes.set_ylabel(r"$E$")
    axes.set_title(rf"{model.name}: $\Phi/\Phi_0={task.p}/{task.q}$")
    axes.margins(x=0.0)
    figure.savefig(figure_path, dpi=_DPI)
    figure.clear()
    return data_path, figure_path


def _number_tag(value: float) -> str:
    return f"{value:.10g}".replace("-", "m").replace(".", "p")


def _packed_spectrum_mask(
    flux: FloatArray,
    energy: FloatArray,
    flux_min: float,
    flux_max: float,
    energy_min: float,
    energy_max: float,
    *,
    width: int = 1360,
    height: int = 900,
) -> tuple[int, int, str]:
    """Rasterize every sampled eigenvalue into a compact one-bit image."""

    x = np.rint(
        (flux - flux_min) / (flux_max - flux_min) * (width - 1)
    ).astype(np.int32)
    y = np.rint(
        (energy_max - energy) / (energy_max - energy_min) * (height - 1)
    ).astype(np.int32)
    np.clip(x, 0, width - 1, out=x)
    np.clip(y, 0, height - 1, out=y)

    # A small circular mask stays smooth when the browser scales the canvas.
    x_left = np.maximum(x - 1, 0)
    x_right = np.minimum(x + 1, width - 1)
    y_up = np.maximum(y - 1, 0)
    y_down = np.minimum(y + 1, height - 1)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y, x] = 1
    mask[y, x_left] = 1
    mask[y, x_right] = 1
    mask[y_up, x] = 1
    mask[y_down, x] = 1
    packed = np.packbits(mask.ravel(), bitorder="big")
    encoded = base64.b64encode(packed.tobytes()).decode("ascii")
    return width, height, encoded


def run_spectrum(
    model: Model, task: SpectrumTask
) -> tuple[Path, Path, Path, Path, Path]:
    """Calculate the spectrum, per-band ranges, and density-flux Wannier diagram."""

    if task.filling_window is not None:
        n_min, n_max = task.filling_window
        if n_min < 0.0 or n_max > model.n_orbitals:
            raise ValueError(
                "Require 0 <= n_min < n_max <= N_orb "
                f"(N_orb = {model.n_orbitals})."
            )

    fractions = rational_fluxes(
        task.flux_min,
        task.flux_max,
        task.q_max,
    )
    if not fractions:
        raise ValueError("The requested range contains no fraction with q <= q_max.")
    hopping_scale = max(abs(hopping.amplitude) for hopping in model.hoppings)
    gap_tolerance = 0.01 * hopping_scale

    stem = (
        f"{model.name}_hofstadter_flux"
        f"{_number_tag(task.flux_min)}_to_{_number_tag(task.flux_max)}"
        f"_qmax{task.q_max}"
    )
    _DATA_DIR.mkdir(exist_ok=True)
    data_path = _DATA_DIR / f"{stem}_spectrum.npz"
    jobs = [
        (p, q, _spectrum_mesh(task, q))
        for p, q in fractions
    ]
    total_spectrum_points = sum(
        mesh[0] * mesh[1] * q * model.n_orbitals
        for _, q, mesh in jobs
    )
    flux_data = np.empty(total_spectrum_points, dtype=np.float64)
    p_data = np.empty(total_spectrum_points, dtype=np.int32)
    q_data = np.empty(total_spectrum_points, dtype=np.int32)
    k1_data = np.empty(total_spectrum_points, dtype=np.float64)
    k2_data = np.empty(total_spectrum_points, dtype=np.float64)
    band_data = np.empty(total_spectrum_points, dtype=np.int32)
    energy_data = np.empty(total_spectrum_points, dtype=np.float64)
    band_ranges = []
    wannier_flux_data = []
    wannier_p_data = []
    wannier_q_data = []
    wannier_band_data = []
    wannier_density_data = []
    wannier_gap_data = []
    interactive_groups = []
    spectrum_offset = 0
    total_jobs = len(jobs)
    print(
        f"Spectrum progress: 0/{total_jobs} (0.0%)",
        end="",
        flush=True,
    )

    for job_index, (p, q, mesh) in enumerate(jobs, start=1):
        flux = p / q
        k_points = _magnetic_mesh(q, mesh)
        # The independent (p, q) = (0, 1) job evaluates H0 on the full-BZ mesh.
        energies = magnetic_energies(model, p, q, k_points)
        n_bands = q * model.n_orbitals
        count = k_points.shape[0] * n_bands
        spectrum_slice = slice(spectrum_offset, spectrum_offset + count)
        flux_data[spectrum_slice] = flux
        p_data[spectrum_slice] = p
        q_data[spectrum_slice] = q
        k1_data[spectrum_slice].reshape(k_points.shape[0], n_bands)[:] = (
            k_points[:, 0, None]
        )
        k2_data[spectrum_slice].reshape(k_points.shape[0], n_bands)[:] = (
            k_points[:, 1, None]
        )
        band_data[spectrum_slice].reshape(k_points.shape[0], n_bands)[:] = (
            np.arange(n_bands, dtype=np.int32)[None, :]
        )
        energy_data[spectrum_slice] = energies.ravel()
        spectrum_offset += count

        band_min = energies.min(axis=0)
        band_max = energies.max(axis=0)
        ranges = np.empty((n_bands, 2, 2), dtype=np.float64)
        ranges[:, :, 0] = flux
        ranges[:, 0, 1] = band_min
        ranges[:, 1, 1] = band_max
        band_ranges.append(ranges)
        gaps = band_min[1:] - band_max[:-1]
        filled_bands = np.flatnonzero(gaps > gap_tolerance) + 1
        gap_intervals = np.column_stack(
            (
                filled_bands,
                band_max[filled_bands - 1],
                band_min[filled_bands],
            )
        ).ravel()
        interactive_groups.append(
            [
                p,
                q,
                np.round(gap_intervals, 8).tolist(),
            ]
        )
        if filled_bands.size:
            wannier_flux_data.append(np.full(filled_bands.size, flux))
            wannier_p_data.append(np.full(filled_bands.size, p, dtype=np.int32))
            wannier_q_data.append(np.full(filled_bands.size, q, dtype=np.int32))
            wannier_band_data.append(filled_bands.astype(np.int32, copy=False))
            wannier_density_data.append(filled_bands / q)
            wannier_gap_data.append(gaps[filled_bands - 1])
        print(
            f"\rSpectrum progress: {job_index}/{total_jobs} "
            f"({100.0 * job_index / total_jobs:.1f}%)",
            end="\n" if job_index == total_jobs else "",
            flush=True,
        )

    if wannier_gap_data:
        wannier_flux = np.concatenate(wannier_flux_data)
        wannier_p = np.concatenate(wannier_p_data)
        wannier_q = np.concatenate(wannier_q_data)
        wannier_band = np.concatenate(wannier_band_data)
        wannier_density = np.concatenate(wannier_density_data)
        wannier_gap = np.concatenate(wannier_gap_data)
    else:
        wannier_flux = np.empty(0, dtype=np.float64)
        wannier_p = np.empty(0, dtype=np.int32)
        wannier_q = np.empty(0, dtype=np.int32)
        wannier_band = np.empty(0, dtype=np.int32)
        wannier_density = np.empty(0, dtype=np.float64)
        wannier_gap = np.empty(0, dtype=np.float64)

    np.savez_compressed(
        data_path,
        name=model.name,
        flux_min=task.flux_min,
        flux_max=task.flux_max,
        q_max=task.q_max,
        k_mesh=np.asarray(task.k_mesh, dtype=np.int32),
        k_mesh_q1=np.asarray(
            () if task.k_mesh_q1 is None else task.k_mesh_q1,
            dtype=np.int32,
        ),
        energy_window=np.asarray(
            () if task.energy_window is None else task.energy_window,
            dtype=np.float64,
        ),
        filling_window=np.asarray(
            () if task.filling_window is None else task.filling_window,
            dtype=np.float64,
        ),
        wannier_gap_threshold=gap_tolerance,
        flux=flux_data,
        p=p_data,
        q=q_data,
        k1=k1_data,
        k2=k2_data,
        band=band_data,
        energy=energy_data,
        wannier_flux=wannier_flux,
        wannier_p=wannier_p,
        wannier_q=wannier_q,
        wannier_band=wannier_band,
        wannier_density=wannier_density,
        wannier_gap=wannier_gap,
    )

    Figure, FigureCanvasAgg = _plot_imports()
    _FIGURE_DIR.mkdir(exist_ok=True)
    spectrum_path = _FIGURE_DIR / f"{stem}_spectrum.png"
    figure = Figure(figsize=(6.0, 5.0), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    spectrum_visible = (
        np.ones(energy_data.size, dtype=bool)
        if task.energy_window is None
        else (
            (energy_data >= task.energy_window[0])
            & (energy_data <= task.energy_window[1])
        )
    )
    spectrum_points = axes.scatter(
        flux_data[spectrum_visible],
        energy_data[spectrum_visible],
        s=0.35,
        marker="o",
        color="0.05",
        linewidths=0.0,
        antialiaseds=True,
    )
    flux_bounds = [task.flux_min, task.flux_max]
    if flux_bounds[0] != flux_bounds[1]:
        padding = 0.01 * (flux_bounds[1] - flux_bounds[0])
        axes.set_xlim(flux_bounds[0] - padding, flux_bounds[1] + padding)
    axes.set_xlabel(r"$\Phi/\Phi_0$")
    axes.set_ylabel(r"$E$")
    axes.set_title(f"{model.name}: Hofstadter spectrum")
    axes.margins(x=0.01, y=0.03)
    if task.energy_window is not None:
        axes.set_ylim(*task.energy_window)
    figure.savefig(spectrum_path, dpi=_DPI)

    from matplotlib.collections import LineCollection

    ranges_path = _FIGURE_DIR / f"{stem}_spectrum_ranges.png"
    ranges = np.concatenate(band_ranges)
    if task.energy_window is not None:
        # Keep bands crossing the window even if neither endpoint lies inside it.
        ranges = ranges[
            (ranges[:, 1, 1] >= task.energy_window[0])
            & (ranges[:, 0, 1] <= task.energy_window[1])
        ]
    point_size = float(spectrum_points.get_sizes()[0])
    point_color = spectrum_points.get_facecolor()
    # Reuse the original axes and freeze their limits before replacing the dots.
    axes.set_xlim(axes.get_xlim())
    axes.set_ylim(axes.get_ylim())
    spectrum_points.remove()
    nonzero = ranges[:, 0, 1] != ranges[:, 1, 1]
    axes.add_collection(
        LineCollection(
            ranges[nonzero],
            colors=point_color,
            linewidths=sqrt(point_size),
            capstyle="round",
            antialiaseds=True,
            # Pixel snapping can collapse subpixel-width bands to empty paths.
            snap=False,
        ),
        autolim=False,
    )
    if np.any(~nonzero):
        # Backends may drop zero-length paths; draw the exact limiting circle.
        axes.scatter(
            ranges[~nonzero, 0, 0],
            ranges[~nonzero, 0, 1],
            s=point_size,
            marker="o",
            color=point_color,
            linewidths=0.0,
            antialiaseds=True,
        )
    axes.set_title(f"{model.name}: Hofstadter band ranges")
    figure.savefig(ranges_path, dpi=_DPI)
    figure.clear()

    wannier_path = _FIGURE_DIR / f"{stem}_wannier.png"
    figure = Figure(figsize=(6.0, 5.0), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    n_bounds = (
        (0.0, float(model.n_orbitals))
        if task.filling_window is None
        else task.filling_window
    )
    wannier_visible = (
        (wannier_density >= n_bounds[0])
        & (wannier_density <= n_bounds[1])
    )
    if np.any(wannier_visible):
        from matplotlib.colors import LinearSegmentedColormap, PowerNorm

        visible_gap = wannier_gap[wannier_visible]
        gap_colormap = LinearSegmentedColormap.from_list(
            "gap_size",
            ("#ffffff", "#1769aa"),
        )
        gap_norm = PowerNorm(
            gamma=0.5,
            vmin=0.0,
            vmax=float(visible_gap.max()),
        )
        points = axes.scatter(
            wannier_flux[wannier_visible],
            wannier_density[wannier_visible],
            c=visible_gap,
            s=1.0,
            marker="o",
            cmap=gap_colormap,
            norm=gap_norm,
            linewidths=0.0,
            antialiaseds=True,
        )
        figure.colorbar(points, ax=axes, label=r"Gap size $\Delta E$")
    if flux_bounds[0] != flux_bounds[1]:
        padding = 0.01 * (flux_bounds[1] - flux_bounds[0])
        axes.set_xlim(flux_bounds[0] - padding, flux_bounds[1] + padding)
    axes.set_ylim(*n_bounds)
    axes.set_yticks(_integer_ticks(*n_bounds))
    axes.set_xlabel(r"$\Phi/\Phi_0$")
    axes.set_ylabel(r"$n$ per primitive cell")
    axes.set_title(f"{model.name}: Wannier diagram")
    axes.margins(x=0.01)
    figure.savefig(wannier_path, dpi=_DPI)
    figure.clear()

    interactive_path = _FIGURE_DIR / f"{stem}_interactive.html"
    if flux_bounds[0] == flux_bounds[1]:
        flux_padding = 0.01 * max(1.0, abs(flux_bounds[0]))
        flux_bounds = [
            flux_bounds[0] - flux_padding,
            flux_bounds[1] + flux_padding,
        ]
    if task.energy_window is None:
        interactive_energy_min = float(energy_data.min())
        interactive_energy_max = float(energy_data.max())
        energy_padding = 0.025 * max(
            1e-12,
            interactive_energy_max - interactive_energy_min,
        )
        interactive_energy_min -= energy_padding
        interactive_energy_max += energy_padding
    else:
        interactive_energy_min, interactive_energy_max = task.energy_window

    visible_energy = (
        (energy_data >= interactive_energy_min)
        & (energy_data <= interactive_energy_max)
    )
    spectrum_width, spectrum_height, spectrum_mask = _packed_spectrum_mask(
        flux_data[visible_energy],
        energy_data[visible_energy],
        flux_bounds[0],
        flux_bounds[1],
        interactive_energy_min,
        interactive_energy_max,
    )

    interactive_groups_visible = []
    for p, q, flat_gaps in interactive_groups:
        visible_gaps = []
        for index in range(0, len(flat_gaps), 3):
            filled, lower, upper = flat_gaps[index : index + 3]
            density = filled / q
            if not n_bounds[0] <= density <= n_bounds[1]:
                continue
            if upper < interactive_energy_min or lower > interactive_energy_max:
                continue
            visible_gaps.extend((filled, lower, upper))
        interactive_groups_visible.append([p, q, visible_gaps])

    save_linked_figure(
        interactive_path,
        name=model.name,
        n_orbitals=model.n_orbitals,
        n_min=n_bounds[0],
        n_max=n_bounds[1],
        flux_min=flux_bounds[0],
        flux_max=flux_bounds[1],
        energy_min=interactive_energy_min,
        energy_max=interactive_energy_max,
        spectrum_width=spectrum_width,
        spectrum_height=spectrum_height,
        spectrum_mask=spectrum_mask,
        gap_threshold=gap_tolerance,
        groups=interactive_groups_visible,
    )
    return data_path, spectrum_path, wannier_path, interactive_path, ranges_path


def main() -> None:
    input_path = model_path_from_command_line()
    task = load_hofstadter_task(input_path)
    thread_limit = nullcontext()
    if task.omp_num_threads is not None:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as error:
            raise ImportError(
                "Setting 'omp_num_threads' requires threadpoolctl; install it "
                "with 'python3 -m pip install threadpoolctl'."
            ) from error
        # Limit BLAS as well as OpenMP: OpenBLAS may use pthreads instead.
        thread_limit = threadpool_limits(limits=task.omp_num_threads)

    with thread_limit:
        model = load_model(input_path)
        if isinstance(task, BandTask):
            output_paths = run_band(model, task)
        else:
            output_paths = run_spectrum(model, task)

    print("Real-space Hermiticity check passed.")
    for path in output_paths:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
