"""Import a two-dimensional model from Wannier90 formatted output."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
_DEFAULT_TOLERANCE: Final = 1.0e-12


@dataclass(frozen=True, slots=True)
class ImportedHopping:
    alpha1: int
    alpha2: int
    ell1: int
    ell2: int
    amplitude: complex


@dataclass(frozen=True, slots=True)
class ImportedModel:
    a1: FloatArray
    a2: FloatArray
    tau: FloatArray
    hoppings: tuple[ImportedHopping, ...]


def _fortran_float(value: str, context: str) -> float:
    try:
        number = float(value.replace("D", "E").replace("d", "e"))
    except ValueError as error:
        raise ValueError(f"Invalid real number in {context}: {value!r}.") from error
    if not np.isfinite(number):
        raise ValueError(f"Non-finite real number in {context}: {value!r}.")
    return number


class _TextLines:
    """Read nonempty whitespace-separated records from a formatted file."""

    __slots__ = ("index", "lines", "path")

    def __init__(self, path: Path) -> None:
        try:
            self.lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as error:
            raise ValueError(f"Cannot read Wannier90 file '{path}'.") from error
        if not self.lines:
            raise ValueError(f"Wannier90 file '{path}' is empty.")
        self.index = 1  # The first line is the Wannier90 header.
        self.path = path

    def next(self, context: str) -> list[str]:
        while self.index < len(self.lines):
            fields = self.lines[self.index].split()
            self.index += 1
            if fields:
                return fields
        raise ValueError(
            f"Unexpected end of '{self.path}' while reading {context}."
        )

    def remaining(self) -> list[str] | None:
        while self.index < len(self.lines):
            fields = self.lines[self.index].split()
            self.index += 1
            if fields:
                return fields
        return None


def _integers(fields: list[str], count: int, context: str) -> tuple[int, ...]:
    if len(fields) != count:
        raise ValueError(f"{context} must contain {count} integers.")
    try:
        return tuple(map(int, fields))
    except ValueError as error:
        raise ValueError(f"Invalid integer in {context}.") from error


def _seed_path(input_path: Path, tag: str, suffix: str) -> Path:
    seed = Path(tag).expanduser()
    if seed.name.endswith("_tb.dat") or seed.name.endswith("_wsvec.dat"):
        raise ValueError(
            "'[wannier90].tag' is the seedname, without '_tb.dat' or "
            "'_wsvec.dat'."
        )
    if not seed.is_absolute():
        seed = input_path.parent / seed
    return seed.parent / f"{seed.name}{suffix}"


def _read_wsvec(
    path: Path,
) -> dict[tuple[int, int, int, int, int], tuple[tuple[int, int, int], ...]]:
    if not path.is_file():
        return {}

    lines = _TextLines(path)
    translations = {}
    while (fields := lines.remaining()) is not None:
        r1, r2, r3, m, n = _integers(
            fields, 5, f"a channel header in '{path}'"
        )
        count, = _integers(
            lines.next(f"the image count for ({r1}, {r2}, {r3}, {m}, {n})"),
            1,
            f"the image count in '{path}'",
        )
        if count < 1:
            raise ValueError(f"Image count in '{path}' must be positive.")
        images = tuple(
            _integers(
                lines.next("a shortest-image translation"),
                3,
                f"a shortest-image translation in '{path}'",
            )
            for _ in range(count)
        )
        key = (r1, r2, r3, m - 1, n - 1)
        if m < 1 or n < 1 or key in translations:
            raise ValueError(f"Invalid or duplicate channel {key} in '{path}'.")
        translations[key] = images
    return translations


def _read_tb(
    path: Path,
) -> tuple[
    FloatArray,
    list[tuple[tuple[int, int, int], int, int, complex, int]],
    FloatArray,
]:
    lines = _TextLines(path)
    lattice_rows = []
    for index in range(3):
        fields = lines.next(f"lattice vector {index + 1}")
        if len(fields) != 3:
            raise ValueError(f"Lattice vector {index + 1} in '{path}' needs 3 values.")
        lattice_rows.append(
            [_fortran_float(value, f"lattice vector {index + 1}") for value in fields]
        )
    lattice = np.asarray(lattice_rows, dtype=np.float64)

    n_wann, = _integers(
        lines.next("the number of Wannier functions"),
        1,
        f"the number of Wannier functions in '{path}'",
    )
    n_r, = _integers(
        lines.next("the number of Wigner-Seitz vectors"),
        1,
        f"the number of Wigner-Seitz vectors in '{path}'",
    )
    if n_wann < 1 or n_r < 1:
        raise ValueError(f"Wannier and R-vector counts in '{path}' must be positive.")

    degeneracies = []
    while len(degeneracies) < n_r:
        fields = lines.next("the Wigner-Seitz degeneracies")
        try:
            degeneracies.extend(map(int, fields))
        except ValueError as error:
            raise ValueError(f"Invalid Wigner-Seitz degeneracy in '{path}'.") from error
    if len(degeneracies) != n_r or any(value < 1 for value in degeneracies):
        raise ValueError(
            f"'{path}' must contain exactly {n_r} positive degeneracies."
        )

    records = []
    r_vectors = []
    for r_index, degeneracy in enumerate(degeneracies):
        r_vector = _integers(
            lines.next(f"Hamiltonian R vector {r_index + 1}"),
            3,
            f"Hamiltonian R vector {r_index + 1} in '{path}'",
        )
        r_vectors.append(r_vector)
        seen = set()
        for _ in range(n_wann * n_wann):
            fields = lines.next(f"Hamiltonian block at R={r_vector}")
            if len(fields) != 4:
                raise ValueError(
                    f"A Hamiltonian row at R={r_vector} in '{path}' needs "
                    "m, n, Re(H), and Im(H)."
                )
            m, n = _integers(
                fields[:2], 2, f"Hamiltonian orbital indices in '{path}'"
            )
            if not (1 <= m <= n_wann and 1 <= n <= n_wann) or (m, n) in seen:
                raise ValueError(
                    f"Invalid or duplicate Hamiltonian indices ({m}, {n}) "
                    f"at R={r_vector} in '{path}'."
                )
            seen.add((m, n))
            amplitude = complex(
                _fortran_float(fields[2], "Re(H)"),
                _fortran_float(fields[3], "Im(H)"),
            )
            records.append((r_vector, m - 1, n - 1, amplitude, degeneracy))

    centres = np.full((n_wann, 3), np.nan, dtype=np.float64)
    for r_index in range(n_r):
        r_vector = _integers(
            lines.next(f"position-operator R vector {r_index + 1}"),
            3,
            f"position-operator R vector {r_index + 1} in '{path}'",
        )
        if r_vector != r_vectors[r_index]:
            raise ValueError(
                f"Hamiltonian and position blocks in '{path}' use different "
                f"R-vector order at block {r_index + 1}."
            )
        seen = set()
        for _ in range(n_wann * n_wann):
            fields = lines.next(f"position block at R={r_vector}")
            if len(fields) != 8:
                raise ValueError(
                    f"A position row at R={r_vector} in '{path}' needs m, n "
                    "and three complex Cartesian components."
                )
            m, n = _integers(
                fields[:2], 2, f"position-operator indices in '{path}'"
            )
            if not (1 <= m <= n_wann and 1 <= n <= n_wann) or (m, n) in seen:
                raise ValueError(
                    f"Invalid or duplicate position indices ({m}, {n}) "
                    f"at R={r_vector} in '{path}'."
                )
            seen.add((m, n))
            components = [
                _fortran_float(value, "a position-matrix component")
                for value in fields[2:]
            ]
            if r_vector == (0, 0, 0) and m == n:
                centres[m - 1] = components[0], components[2], components[4]

    if lines.remaining() is not None:
        raise ValueError(f"Unexpected data after the position blocks in '{path}'.")
    if np.isnan(centres).any():
        raise ValueError(f"'{path}' does not contain every diagonal r_mm(R=0).")
    return lattice, records, centres


def _project_to_2d(
    lattice: FloatArray,
    centres: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    a1_3d = lattice[0]
    a2_3d = lattice[1]
    length1 = float(np.linalg.norm(a1_3d))
    if length1 == 0.0:
        raise ValueError("The first Wannier90 lattice vector has zero length.")
    projection = float(np.dot(a1_3d, a2_3d) / length1)
    perpendicular_squared = float(np.dot(a2_3d, a2_3d) - projection**2)
    tolerance = 128.0 * np.finfo(np.float64).eps * float(np.dot(a2_3d, a2_3d))
    if perpendicular_squared <= tolerance:
        raise ValueError("The first two Wannier90 lattice vectors are collinear.")

    a1 = np.array([length1, 0.0], dtype=np.float64)
    a2 = np.array([projection, np.sqrt(perpendicular_squared)], dtype=np.float64)
    direct_3d = np.column_stack((a1_3d, a2_3d))
    tau = np.linalg.lstsq(direct_3d, centres.T, rcond=None)[0].T
    tau[np.abs(tau) < 1.0e-14] = 0.0
    return a1, a2, tau


def _convert_hoppings(
    records: list[tuple[tuple[int, int, int], int, int, complex, int]],
    wsvec: dict[tuple[int, int, int, int, int], tuple[tuple[int, int, int], ...]],
    tolerance: float,
) -> tuple[ImportedHopping, ...]:
    combined: dict[tuple[int, int, int, int], complex] = {}
    record_keys = set()
    for r_vector, alpha1, alpha2, amplitude, degeneracy in records:
        key_3d = (*r_vector, alpha1, alpha2)
        record_keys.add(key_3d)
        images = wsvec.get(key_3d, ((0, 0, 0),))
        value = amplitude / (degeneracy * len(images))
        for translation in images:
            effective = tuple(r_vector[index] + translation[index] for index in range(3))
            if effective[2] != 0:
                if abs(value) > tolerance:
                    raise ValueError(
                        "The Wannier90 model contains a significant hopping along "
                        f"the third lattice direction: R={effective}, "
                        f"orbitals=({alpha1 + 1}, {alpha2 + 1}), |H|={abs(value):.3e}. "
                        "This code accepts two-dimensional models only; increase "
                        "'[wannier90].hopping_tolerance' only if this is residual "
                        "vacuum coupling."
                    )
                continue
            key_2d = (alpha1, alpha2, effective[0], effective[1])
            combined[key_2d] = combined.get(key_2d, 0.0j) + value

    unused_wsvec = wsvec.keys() - record_keys
    if unused_wsvec:
        raise ValueError(
            "The Wannier90 wsvec file contains a channel absent from tb.dat: "
            f"{next(iter(unused_wsvec))}."
        )

    selected = {}
    visited = set()
    for key, value in combined.items():
        if key in visited:
            continue
        alpha1, alpha2, ell1, ell2 = key
        reverse_key = (alpha2, alpha1, -ell1, -ell2)
        reverse = combined.get(reverse_key)
        if reverse is None:
            if abs(value) > tolerance:
                raise ValueError(
                    "Wannier90 hopping has no Hermitian partner after the 2D "
                    f"conversion: {key}."
                )
            visited.add(key)
            continue
        visited.update((key, reverse_key))
        if max(abs(value), abs(reverse)) > tolerance:
            selected[key] = value
            selected[reverse_key] = reverse

    hoppings = tuple(
        ImportedHopping(*key, amplitude)
        for key, amplitude in sorted(selected.items())
    )
    if not hoppings:
        raise ValueError(
            "No Wannier90 hopping survives '[wannier90].hopping_tolerance'."
        )
    return hoppings


def load_wannier90(
    input_path: Path,
    tag: str,
    hopping_tolerance: float = _DEFAULT_TOLERANCE,
) -> ImportedModel:
    """Load a 2D Wannier90 model associated with one TOML input file."""

    tb_path = _seed_path(input_path, tag, "_tb.dat")
    if not tb_path.is_file():
        raise ValueError(f"Wannier90 file '{tb_path}' was not found for tag {tag!r}.")
    lattice, records, centres = _read_tb(tb_path)
    wsvec = _read_wsvec(_seed_path(input_path, tag, "_wsvec.dat"))
    a1, a2, tau = _project_to_2d(lattice, centres)
    return ImportedModel(
        a1,
        a2,
        tau,
        _convert_hoppings(records, wsvec, hopping_tolerance),
    )
