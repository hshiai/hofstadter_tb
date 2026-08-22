"""Read the local zero-field tight-binding model."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib
from typing import Final

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
_TWO_PI: Final = 2.0 * np.pi
_LOCAL_INPUT: Final = Path(__file__).with_name("model.toml")


def _as_float_array(value: object, name: str) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f"'{name}' must contain real numbers.") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"'{name}' must contain only finite values.")
    return array


def _read_only(array: FloatArray) -> FloatArray:
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class Hopping:
    j: int
    alpha1: int
    alpha2: int
    ell1: int
    ell2: int
    amplitude: complex


@dataclass(frozen=True, slots=True)
class Model:
    """Zero-field tight-binding model in the article's convention.

    ``tau[alpha]`` stores the fractional coordinates
    ``(xi_1^alpha, xi_2^alpha)`` without wrapping them into ``[0, 1)``.
    """

    name: str
    a1: FloatArray
    a2: FloatArray
    tau: FloatArray
    hoppings: tuple[Hopping, ...]
    area: float = field(init=False)
    b1: FloatArray = field(init=False, repr=False)
    b2: FloatArray = field(init=False, repr=False)
    tau_cart: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or not self.name.isascii()
            or not self.name.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("'name' must contain only letters, numbers, '-' or '_'.")

        a1 = _as_float_array(self.a1, "a1")
        a2 = _as_float_array(self.a2, "a2")
        tau = _as_float_array(self.tau, "tau")

        if a1.shape != (2,) or a2.shape != (2,):
            raise ValueError("'a1' and 'a2' must each contain two numbers.")
        if tau.ndim != 2 or tau.shape[0] == 0 or tau.shape[1] != 2:
            raise ValueError("'tau' must be a nonempty N_orb x 2 array.")

        direct = np.column_stack((a1, a2))
        area = float(np.linalg.det(direct))
        scale = float(np.linalg.norm(a1) * np.linalg.norm(a2))
        tolerance = 64.0 * np.finfo(np.float64).eps * scale
        if abs(area) <= tolerance:
            raise ValueError("'a1' and 'a2' must be linearly independent.")
        if area < 0.0:
            raise ValueError("The lattice orientation must satisfy (a1 x a2)_z > 0.")

        reciprocal = _TWO_PI * np.linalg.inv(direct).T
        tau_cart = tau @ direct.T

        object.__setattr__(self, "a1", _read_only(a1))
        object.__setattr__(self, "a2", _read_only(a2))
        object.__setattr__(self, "tau", _read_only(tau))
        object.__setattr__(self, "area", area)
        object.__setattr__(self, "b1", _read_only(np.ascontiguousarray(reciprocal[:, 0])))
        object.__setattr__(self, "b2", _read_only(np.ascontiguousarray(reciprocal[:, 1])))
        object.__setattr__(self, "tau_cart", _read_only(np.ascontiguousarray(tau_cart)))

    @property
    def n_orbitals(self) -> int:
        return self.tau.shape[0]

    @property
    def n_hoppings(self) -> int:
        return len(self.hoppings)

    def first_brillouin_zone(self) -> FloatArray:
        """Return the counterclockwise vertices of the first Brillouin zone."""

        from lattice_plot import first_brillouin_zone

        return first_brillouin_zone(self.b1, self.b2)

    def plot(
        self,
        *,
        cells: int = 2,
    ) -> tuple[Path, Path]:
        """Save real-space lattice and first-BZ figures.

        ``cells=2`` displays a 5-by-5 array of primitive cells. Matplotlib is
        imported only when this method is called, so plotting has no cost for
        band calculations.
        """

        from lattice_plot import save_lattice_figures

        return save_lattice_figures(
            self,
            cells=cells,
        )


def _parse_hoppings(text: object, n_orbitals: int) -> tuple[Hopping, ...]:
    if not isinstance(text, str):
        raise ValueError("'hopping' must be a multiline string.")

    hoppings = []
    used_j = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 7:
            raise ValueError(f"Hopping line {line_number} must contain seven columns.")
        try:
            j, alpha1, alpha2, ell1, ell2 = map(int, fields[:5])
            real, imag = map(float, fields[5:])
        except ValueError as error:
            raise ValueError(f"Invalid value on hopping line {line_number}.") from error

        if j < 0 or j in used_j:
            raise ValueError(f"Hopping index j={j} must be nonnegative and unique.")
        if not 0 <= alpha1 < n_orbitals or not 0 <= alpha2 < n_orbitals:
            raise ValueError(
                f"Orbital indices on hopping line {line_number} must be in "
                f"0, ..., {n_orbitals - 1}."
            )
        if not np.isfinite(real) or not np.isfinite(imag):
            raise ValueError(f"Non-finite amplitude on hopping line {line_number}.")

        used_j.add(j)
        hoppings.append(Hopping(j, alpha1, alpha2, ell1, ell2, complex(real, imag)))

    if not hoppings:
        raise ValueError("At least one hopping line is required.")
    hoppings = tuple(hoppings)
    _check_hermiticity(hoppings)
    return hoppings


def _check_hermiticity(hoppings: tuple[Hopping, ...]) -> None:
    """Check h_ab(R) = h_ba(-R)* for every hopping channel."""

    by_channel = {}
    for hopping in hoppings:
        key = (hopping.alpha1, hopping.alpha2, hopping.ell1, hopping.ell2)
        if key in by_channel:
            other = by_channel[key]
            raise ValueError(
                f"Hoppings j={other.j} and j={hopping.j} describe the same channel."
            )
        by_channel[key] = hopping

    for hopping in hoppings:
        reverse_key = (
            hopping.alpha2,
            hopping.alpha1,
            -hopping.ell1,
            -hopping.ell2,
        )
        reverse = by_channel.get(reverse_key)
        if reverse is None:
            raise ValueError(
                f"Hopping j={hopping.j} has no Hermitian partner "
                f"({reverse_key[0]} {reverse_key[1]} "
                f"{reverse_key[2]} {reverse_key[3]})."
            )
        if not np.isclose(
            reverse.amplitude,
            hopping.amplitude.conjugate(),
            rtol=1.0e-12,
            atol=1.0e-12,
        ):
            raise ValueError(
                f"Hoppings j={hopping.j} and j={reverse.j} violate Hermiticity: "
                "their amplitudes must be complex conjugates."
            )


def _load_model(path: Path) -> Model:
    """Load one model file."""

    with path.open("rb") as stream:
        data = tomllib.load(stream)

    required = {"name", "a1", "a2", "tau", "hopping"}
    optional = {"band_path", "n_k", "hofstadter"}
    missing = required - data.keys()
    extra = data.keys() - required - optional
    if missing:
        raise ValueError(f"Missing model field(s): {', '.join(sorted(missing))}.")
    if extra:
        raise ValueError(f"Unknown model field(s): {', '.join(sorted(extra))}.")

    tau = _as_float_array(data["tau"], "tau")
    if tau.ndim != 2 or tau.shape[0] == 0 or tau.shape[1] != 2:
        raise ValueError("'tau' must be a nonempty N_orb x 2 array.")

    return Model(
        name=data["name"],
        a1=data["a1"],
        a2=data["a2"],
        tau=tau,
        hoppings=_parse_hoppings(data["hopping"], tau.shape[0]),
    )


def load_model() -> Model:
    """Load the local ``model.toml`` file."""

    return _load_model(_LOCAL_INPUT)


def main() -> None:
    model = load_model()
    print(f"name  = {model.name}")
    print(f"N_orb = {model.n_orbitals}")
    print(f"N_hop = {model.n_hoppings}")
    print(f"S_0   = {model.area:.16g}")
    print(f"b_1   = {model.b1}")
    print(f"b_2   = {model.b2}")
    lattice_path, bz_path = model.plot()
    print(f"Saved {lattice_path}")
    print(f"Saved {bz_path}")


if __name__ == "__main__":
    main()
