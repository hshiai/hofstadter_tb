# Hofstadter TB

A compact Python implementation for zero-field and Hofstadter-band calculations
of general two-dimensional multi-orbital tight-binding models. The magnetic
Hamiltonian uses the analytic oblique-gauge expression developed in the
companion work.

## Features

- General two-dimensional Bravais lattices and orbital embeddings
- Real or complex hopping amplitudes
- Automatic real-space Hermiticity validation
- Real-space lattice and Brillouin-zone plots
- Zero-field band structures
- Magnetic bands at a selected rational flux
- Hofstadter spectra over a flux interval
- Wannier diagrams with gap-size coloring
- A linked interactive spectrum/Wannier figure

## Requirements

- Python 3.11 or newer
- NumPy
- Matplotlib

Install the dependencies with

```bash
python3 -m pip install numpy matplotlib
```

## Quick start

Each model is stored in a TOML file. Run the commands from this directory:

```bash
cd hofstadter_tb
```

The repository includes square, triangular, honeycomb, Kagome, QWZ,
topological flat-band, and imbalanced-bipartite flat-band (IBF) example inputs. Pass the desired TOML file to
any command:

```bash
python3 model.py example_triangular.toml
python3 band.py example_triangular.toml
python3 hofstadter.py example_triangular.toml
```

If the file name is omitted, the programs read a local `model.toml` if one is
provided by the user.

### 1. Define the lattice and hopping model

The primitive lattice vectors are Cartesian coordinates:

```toml
name = "graphene"

a1 = [+0.5, 0.8660254037844386]
a2 = [-0.5, 0.8660254037844386]
```

Orbital positions are fractional coordinates in the primitive-vector basis,

$$
\boldsymbol{\tau}_\alpha
= \xi_1^\alpha \mathbf a_1 + \xi_2^\alpha \mathbf a_2,
$$

and orbital indices start from zero:

```toml
tau = [
  [0.3333333333333333, 0.3333333333333333],
  [0.6666666666666666, 0.6666666666666666],
]
```

Each hopping line contains

```text
j  alpha1  alpha2  ell1  ell2  Re(t)  Im(t)
```

and represents a hopping from orbital `alpha2` in the reference cell to
orbital `alpha1` in cell `ell1*a1 + ell2*a2`:

```toml
hopping = '''
0  0  1   0  0  -1.0  0.0
1  1  0   0  0  -1.0  0.0
'''
```

Both `j` and the orbital indices start from zero. Every hopping must have its
Hermitian partner,

```text
(alpha1, alpha2, ell1, ell2, t)
    <->
(alpha2, alpha1, -ell1, -ell2, conjugate(t)).
```

Invalid or duplicate hopping channels are rejected before any calculation.

### 2. Plot the lattice and Brillouin zone

```bash
python3 model.py model.toml
```

This saves two figures in `figure/`:

- `<name>_lattice.png`
- `<name>_brillouin_zone.png`

The Brillouin-zone figure contains both the Wigner-Seitz Brillouin zone and
the reciprocal-cell parallelogram spanned by `b1` and `b2`.

### 3. Calculate zero-field bands

Set a path in reciprocal fractional coordinates,
$\mathbf k=k_1\mathbf b_1+k_2\mathbf b_2$:

```toml
band_path = '''
G   0.0                 0.0
K   0.3333333333333333 -0.3333333333333333
M   0.5                 0.0
G   0.0                 0.0
'''

n_k = 100
```

Then run

```bash
python3 band.py model.toml
```

The numerical bands are saved in `data/<name>_band.dat`, and the plot is saved
in `figure/<name>_band.png`.

## Hofstadter calculations

The active calculation is selected by

```toml
[hofstadter]
mode = "band"       # or "spectrum"
```

For a single magnetic-band calculation, use a signed integer `p` and a positive
integer `q`. A spectrum calculation uses the signed interval from `flux_min`
to `flux_max`.

### Magnetic bands at one rational flux

For $\Phi/\Phi_0=p/q$, use

```toml
[hofstadter.band]
p = 1
q = 7

# k = k1*P1 + k2*P2, with P1 = b1/q and P2 = b2
k_path = '''
0.2  0.0
0.2  1.0
1.0  1.0
'''

n_k = 80
```

The integer `p` may be positive, zero, or negative, while `q` must be positive;
`abs(p)` and `q` must be coprime. Use `p = -1` for flux $-1/q$. Set
`mode = "band"`, then run

```bash
python3 hofstadter.py model.toml
```

The band data are written to `data/`, and the magnetic-band plot is written to
`figure/`.

### Hofstadter spectrum and Wannier diagram

Set `mode = "spectrum"` and specify

```toml
[hofstadter.spectrum]
flux_min = -0.3
flux_max = 0.3
q_max = 67
k_mesh = [6, 6]

# Optional: denser sampling for the broader small-q bands.  The mesh decreases
# as 1/sqrt(q) and never goes below k_mesh.
k_mesh_q1 = [20, 20]

# Optional display windows. Omit either pair to use the full energy or
# filling range. Both bounds in a pair must be provided together.
energy_min = -1.0
energy_max = 1.0
n_min = 0.5
n_max = 1.5
```

The program evaluates every signed reduced fraction `p/q` in the requested
interval with `q <= q_max`, keeping `p` signed and `q` positive. Thus a single
run with `flux_min = -1.0` and `flux_max = 1.0` covers both field directions.
The optional energy and filling windows control the displayed ranges of the
static and interactive figures. The compressed NPZ file continues to store the
complete calculated spectrum and gap data. The Wannier-diagram vertical axis
shows only integer filling ticks within the selected range.
The momentum mesh covers the spectrally distinct magnetic Brillouin zone,

```text
k1 in [0, 1),    k2 in [0, 1/q).
```

Run

```bash
python3 hofstadter.py model.toml
```

This produces

- `data/<name>_hofstadter_flux..._spectrum.npz`: numerical spectrum and gap data
- `figure/<name>_hofstadter_flux..._spectrum.png`: Hofstadter spectrum
- `figure/<name>_hofstadter_flux..._wannier.png`: Wannier diagram
- `figure/<name>_hofstadter_flux..._interactive.html`: linked interactive figure

Open the HTML file in a web browser. Clicking a spectral gap highlights the
corresponding point $(\Phi/\Phi_0,n=r/q)$ in the Wannier diagram; clicking a
Wannier point highlights the corresponding energy gap.

Only gaps larger than

```text
0.01 * max(|hopping amplitude|)
```

are included in the Wannier diagram. The color represents the gap size, with
white corresponding to zero and darker blue corresponding to a larger gap.

## Accuracy and performance

- Increase `k_mesh` to obtain more reliable band edges and gap sizes.
- If `k_mesh_q1` is given, the spectrum uses
  `max(k_mesh, ceil(k_mesh_q1/sqrt(q)))` in each momentum direction.  Omitting
  it keeps the fixed `k_mesh` behavior used by older input files.
- Increase `q_max` for a denser Hofstadter spectrum.
- The magnetic Hamiltonian dimension is `q * N_orb`, so large denominators are
  more expensive. Dense batches are automatically reduced at large `q` so that
  the temporary Hamiltonian stack remains below about 256 MiB.
- Output directories are created automatically.

All energies are expressed in the same units as the hopping amplitudes.
