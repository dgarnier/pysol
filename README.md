# Inductance

[![PyPI](https://img.shields.io/pypi/v/inductance.svg)][pypi status]
[![Status](https://img.shields.io/pypi/status/inductance.svg)][pypi status]
[![Python Version](https://img.shields.io/pypi/pyversions/inductance)][pypi status]
[![License](https://img.shields.io/pypi/l/inductance)][license]
[![Read the documentation at https://inductance.readthedocs.io/](https://img.shields.io/readthedocs/inductance/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/dgarnier/inductance/workflows/Tests/badge.svg)][tests]
[![Codecov](https://codecov.io/gh/dgarnier/inductance/branch/main/graph/badge.svg)][codecov]
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)][pre-commit]
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)][black]

[pypi status]: https://pypi.org/project/inductance/
[read the docs]: https://inductance.readthedocs.io/
[tests]: https://github.com/dgarnier/inductance/actions?workflow=Tests
[codecov]: https://app.codecov.io/gh/dgarnier/inductance
[pre-commit]: https://github.com/pre-commit/pre-commit
[black]: https://github.com/psf/black

This is a python library to calculate inductance. Mostly for the purposes of calcualting magnetically confined plasmas. It might someday actually contain some plasma physics, but lets not get too carried away.

## Features

- Self-inductance formulas

  - self inductance of circular, circular hollow, and rectangular section by Maxwell's approximation
  - Lyle's approximation for thick coil solenoid self inductances to 4th and 6th order.
  - Butterworth's approximation for long solenoids
  - Lorentz's perfect analytic solution for current sheet solenoids
  - Babic and Akyel's approximation for thin solenoids

- Mutual-inductance formulas

  - mutual inductance of filaments (Maxwell)

- Filamentary models

  - utility functions to create filament arrays from rectangular definitions of coils and subcoils
  - calculation for filament array mutual inductance
  - calculation of filament array self inductance

- Green's functions

  - calculation of Green's functions for Psi, Br, and Bz from filamented coils to points
    - with Numba, calculation of green's functions for arbitrary grids of points
  - calculation fo Green's functions for coil forces

- Arbitrary coil shapes
  - rudimentary support for arbitrary wire filament coil shapes

## Requirements

_Inductance_ requires [_NumPy_][numpy] and uses [_Numba_][numba] for acceleration. It is written in mostly pure python referencing academic articles for calculating inductances by various methods, most of which rely on elliptic functions. _Inductance_ provides _Numba_ accelerated pure python elliptic functions.

It is possible to remove the dependence on Numba and get most of the functionality of _Inductance_. The plan is to provide different options, including with alternative accelerators, such as [_JAX_][jax]. For now, the requirements are:

- python >= 3.8
- numpy >= 1.24
- numba >= 0.57

[numba]: https://numba.readthedocs.io/
[numpy]: https://numpy.org
[jax]: https://jax.readthedocs.io/

## Installation

You can install _inductance_ via [pip] from [PyPI]:

```console
$ pip install inductance
```

## Reference

Please see the [reference] for details.

## Contributing

Contributions are very welcome.
To learn more, see the [Contributor Guide].

## License

Distributed under the terms of the [MIT license][license],
_inductance_ is free and open source software.

## Issues

For now, this is a very early release. It is likely a new top level API will be
developed as the library matures.

If you encounter any problems,
please [file an issue] along with a detailed description.

## Credits

This project was generated from [@cjolowicz]'s [Hypermodern Python Cookiecutter] template.

[@cjolowicz]: https://github.com/cjolowicz
[pypi]: https://pypi.org/
[hypermodern python cookiecutter]: https://github.com/cjolowicz/cookiecutter-hypermodern-python
[file an issue]: https://github.com/dgarnier/inductance/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/dgarnier/inductance/blob/main/LICENSE
[contributor guide]: https://github.com/dgarnier/inductance/blob/main/CONTRIBUTING.md
[reference]: https://inductance.readthedocs.io/en/latest/reference.html
