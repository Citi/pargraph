We welcome contributions to the Pargraph library.


## Helpful Resources

* [README.md](./README.md)

* Documentation: [./docs](./docs) (see [README.md](./README.md) for building instructions)

* [Repository](https://github.com/citi/pargraph)

* [Issue tracking](https://github.com/citi/pargraph/issues)


## Contributing Guide

When contributing to the project, please take care of following these requirements.


### Style guide

**We enforce the [PEP 8](https://peps.python.org/pep-0008/) coding style, with a relaxed constraint on the maximum line
length (120 columns)**.

Before merging your changes into your `master` branch, our CI system will run the following checks:

```bash
isort --profile black --line-length 120
black -l 120 -C
flake8 --max-line-length 120 --extend-ignore=E203
```

The `isort`, `black` and `flake8` packages can be installed through Python's PIP.


### Bump version number

You must update the version defined in [about.py](pargraph/about.py) for every contribution. Please follow
[semantic versioning](https://semver.org) in the format `MAJOR.MINOR.PATCH`.


## Code of Conduct

We are committed to making open source an enjoyable and respectful experience for our community. See
[`CODE_OF_CONDUCT`](https://github.com/citi/.github/blob/main/CODE_OF_CONDUCT.md) for more information.