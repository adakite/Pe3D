# Pe3d Release Notes

These notes assume `pe3d` is being released as a standalone Python fork repository without bundling the upstream MATLAB tree or older Python port.

## Local Build

Use the project virtualenv:

```bash
.venv/bin/pip install build twine setuptools wheel
.venv/bin/python -m build --sdist --wheel --no-isolation
.venv/bin/python -m twine check dist/*
```

## Inspect Artifacts

Expected release artifacts:

- `dist/pe3d-<version>.tar.gz`
- `dist/pe3d-<version>-py3-none-any.whl`

The source distribution should only contain:

- `src/pe3d/`
- `README_PYPI.md`
- `LICENSE`
- `NOTICE.md`
- packaging metadata

Before upload, confirm the generated metadata no longer points at the upstream
`G3Point` repository as the package homepage or source URL. Use the Python fork
repository URL instead.

Also confirm the long description describes `pe3d` as a fork or
reimplementation of `G3Point`, not as the upstream MATLAB package itself.

## Upload To TestPyPI

```bash
.venv/bin/python -m twine upload --repository testpypi dist/*
```

If you use an API token, set:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=<your-testpypi-token>
```

## Upload To PyPI

Only after validating the TestPyPI package:

```bash
.venv/bin/python -m twine upload dist/*
```
