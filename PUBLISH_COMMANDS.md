# Publishing Commands

## 1. Publish to PyPI

### Option A: Direct to PyPI (if confident)
```bash
twine upload dist/*
```

### Option B: Test on TestPyPI first (recommended)
```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps stsw
```

### Using API Token (more secure)
```bash
# With API token
twine upload dist/* -u __token__ -p <your-pypi-token>
```

You'll need:
- PyPI account: https://pypi.org/account/register/
- API token: https://pypi.org/manage/account/token/

## 2. After PyPI is Live

Test the PyPI package:
```bash
# Create clean environment
python -m venv test_pypi
source test_pypi/bin/activate  # On Windows: test_pypi\Scripts\activate

# Install from PyPI
pip install stsw

# Test it
stsw --version
stsw --help

# With optional dependencies
pip install stsw[numpy]
stsw selftest

deactivate
```

## 3. Publish to npm

```bash
# Login to npm (first time only)
npm login

# Publish
npm publish

# Or dry run first
npm publish --dry-run
```

## 4. Test npm Package

```bash
# Install globally
npm install -g stsw

# Test it
stsw --version
stsw --help
```

## Quick Copy-Paste Commands

```bash
# 1. PyPI
twine upload dist/*

# 2. Test PyPI install
pip install stsw

# 3. npm  
npm publish
```