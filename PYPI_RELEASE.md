# PyPI Release Instructions

## Prerequisites

1. PyPI account: https://pypi.org/account/register/
2. Twine installed: `pip install twine` ✓

## Step 1: Test on TestPyPI (Recommended)

First, test the upload on TestPyPI:

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --no-deps stsw
```

## Step 2: Upload to PyPI

Once tested, upload to the real PyPI:

```bash
# Upload to PyPI
twine upload dist/*
```

You'll be prompted for:
- Username: `__token__` (if using API token) or your PyPI username
- Password: Your API token or password

## Step 3: Verify Installation

```bash
# Install from PyPI
pip install stsw

# Test it works
stsw --version
stsw selftest
```

## Using API Tokens (Recommended)

1. Go to https://pypi.org/manage/account/token/
2. Create a new API token
3. Use it with twine:

```bash
twine upload dist/* -u __token__ -p pypi-YOUR-TOKEN-HERE
```

## Automated Release

The GitHub Actions workflow is configured to automatically publish to PyPI when you push a tag. To enable this:

1. Go to https://pypi.org/manage/project/stsw/settings/publishing/ (after first manual upload)
2. Add trusted publisher:
   - Repository owner: `just-do-halee`
   - Repository name: `stsw`
   - Workflow name: `release.yml`
   - Environment name: (leave blank)

Then future releases will be automatic when you push tags!