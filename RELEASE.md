# Release Instructions

## Prerequisites

1. Ensure all tests pass: `make test`
2. Ensure type checking passes: `make type`
3. Ensure linting passes: `make lint`
4. Ensure documentation is up to date
5. Update CHANGELOG.md with release notes

## Release Process

### 1. Test the Release

```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build the package
python -m build

# Test the package locally
pip install dist/stsw-1.0.0-py3-none-any.whl
python -c "import stsw; print(stsw.__version__)"
stsw selftest
```

### 2. Create Git Tag

```bash
# Ensure you're on main branch
git checkout main
git pull origin main

# Create annotated tag
git tag -a v1.0.0 -m "Release v1.0.0 - The Last-Word Safe-Tensor Stream Suite"

# Push tag to trigger release workflow
git push origin v1.0.0
```

### 3. Manual PyPI Upload (if needed)

If the automated release fails, you can manually upload:

```bash
# Install twine
pip install twine

# Upload to Test PyPI first (optional)
twine upload --repository testpypi dist/*

# Upload to PyPI
twine upload dist/*
```

### 4. Verify Release

1. Check GitHub Releases page
2. Check PyPI page: https://pypi.org/project/stsw/
3. Test installation from PyPI:
   ```bash
   pip install stsw
   python -c "import stsw; print(stsw.__version__)"
   ```

### 5. Post-Release

1. Update documentation if needed
2. Announce release (if applicable)
3. Start development on next version

## Notes

- The GitHub Actions workflow will automatically:
  - Build the package
  - Create a GitHub release
  - Upload to PyPI using trusted publishing
- Trusted publishing requires PyPI project configuration
- Always test on Test PyPI first for major releases