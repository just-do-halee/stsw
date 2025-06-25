#!/bin/bash

echo "=== Publishing stsw to PyPI ==="
echo
echo "You have several options:"
echo
echo "1. Interactive upload (enter credentials when prompted):"
echo "   twine upload dist/*"
echo
echo "2. Using PyPI API token (recommended):"
echo "   twine upload dist/* -u __token__ -p <your-pypi-token>"
echo
echo "3. Using username/password:"
echo "   twine upload dist/* -u <your-username> -p <your-password>"
echo
echo "4. Using environment variables:"
echo "   export TWINE_USERNAME=__token__"
echo "   export TWINE_PASSWORD=<your-pypi-token>"
echo "   twine upload dist/*"
echo
echo "To get a PyPI API token:"
echo "1. Go to https://pypi.org/manage/account/token/"
echo "2. Create a new token"
echo "3. Copy the token (starts with 'pypi-')"
echo
echo "Ready to upload these files:"
ls -la dist/
echo
echo "Run one of the commands above to publish to PyPI!"