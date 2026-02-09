### Pasting all .py code into a single file with filenames

for f in *.py; do
echo "=== FILE: $f ==="
cat "$f"
done > code.txt

### Removing all .pyc, .pyo, .pyd, .so files and __pycache__ directories - garbage junk

find . -type f -name "*.pyc" -delete -o -type f -name "*.pyo" -delete -o -type f -name "*.pyd" -delete -o -type f
-name "*.so" -delete -o -type d -name "__pycache__" -exec rm -rf {} +
