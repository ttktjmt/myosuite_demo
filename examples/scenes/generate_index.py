from pathlib import Path
import json
import os

_HERE = Path(__file__).parent

_ALLOWED_EXTENSIONS = [".xml", ".png", ".stl", ".obj", ".mjb"]

if __name__ == "__main__":
    files_to_download = []
    for path in _HERE.rglob("*"):
      if path.is_file() and path.suffix in _ALLOWED_EXTENSIONS:
         files_to_download.append(str(path.relative_to(_HERE)))
    files_to_download.sort()
    print(f"{files_to_download = }")
    with open(os.path.join(_HERE, "index.json"), mode="w") as f:
        json.dump(files_to_download, f, indent=2)
