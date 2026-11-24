from __future__ import annotations

import argparse
import shutil
import subprocess
import sys

from quant_workshop.native import LIBRARY_SUFFIX, NATIVE_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Build either optional native kernel.")
    parser.add_argument("language", choices=("cpp", "rust", "both"))
    language: str = parser.parse_args().language
    output_dir = NATIVE_DIR / "build"
    output_dir.mkdir(exist_ok=True)
    languages = ("cpp", "rust") if language == "both" else (language,)
    for current in languages:
        compiler = shutil.which("c++" if current == "cpp" else "rustc")
        if compiler is None:
            raise SystemExit(f"Missing {'C++ compiler' if current == 'cpp' else 'rustc'}.")
        output = output_dir / f"downside_{current}{LIBRARY_SUFFIX}"
        if current == "cpp":
            if sys.platform == "win32":
                raise SystemExit("Use Rust on Windows, or build C++ inside WSL.")
            shared = "-dynamiclib" if sys.platform == "darwin" else "-shared"
            command = [
                compiler,
                "-std=c++17",
                "-O3",
                "-fPIC",
                shared,
                str(NATIVE_DIR / "downside.cpp"),
                "-o",
                str(output),
            ]
        else:
            command = [
                compiler,
                "--edition=2021",
                "--crate-type=cdylib",
                "-C",
                "opt-level=3",
                str(NATIVE_DIR / "downside.rs"),
                "-o",
                str(output),
            ]
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        print(f"Built {output.name}")


if __name__ == "__main__":
    main()
