"""Compiles a Core AI model ahead of time for the Neural Engine of each iOS architecture.

`xcrun coreai-build` writes one package per architecture, `<model>.<architecture>.aimodelc`, and the
app loads the one `AIModel.deviceArchitectureName` names, with
`SpecializationOptions(preferredComputeUnitKind: .neuralEngine)`. The load options have to match the
compile options: otherwise the device compiles the model again when it loads, for minutes on a large
model.

The Neural Engine returns integer results mislabelled: an argmax declared Int32 holds int16 values at
twice its row stride, declared Int16 only its even elements, and cast to Float16 int16 bits read as
Float16. Models with integer outputs or index reductions are refused; compute such values with
Float16 operations instead, as omnivoice-coreml's `Step(float_tokens=True)` does for the argmax of
the tokens.
"""

import argparse
import json
import plistlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TypedDict

from huggingface_hub import HfApi

# The app runs on the A17 Pro and later.
ARCHITECTURES = ("h16p", "h17g", "h17p", "h18p", "h19p")


def refusals(model: Path) -> Iterator[str]:
    """The reasons `model` would compute wrong values on the Neural Engine."""
    inspected = subprocess.run(
        ["xcrun", "coreai-build", "inspect", str(model), "--json", "--io", "--ops", "--no-metadata"],
        check=True,
        capture_output=True,
        text=True,
    )
    summary: _Summary = json.loads(inspected.stdout)["summary"]
    yield from (
        f"{function['name']} returns {output['name']} as {output['type']}"
        for function in summary["functions"]
        for output in function["outputs"]
        if _INTEGER.search(output["type"])
    )
    yield from (
        f"{operation['count']} {operation['name']} operations"
        for operation in summary["operationDistribution"]
        if operation["name"] in _INDEX_REDUCTIONS
    )


def compile_for_neural_engine(model: Path, output: Path, architectures: Sequence[str]) -> tuple[Path, ...]:
    """Compiles `model` into `output` for `architectures`, or for every iOS architecture when empty.

    Returns the packages, which replace any of the same name in `output`.
    """

    def replace(package: Path) -> Path:
        target = output / package.name
        if target.exists():
            shutil.rmtree(target)
        return package.rename(target)

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as staging:
        subprocess.run(
            [
                "xcrun",
                "coreai-build",
                "compile",
                str(model),
                "--platform",
                "iOS",
                "--preferred-compute",
                "neural-engine",
                "--output",
                staging,
                *(flag for architecture in architectures for flag in ("--architecture", architecture)),
            ],
            check=True,
        )
        return tuple(replace(package) for package in sorted(Path(staging).glob("*.aimodelc")))


def partly_off_neural_engine(package: Path) -> Iterator[str]:
    """The functions of `package` that MPSGraph did not place entirely on the Neural Engine.

    Whatever the Neural Engine cannot run, MPSGraph places on another unit, and the app cannot
    keep the GPU out: iOS denies the GPU to apps in the background.
    """
    for manifest in package.glob("*-delegates/MPSGraph/mpsExecutable.mpsgraphpackage/manifest.plist"):
        with manifest.open("rb") as file:
            for attributes in _entry_function_attributes(plistlib.load(file)):
                yield from (
                    _SYMBOL_SUFFIX.sub("", symbol)
                    for symbol, flags in attributes.items()
                    if "mps.fullyPlacedOnANE" not in flags
                )


def _entry_function_attributes(value: object) -> Iterator[dict[str, list[object]]]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "Entry Function Attributes":
                yield child
            else:
                yield from _entry_function_attributes(child)
    elif isinstance(value, list):
        for child in value:
            yield from _entry_function_attributes(child)


def upload(packages: Sequence[Path], repo: str) -> str:
    """Uploads `packages` to the model repository `repo` and returns the commit, for the app to pin."""
    commit = HfApi().upload_folder(
        repo_id=repo,
        folder_path=packages[0].parent,
        allow_patterns=[f"{package.name}/**" for package in packages],
        commit_message=f"Add Neural Engine packages {', '.join(package.name for package in packages)}",
    )
    return commit.oid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("model", type=Path, help="the source model, an .aimodel")
    parser.add_argument("output", type=Path, help="the directory for the compiled packages")
    parser.add_argument(
        "--architecture", action="append", default=[], help="an architecture to compile for; ARCHITECTURES when omitted"
    )
    parser.add_argument("--upload", metavar="REPO", help="a Hugging Face model repository to upload to")
    args = parser.parse_args()

    if reasons := tuple(refusals(args.model)):
        raise SystemExit(
            "\n  ".join([f"{args.model} would compute wrong values on the Neural Engine:", *reasons])
        )
    packages = compile_for_neural_engine(args.model, args.output, args.architecture or ARCHITECTURES)
    print("\n".join(str(package) for package in packages))
    for package in packages:
        if functions := sorted(set(partly_off_neural_engine(package))):
            print(f"warning: {package.name} runs partly off the Neural Engine: {', '.join(functions)}")
    if args.upload:
        print(f"uploaded to {args.upload} at revision {upload(packages, args.upload)}")


class _Value(TypedDict):
    name: str
    type: str


class _Function(TypedDict):
    name: str
    outputs: list[_Value]


class _Operation(TypedDict):
    name: str
    count: int


class _Summary(TypedDict):
    functions: list[_Function]
    operationDistribution: list[_Operation]


_INTEGER = re.compile(r"\(U?Int\d+,")
# MPSGraph names an entry function `<function>_<22-character hash>_<index>`.
_SYMBOL_SUFFIX = re.compile(r"_[\w-]{22}_\d+$")
_INDEX_REDUCTIONS = {"argmax", "argmin", "reduce_index"}
