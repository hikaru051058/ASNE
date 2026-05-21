# TRIBE v2 Import Diagnostic

## Current Failure

From the ASNE repository root, this fails:

```bash
python -c "from tribev2 import TribeModel; print(TribeModel)"
```

Observed error:

```text
ImportError: cannot import name 'TribeModel' from 'tribev2' (unknown location)
```

From the ASNE root, `import tribev2` resolves to a namespace package:

```text
<module 'tribev2' (<_frozen_importlib_external.NamespaceLoader object ...>)>
['__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__']
```

That namespace package does not expose `TribeModel`.

## Commands Tried

```bash
find . -maxdepth 4 -type f | grep -i "tribe\|setup\|pyproject\|requirements"
python -c "import sys; print('\n'.join(sys.path))"
python -c "import tribev2; print(tribev2); print(dir(tribev2)[:50])"
python -c "from tribev2 import TribeModel; print(TribeModel)"
grep -R "class TribeModel\|def from_pretrained\|TribeModel" -n . | head -100
sed -n '1,220p' tribev2/tribev2/__init__.py
PYTHONPATH=./tribev2 python -c "from tribev2 import TribeModel; print(TribeModel)"
python -m pip show tribev2
cd /tmp && python -c "import tribev2; print(tribev2); from tribev2 import TribeModel; print(TribeModel)"
```

## Package Layout Observations

The ASNE repository contains a local TRIBE v2 checkout:

```text
./tribev2/pyproject.toml
./tribev2/tribev2/__init__.py
./tribev2/tribev2/demo_utils.py
```

`TribeModel` is defined in:

```text
./tribev2/tribev2/demo_utils.py
```

The inner package initializer exports it:

```python
from tribev2.demo_utils import TribeModel

__all__ = ["TribeModel"]
```

However, from the ASNE root, Python sees the outer `./tribev2` directory first. Because the outer directory does not have `./tribev2/__init__.py`, Python treats it as a namespace package and does not reach the inner `./tribev2/tribev2/__init__.py` package export.

The active environment also has a separate editable TRIBE v2 install:

```text
Name: tribev2
Version: 0.1.0
Editable project location: /Users/hikaru/tribev2
```

Outside the ASNE root, that installed editable package imports successfully:

```text
<module 'tribev2' from '/Users/hikaru/tribev2/tribev2/__init__.py'>
<class 'tribev2.demo_utils.TribeModel'>
```

This confirms that the local ASNE `./tribev2` folder is shadowing the installed editable package when commands are run from `/Users/hikaru/ASNE`.

## Dependency and Install Metadata

TRIBE v2 metadata is in:

```text
tribev2/pyproject.toml
```

Its core dependencies include heavy ML/neuroimaging packages:

```text
neuralset==0.0.2
neuraltrain==0.0.2
torch>=2.5.1,<2.7
torchvision>=0.20,<0.22
x_transformers==1.27.20
moviepy>=2.2.1
huggingface_hub
gtts
spacy
julius
transformers
```

Because these dependencies include PyTorch and related packages, ASNE should not blindly run `pip install -e ./tribev2` as part of normal setup.

## Likely Root Cause

The likely root cause is import shadowing caused by the nested local checkout:

```text
/Users/hikaru/ASNE/tribev2
```

When running Python from `/Users/hikaru/ASNE`, the current directory appears first on `sys.path`. Python finds `./tribev2` before the installed editable package at `/Users/hikaru/tribev2`, but `./tribev2` is the repository wrapper directory, not the importable inner package.

## Recommended Next Command

For ASNE's built-in adapter diagnostic from the repository root:

```bash
python -m asne.cli check-adapter --adapter tribev2 --tribev2-package-path ./tribev2
```

This prepends `./tribev2` only during lazy import and only checks that `TribeModel` is importable. It does not call `from_pretrained`, download model weights, or run inference.

For a manual temporary diagnostic import from the ASNE root:

```bash
PYTHONPATH=./tribev2 python -c "from tribev2 import TribeModel; print(TribeModel)"
```

This succeeded during diagnostics and does not download model weights.

For a more durable local setup, choose one of these approaches:

```bash
python -m pip install -e ./tribev2 --no-deps
```

or avoid keeping the upstream checkout at `./tribev2` inside ASNE if another editable TRIBE v2 install is already active.

Use `--no-deps` only if the required TRIBE v2 dependencies are already installed. Running `pip install -e ./tribev2` without `--no-deps` may install or change heavy dependencies such as PyTorch.

## Risks and Notes

- Importing `TribeModel` does not download model weights by itself.
- Calling `TribeModel.from_pretrained("facebook/tribev2", ...)` may download upstream model files through Hugging Face if they are not already cached.
- TRIBE v2 uses a separate upstream license, reported by local metadata as CC BY-NC 4.0. ASNE should not redistribute TRIBE v2 model weights, checkpoints, or third-party assets.
- ASNE mock mode should remain the default until real TRIBE v2 input mapping, output shapes, ROI metadata, and license handling are verified.
