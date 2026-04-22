# Models

This package is split by model family and uses consistent canonical module names.

## Layout

- `ivit/`: I-ViT / DeiT / Swin integer-only Relay modules.
  Canonical names: `builder.py`, `layers.py`, `vit.py`, `swin.py`, `utils.py`.
- `repq/`: RepQ-ViT Relay modules.
  Canonical names: `builder.py`, `layers.py`, `vit.py`, `swin.py`.
- `models/__init__.py`: compatibility aliases for old module names.

## Recommended Imports

- I-ViT: `from models.ivit import builder`
- RepQ: `from models.repq import builder`
- Legacy imports such as `import models.build_model` and `import models.repq_layers` still work via aliases.

## Notes

- There are no duplicate source files at the top level anymore.
- Old names are preserved only as import aliases, not as copied modules.
