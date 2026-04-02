# Security Policy

## Model Loading

- **sklearn models** (.pkl): Loaded via `joblib.load()` which uses pickle internally. Only load models you created yourself or trust completely. Pickle can execute arbitrary code.
- **PyTorch models** (.pt): Loaded with `torch.load(weights_only=True)` which restricts deserialization to safe types (tensors, dicts, lists). This is the secure default.
- **Never load .pkl or .pt files from untrusted sources.**

## Data Files

- `.npz` recordings are loaded with `allow_pickle=False` — safe against code injection.
- `SYNTHETIC_` prefixed files are clearly marked as non-real data.

## Network

- BrainFlow communicates via Bluetooth LE or WiFi (local only).
- No data is sent to any remote server.
- The GUI does not open any network ports.

## Dependencies

- All dependencies are from PyPI (pip). Verify package authenticity if concerned.
- The `install.sh` script uses `pip install` without `--trusted-host` overrides.

## Reporting Vulnerabilities

If you discover a security issue, please open a private issue on the GitHub repository or contact the maintainers directly.
