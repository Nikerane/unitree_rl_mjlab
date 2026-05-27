from pathlib import Path

import mjlab.utils.os as _mjlab_os_module

if not hasattr(_mjlab_os_module, "update_assets"):
    def _update_assets(assets: dict, path, meshdir: str) -> None:
        """Compat shim: mjlab 1.3+ removed this; replicate its asset-loading logic."""
        asset_dir = Path(path) / meshdir if meshdir else Path(path)
        if asset_dir.exists():
            for f in asset_dir.iterdir():
                if f.is_file():
                    assets[f.name] = f.read_bytes()
    _mjlab_os_module.update_assets = _update_assets

SRC_PATH: Path = Path(__file__).parent