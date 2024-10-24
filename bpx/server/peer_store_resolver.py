from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional


class PeerStoreResolver:
    """
    Determines the peers data file path using values from the config
    """

    def __init__(
        self,
        root_path: Path,
        config: Dict[str, Any],
        *,
        selected_network: str,
        peers_file_path_key: str,  # config key for the peers data file relative path
        default_peers_file_path: str,  # default value for the peers data file relative path
    ):
        self.root_path = root_path
        self.config = config
        self.selected_network = selected_network
        self.peers_file_path_key = peers_file_path_key

    def _resolve_and_update_config(self) -> Path:
        """
        Resolve the peers data file path from the config, and update the config if necessary.

        """
        peers_file_path_str: Optional[str] = self.config.get(self.peers_file_path_key)
        if peers_file_path_str is None:
            peers_file_path_str = os.fspath(Path(self.default_peers_file_path).parent / self._peers_file_name)
            # Update the config
            self.config[self.peers_file_path_key] = peers_file_path_str
        return self.root_path / Path(peers_file_path_str)

    @property
    def _peers_file_name(self) -> str:
        """
        Internal property to get the name component of the peers data file path
        """
        if self.selected_network == "mainnet":
            return Path(self.default_peers_file_path).name
        else:
            # For testnets, we include the network name in the peers data filename
            path = Path(self.default_peers_file_path)
            return path.with_name(f"{path.stem}_{self.selected_network}{path.suffix}").name

    @property
    def peers_file_path(self) -> Path:
        """
        Path to the peers data file, resolved using data from the config
        """
        return self._resolve_and_update_config()
