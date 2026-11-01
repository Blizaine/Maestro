"""Run Maestro's owner-provisioned local MCP connector over stdio."""
from __future__ import annotations

import argparse
from pathlib import Path

from services.agent_access.auth import AuthError
from services.agent_access.local_connector import (
    LocalCredentialError,
    run_local_stdio,
    validate_client_id,
)
from services.agent_access.state import RuntimeState


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--client", required=True, help="Saved local client ID")
    args = parser.parse_args()
    try:
        client_id = validate_client_id(args.client)
        app_root = Path(__file__).resolve().parent
        runtime = RuntimeState(
            app_root / "settings" / "agent-access",
            app_root=app_root,
        )
        run_local_stdio(runtime, client_id)
    except (AuthError, LocalCredentialError, OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
