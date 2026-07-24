"""
One-off diagnostic: list the real Step Index symbol codes on the new Deriv
Options API for your account, since the platform's default.yaml config had
guessed codes (STPRNG100-500) that Deriv rejects with InvalidSymbol.

Usage:
    python find_step_index_symbols.py

Reuses the same OTP bootstrap as the main client (same rest_base_url/app_id/
api_token/account_id from configs/default.yaml), so no extra credentials
needed. Prints every active symbol whose name/market mentions "step",
along with its correct code — copy those into configs/default.yaml's
market_data.connection.symbols list.
"""

from __future__ import annotations

import asyncio
import json

import websockets

from configs.loader import load_config
from data.deriv_client import DerivOTPBootstrap


async def main() -> None:
    cfg = load_config("configs/default.yaml")
    conn_cfg = cfg.market_data.connection

    if conn_cfg.is_authenticated_mode:
        otp = DerivOTPBootstrap(conn_cfg)
        url = await otp.fetch_authenticated_ws_url()
    else:
        url = conn_cfg.ws_public_url

    print("Connecting...\n")

    async with websockets.connect(url, ping_interval=conn_cfg.ping_interval_seconds) as ws:
        await ws.send(json.dumps({"active_symbols": "brief", "req_id": 1}))
        async for raw in ws:
            message = json.loads(raw)
            if message.get("error"):
                print("ERROR:", message["error"])
                return
            if message.get("msg_type") == "active_symbols":
                symbols = message["active_symbols"]
                matches = [
                    s for s in symbols
                    if "step" in json.dumps(s).lower()
                ]
                if not matches:
                    print(
                        "No symbols matched 'step'. Full list has "
                        f"{len(symbols)} symbols — dumping first 20 for reference:"
                    )
                    matches = symbols[:20]
                print(f"{'CODE':<20} {'NAME':<30} MARKET")
                for s in matches:
                    code = s.get("underlying_symbol") or s.get("symbol") or "?"
                    name = s.get("underlying_symbol_name") or s.get("display_name") or "?"
                    market = s.get("market") or "?"
                    print(f"{code:<20} {name:<30} {market}")
                return


if __name__ == "__main__":
    asyncio.run(main())
