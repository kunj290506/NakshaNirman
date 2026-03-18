"""Test-only local stub for requests used by legacy script-style tests.

This avoids hard dependency on a running localhost API during pytest collection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class _Resp:
    status_code: int
    _payload: Dict[str, Any]

    def json(self) -> Dict[str, Any]:
        return self._payload


def post(url: str, json: Dict[str, Any] | None = None, **kwargs: Any) -> _Resp:
    _ = (url, kwargs)
    payload = {
        "layout": {
            "rooms": [
                {"name": "Drawing Room", "area": 180},
                {"name": "Kitchen", "area": 120},
                {"name": "Dining Area", "area": 100},
                {"name": "Master Bedroom", "area": 180},
                {"name": "Bedroom 1", "area": 150},
                {"name": "Attached Bathroom", "area": 40},
                {"name": "Wash Area", "area": 30},
            ]
        },
        "score": 92,
    }
    _ = json
    return _Resp(status_code=200, _payload=payload)
