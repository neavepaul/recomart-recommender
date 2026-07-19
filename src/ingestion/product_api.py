"""Mock REST API backed by the two RetailRocket item-property files."""

from __future__ import annotations

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from src.ingestion.item_property_csv import read_item_properties


def property_rows(raw_dir: Path, offset: int, limit: int):
    skipped = emitted = 0
    for filename in ("item_properties_part1.csv", "item_properties_part2.csv"):
        for row in read_item_properties(raw_dir / filename):
            if skipped < offset:
                skipped += 1
                continue
            if emitted >= limit:
                return
            emitted += 1
            yield row


def handler_for(raw_dir: Path):
    class ProductApiHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/health":
                self._json({"status": "ok"})
                return
            if parsed.path != "/item-properties":
                self.send_error(404)
                return
            query = urllib.parse.parse_qs(parsed.query)
            try:
                offset = max(0, int(query.get("offset", ["0"])[0]))
                limit = min(10_000, max(1, int(query.get("limit", ["1000"])[0])))
            except ValueError:
                self.send_error(400, "offset and limit must be integers")
                return
            items = list(property_rows(raw_dir, offset, limit))
            self._json({
                "items": items,
                "offset": offset,
                "next_offset": offset + len(items),
                "has_more": len(items) == limit,
            })

        def _json(self, payload: object):
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return ProductApiHandler


def make_server(host: str, port: int, raw_dir: Path) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), handler_for(raw_dir))
