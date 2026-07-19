import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import recomart
from src.ingestion.item_property_csv import read_item_properties


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.raw = root / "raw"
        self.raw.mkdir()
        self.db = root / "test.db"
        self.write("events.csv", ["timestamp", "visitorid", "event", "itemid", "transactionid"], [
            [1000, 1, "view", 10, ""], [2000, 1, "addtocart", 10, ""],
            [3000, 1, "transaction", 10, 99], [4000, 2, "invalid", 10, ""],
        ])
        props = [[1000, 10, "categoryid", "7"], [1000, 10, "available", "0"],
                 [2000, 10, "available", "1"], [1000, 10, "400", "n1"]]
        self.write("item_properties_part1.csv", ["timestamp", "itemid", "property", "value"], props)
        self.write("item_properties_part2.csv", ["timestamp", "itemid", "property", "value"], [])
        self.write("category_tree.csv", ["categoryid", "parentid"], [[7, 3]])

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, header, rows):
        with (self.raw / name).open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh); writer.writerow(header); writer.writerows(rows)

    def test_end_to_end(self):
        with patch.object(recomart, "RAW", self.raw):
            recomart.ingest_categories(self.db)
            recomart.replay_events(self.db)
            server = recomart.make_server("127.0.0.1", 0)
            import threading
            thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
            try:
                recomart.ingest_products(self.db, f"http://127.0.0.1:{server.server_address[1]}", 2)
            finally:
                server.shutdown(); server.server_close(); thread.join()
            recomart.transform(self.db, 16)
            report = recomart.validate(self.db)
        self.assertTrue(report["ok"])
        self.assertEqual(report["counts"]["silver_user_events"], 3)
        db = recomart.connect(self.db)
        self.assertEqual(db.execute("SELECT available FROM silver_products").fetchone()[0], 1)
        self.assertEqual(db.execute("SELECT interaction_score FROM gold_user_item_features").fetchone()[0], 9)
        db.close()

    def test_property_reader_preserves_quoted_and_unquoted_commas(self):
        path = self.raw / "comma_values.csv"
        path.write_text(
            'timestamp,itemid,property,value\n'
            '1000,10,description,"red, large"\n'
            '2000,11,description,blue,small,sale\n',
            encoding="utf-8",
        )
        rows = list(read_item_properties(path))
        self.assertEqual(rows[0]["value"], "red, large")
        self.assertEqual(rows[1]["value"], "blue,small,sale")


if __name__ == "__main__":
    unittest.main()
