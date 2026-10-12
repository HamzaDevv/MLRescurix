import json
from pathlib import Path
import unittest
from mlrescurix_worker.protocol import decode_frame, encode_frame, ProtocolError
ROOT = Path(__file__).resolve().parents[2]
class ProtocolTests(unittest.TestCase):
    def test_shared_corpus(self):
        folder=ROOT/'contracts/v1/fixtures'
        for case in json.loads((folder/'manifest.json').read_text()):
            with self.subTest(case=case['file']):
                data=(folder/case['file']).read_bytes()
                if case['accepted']:
                    value=decode_frame(data)
                    self.assertEqual(decode_frame(encode_frame(value)),value)
                else:
                    with self.assertRaises(ProtocolError):decode_frame(data)
    def test_frame_bounds(self):
        for data in [b' '*65537, b'\xff', b'{', b'{"version":1,"kind":"observation","payload":NaN}', b'['*1000]:
            with self.assertRaises(ProtocolError):decode_frame(data)
    def test_schema_copy(self):
        self.assertEqual((ROOT/'contracts/v1/frame.schema.json').read_bytes(),(ROOT/'python/src/mlrescurix_worker/protocol/frame.schema.json').read_bytes())
if __name__=='__main__': unittest.main()
