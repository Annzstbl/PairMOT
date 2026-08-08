import json
import tempfile
import unittest
from pathlib import Path

from projects.multispec_pair_rotated_rtdetr.tools \
    .verify_decoder_epoch72_goal import find_epoch_eval


class TestVerifyDecoderEpoch72Goal(unittest.TestCase):

    def test_explicit_payload_step_for_evaluation_only_runner(self):
        with tempfile.TemporaryDirectory() as temporary:
            work_dir = Path(temporary)
            eval_dir = work_dir / 'val_track_eval' / 'val_track_0001'
            eval_dir.mkdir(parents=True)
            payload = {'step': 1}
            (eval_dir / 'async_track_eval_payload.json').write_text(
                json.dumps(payload), encoding='utf-8')

            found_dir, found_payload = find_epoch_eval(
                work_dir, epoch=72, payload_step=1)
            self.assertEqual(found_dir, eval_dir)
            self.assertEqual(found_payload, payload)

            with self.assertRaises(RuntimeError):
                find_epoch_eval(work_dir, epoch=72)


if __name__ == '__main__':
    unittest.main()
