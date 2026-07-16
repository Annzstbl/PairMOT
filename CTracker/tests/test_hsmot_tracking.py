import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from test_hsmot import (Detection, Track, match_tracks,
                        track_detection_similarity)


def detection(frame, x, next_x, label=0):
    prev_box = torch.tensor([x, 10.0, 8.0, 4.0, 0.0])
    curr_box = torch.tensor([next_x, 10.0, 8.0, 4.0, 0.0])
    return Detection(frame, prev_box, curr_box, 0.9, label)


class TestOriginalTrackingParity(unittest.TestCase):
    def test_adjacent_frame_uses_paired_next_box(self):
        first = detection(1, 10.0, 20.0)
        first.track_id = 7
        track = Track(first)
        current = detection(2, 20.0, 30.0)
        self.assertGreater(
            track_detection_similarity(track, current), 0.999)
        matches, unmatched_tracks, unmatched_detections = match_tracks(
            [track], [current], 0.5)
        self.assertEqual(matches, [(0, 0)])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])

    def test_gap_uses_original_linear_velocity_extrapolation(self):
        first = detection(1, 10.0, 12.0)
        track = Track(first)
        track.update(detection(2, 12.0, 14.0))
        torch.testing.assert_close(track.velocity, torch.tensor([2.0, 0.0]))
        after_gap = detection(4, 16.0, 18.0)
        self.assertGreater(
            track_detection_similarity(track, after_gap), 0.999)

    def test_six_frame_velocity_average_and_class_constraint(self):
        track = Track(detection(1, 2.0, 4.0, label=3))
        for frame in range(2, 7):
            track.update(detection(
                frame, float(frame * 2), float(frame * 2 + 2), label=3))
        torch.testing.assert_close(track.velocity, torch.tensor([2.0, 0.0]))
        wrong_class = detection(7, 14.0, 16.0, label=2)
        self.assertEqual(track_detection_similarity(track, wrong_class), 0)
        matches, unmatched_tracks, unmatched_detections = match_tracks(
            [track], [wrong_class], 0.5)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_detections, [0])


if __name__ == '__main__':
    unittest.main()
