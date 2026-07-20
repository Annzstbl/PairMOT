"""Stage-2 HSMOT experiment sampling frame offsets in [-2, 2]."""

from exps.example.mot.yolo11l_diffusion_track_hsmot import Exp as BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.exp_name = "yolo11l_diffusion_track_hsmot_inter2"
        self.pair_interval = 2
        self.interval = 2
