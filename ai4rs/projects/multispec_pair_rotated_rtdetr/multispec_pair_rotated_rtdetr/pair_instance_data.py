# Copyright (c) AI4RS. All rights reserved.
"""Pair detection / tracking instance container (M4)."""

from mmengine.structures import InstanceData


class PairInstanceData(InstanceData):
    """Structured pair prediction output.

    Typical fields after ``PairRotatedRTDETRHead.predict``:

    - ``scores`` (Tensor): ``(num_instances,)`` shared cls confidence.
    - ``labels`` (Tensor): ``(num_instances,)`` class ids.
    - ``bboxes_prev`` (Tensor): ``(num_instances, 5)`` prev-frame OBB
      ``(cx, cy, w, h, angle)`` in image space.
    - ``bboxes_curr`` (Tensor): ``(num_instances, 5)`` curr-frame OBB.
    - ``presence_prev`` (Tensor): ``(num_instances,)`` prev visibility prob.
    - ``presence_curr`` (Tensor): ``(num_instances,)`` curr visibility prob.
    """
