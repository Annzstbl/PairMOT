import numpy as np


def soft_nms(boxes_in, sigma=0.5, threshold=0.3, score_threshold=0.001,
             method=0):
    """Pure NumPy Soft-NMS for CTracker's paired 8-coordinate boxes."""
    boxes = np.asarray(boxes_in, dtype=np.float32).copy()
    count = boxes.shape[0]
    indices = np.arange(count)

    i = 0
    while i < count:
        max_pos = i + np.argmax(boxes[i:count, 8])
        if max_pos != i:
            boxes[[i, max_pos]] = boxes[[max_pos, i]]
            indices[[i, max_pos]] = indices[[max_pos, i]]

        pos = i + 1
        while pos < count:
            xx1 = max(boxes[i, 0], boxes[pos, 0])
            yy1 = max(boxes[i, 1], boxes[pos, 1])
            xx2 = min(boxes[i, 2], boxes[pos, 2])
            yy2 = min(boxes[i, 3], boxes[pos, 3])
            width = max(0.0, xx2 - xx1 + 1.0)
            height = max(0.0, yy2 - yy1 + 1.0)

            if width > 0 and height > 0:
                area_i = ((boxes[i, 2] - boxes[i, 0] + 1.0) *
                          (boxes[i, 3] - boxes[i, 1] + 1.0))
                area_pos = ((boxes[pos, 2] - boxes[pos, 0] + 1.0) *
                            (boxes[pos, 3] - boxes[pos, 1] + 1.0))
                overlap = width * height / (area_i + area_pos - width * height)

                if method == 1:
                    weight = 1.0 - overlap if overlap > threshold else 1.0
                elif method == 2:
                    weight = np.exp(-(overlap * overlap) / sigma)
                else:
                    weight = 0.0 if overlap > threshold else 1.0
                boxes[pos, 8] *= weight

                if boxes[pos, 8] < score_threshold:
                    boxes[pos] = boxes[count - 1]
                    indices[pos] = indices[count - 1]
                    count -= 1
                    continue
            pos += 1
        i += 1

    return boxes[:count], indices[:count]


def cython_soft_nms_wrapper(thresh, sigma=0.5, score_thresh=0.001, method='linear'):
    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)
    def _nms(dets):
        dets, _ = soft_nms(
            np.ascontiguousarray(dets, dtype=np.float32), sigma, thresh,
            score_thresh, methods[method]
        )
        return dets
    return _nms


def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def wnms_wrapper(thresh_lo, thresh_hi):
    def _nms(dets):
        return py_weighted_nms(dets, thresh_lo, thresh_hi)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return dets[keep, :]


def py_weighted_nms(dets, thresh_lo, thresh_hi):
    """
    voting boxes with confidence > thresh_hi
    keep boxes overlap <= thresh_lo
    rule out overlap > thresh_hi
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh_lo: retain overlap <= thresh_lo
    :param thresh_hi: vote overlap > thresh_hi
    :return: indexes to keep
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order] - inter)

        inds = np.where(ovr <= thresh_lo)[0]
        inds_keep = np.where(ovr > thresh_hi)[0]
        if len(inds_keep) == 0:
            break

        order_keep = order[inds_keep]

        tmp=np.sum(scores[order_keep])
        x1_avg = np.sum(scores[order_keep] * x1[order_keep]) / tmp
        y1_avg = np.sum(scores[order_keep] * y1[order_keep]) / tmp
        x2_avg = np.sum(scores[order_keep] * x2[order_keep]) / tmp
        y2_avg = np.sum(scores[order_keep] * y2[order_keep]) / tmp

        keep.append([x1_avg, y1_avg, x2_avg, y2_avg, scores[i]])
        order = order[inds]
    return np.array(keep)
