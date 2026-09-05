from metrics.depth_metrics import DepthMetrics
from metrics.segment_metrics import SegmentationMetrics

class MultiTaskMetrics:
    def __init__(self, num_classes=41):
        self.num_classes = num_classes
        self.depth_meter = DepthMetrics()
        self.seg_meter = SegmentationMetrics(num_classes=num_classes)

    def reset(self):
        self.depth_meter.reset()
        self.seg_meter.reset()

    def update(self, pred_seg, pred_depth, labels, depths):
        self.depth_meter.update(pred_depth, depths)
        self.seg_meter.update(pred_seg, labels)

    def compute(self, pred_seg=None, pred_depth=None, labels=None, depths=None):
        if pred_seg is not None and labels is not None:
            # Batch mode for backwards compatibility
            d_res = DepthMetrics.compute(pred_depth, depths)
            s_res = SegmentationMetrics.compute(pred_seg, labels, self.num_classes)
            return {
                "depth": d_res,
                "segmentation": s_res,
            }
        # Accumulated dataset mode
        return {
            "depth": self.depth_meter.get_results(),
            "segmentation": self.seg_meter.get_results(),
        }

    