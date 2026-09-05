from metrics.depth_metrics import DepthMetrics
from metrics.segment_metrics import SegmentationMetrics

class MultiTaskMetrics:

    def __init__(self, num_classes=41):
        self.num_classes = num_classes

    def compute(
        self,
        pred_seg,
        pred_depth,
        labels,
        depths,
    ):

        depth_metrics = DepthMetrics.compute(
            pred_depth,
            depths
        )

        seg_metrics = SegmentationMetrics.compute(
            pred_seg,
            labels,
            self.num_classes
        )

        return {
            "depth": depth_metrics,
            "segmentation": seg_metrics,
        }

    