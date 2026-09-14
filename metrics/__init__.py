from metrics.depth_metrics import DepthMetrics
from metrics.segment_metrics import SegmentationMetrics
from metrics.boundary_metrics import BoundaryMetrics

class MultiTaskMetrics:
    def __init__(self, num_classes=41):
        self.num_classes = num_classes
        self.depth_meter = DepthMetrics()
        self.seg_meter = SegmentationMetrics(num_classes=num_classes)
        self.bound_meter = BoundaryMetrics()

    def reset(self):
        self.depth_meter.reset()
        self.seg_meter.reset()
        self.bound_meter.reset()

    def update(self, pred_seg, pred_depth, *args):
        if len(args) == 2:
            labels, depths = args
            self.depth_meter.update(pred_depth, depths)
            self.seg_meter.update(pred_seg, labels)
        elif len(args) == 4:
            pred_bound, labels, depths, bounds = args
            self.depth_meter.update(pred_depth, depths)
            self.seg_meter.update(pred_seg, labels)
            if pred_bound is not None and bounds is not None:
                self.bound_meter.update(pred_bound, bounds)
        else:
            raise ValueError(f"MultiTaskMetrics.update expects 4 or 6 positional arguments, got {2 + len(args)}")

    def compute(self, pred_seg=None, pred_depth=None, pred_bound=None, labels=None, depths=None, bounds=None):
        if pred_seg is not None and labels is not None:
            # Batch mode for backwards compatibility
            d_res = DepthMetrics.compute(pred_depth, depths)
            s_res = SegmentationMetrics.compute(pred_seg, labels, self.num_classes)
            b_res = BoundaryMetrics.compute(pred_bound, bounds)

            return {
                "depth": d_res,
                "segmentation": s_res,
                "boundary": b_res
            }
        # Accumulated dataset mode
        return {
            "depth": self.depth_meter.get_results(),
            "segmentation": self.seg_meter.get_results(),
            "boundary": self.bound_meter.get_results()
        }

    