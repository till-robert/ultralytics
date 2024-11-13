# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import ZAxisMetrics, batch_probiou
from ultralytics.utils.plotting import output_to_z_target, plot_images


class ZAxisValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    Example:
        ```python
        from ultralytics.models.yolo.obb import OBBValidator

        args = dict(model="yolov8n-obb.pt", data="dota8.yaml")
        validator = OBBValidator(args=args)
        validator(model=args["model"])
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.args.task = "zaxis"
        self.metrics = ZAxisMetrics(save_dir=self.save_dir, plot=True, on_plot=self.on_plot)

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""
        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            rotated=True,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Perform computation of the correct prediction matrix for a batch of detections and ground truth bounding boxes.

        Args:
            detections (torch.Tensor): A tensor of shape (N, 7) representing the detected bounding boxes and associated
                data. Each detection is represented as (x1, y1, x2, y2, conf, class, angle).
            gt_bboxes (torch.Tensor): A tensor of shape (M, 5) representing the ground truth bounding boxes. Each box is
                represented as (x1, y1, x2, y2, angle).
            gt_cls (torch.Tensor): A tensor of shape (M,) representing class labels for the ground truth bounding boxes.

        Returns:
            (torch.Tensor): The correct prediction matrix with shape (N, 10), which includes 10 IoU (Intersection over
                Union) levels for each detection, indicating the accuracy of predictions compared to the ground truth.

        Example:
            ```python
            detections = torch.rand(100, 7)  # 100 sample detections
            gt_bboxes = torch.rand(50, 5)  # 50 sample ground truth boxes
            gt_cls = torch.randint(0, 5, (50,))  # 50 ground truth class labels
            correct_matrix = OBBValidator._process_batch(detections, gt_bboxes, gt_cls)
            ```

        Note:
            This method relies on `batch_probiou` to calculate IoU between detections and ground truth bounding boxes.
        """
        iou = batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        z_position = batch["z"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        return {"cls": cls, "bbox": bbox, "z": z_position,"ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares and returns a batch for OBB validation with scaled and padded bounding boxes."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        return predn

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        batch_id, class_id, box,z, conf = output_to_z_target(preds, max_det=self.args.max_det)
        plot_images(
            batch["img"],
            batch_id, class_id, box, conf,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
            z=z
        )  # pred



