# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import ZAxisMetrics, batch_probiou,box_iou
from ultralytics.utils.plotting import output_to_z_target, plot_images
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import r2_score

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
        self.z_corr = "z_corr" in args and args["z_corr"]
        if(self.z_corr):
            self.downsampled_reference = torch.tensor(np.load("data_gen/ripples_downsampled.npy")/10000, device=self.device)-2



    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # is COCO
        # self.stats["pred_z"] = []
        # self.stats["target_z"] = []
        self.stats["z_pairs"] = []
        if(self.z_corr):
            self.downsampled_reference=self.downsampled_reference.to(self.device)


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
            rotated=False,
        )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.

        Note:
            The function does not return any value directly usable for metrics calculation. Instead, it provides an
            intermediate representation used for evaluating predictions against ground truth.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    def match_predictions(self, pred_classes, true_classes, iou):
        """
        Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape(N,).
            true_classes (torch.Tensor): Target class indices of shape(M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground of truth
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape(N,10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        gt_pred_matches = np.zeros((self.iouv.shape[0],true_classes.shape[0],pred_classes.shape[0] ), dtype=bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        # gt_idx = gt_idx.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):

            # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708

            cost_matrix = iou * (iou >= threshold)
            if cost_matrix.any():
                labels_idx, detections_idx = linear_sum_assignment(cost_matrix, maximize=True)
                valid = cost_matrix[labels_idx, detections_idx] > 0
                if valid.any():
                    correct[detections_idx[valid], i] = True
                    gt_pred_matches[i,labels_idx,detections_idx] = valid
            # else:
            #     matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
            #     matches = np.array(matches).T
            #     if matches.shape[0]:
            #         if matches.shape[0] > 1:
            #             matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            #             matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            #             # matches = matches[matches[:, 2].argsort()[::-1]]
            #             matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            #         correct[matches[:, 1].astype(int), i] = True
            #         matched_z[matches[:, 1].astype(int), i] = true_z[]
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device),torch.tensor(gt_pred_matches,device=pred_classes.device)
    def _prepare_batch(self, si, batch):
        """Prepares and returns a batch for OBB validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        z_position = batch["z"][idx].squeeze(-1)
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "z": z_position,"ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad, "img":batch["img"][si]}

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                z_pairs=torch.zeros(len(batch["cls"]), self.niou,2, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox,gt_z = pbatch.pop("cls"), pbatch.pop("bbox"),pbatch.pop("z")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            # stat["target_z"] = z
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            pred_z = self.z_correlation(predn[:,:4],pbatch.pop("img"), pbatch["ori_shape"], pbatch["ratio_pad"]) if self.z_corr else predn[:,6].squeeze(-1)

            # Evaluate
            if nl:
                stat["tp"],gt_pred_matcher = self._process_batch(predn, bbox, cls)
                matched = (pred_z*gt_pred_matcher).sum(axis=2) #z predictions matched to ground truth for different iou values, unmatched are 0
                matched[~gt_pred_matcher.sum(axis=2).type(torch.bool)] = torch.nan #set unmatched detections to nan
                stat["z_pairs"] = torch.cat([gt_z.expand((len(matched),10)).unsqueeze(-1),matched.unsqueeze(-1)],2).transpose(0,1) #paired up z-values
                #stat["z_pairs"] = [[pair for pair in row if not torch.any(torch.isnan(pair))] for row in z_pairs] #turn into list where nans are excluded
                # stat["tp_z"] = self.
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)

            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    # pred_z,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )
    
    def z_correlation(self,bbox,img,ori_shape,rpad):
        x1,y1,x2,y2 = bbox.T
        w,h = x2-x1,y2-y1

        mask = x1==0
        x1[mask]=x1[mask]-h[mask]-w[mask]
        w[mask] = h[mask]

        mask = y1==0
        y1[mask]=y1[mask]-w[mask]-h[mask]
        h[mask] = w[mask]

        mask = x2 >= 511
        x2[mask]=x2[mask]+h[mask]-w[mask]
        w[mask] = h[mask]

        mask = y2 >= 511
        y2[mask]=y2[mask]+w[mask]-h[mask]
        h[mask] = w[mask]

        x,y = 0.5*(x1+x2),0.5*(y1+y2)
        # print(x,y,w,h)
        # plt.figure(1)
        # rect = Rectangle((x1,y1),w,h, linewidth=1, edgecolor="blue", facecolor='none')
        # plt.scatter(x,y,c="r",marker="x")
        # plt.text(*rect.get_xy(),f"{i},z={z:.3f}")
        # plt.gca().add_patch(rect)
        # plt.figure(2)
        # plt.subplot(1,len(res),i+1)
        size=128
        padding = 200
        extra_padding = padding-max(rpad[1])
        y = torch.round(y).type(torch.int).clamp(-size,511+size)
        x = torch.round(x).type(torch.int).clamp(-size,511+size)
        padded_image = torch.nn.functional.pad(img,(extra_padding,)*4,mode="constant", value=0.5)
        captures = torch.cat([padded_image[:,y[i]+padding-size//2:y[i]+padding+size//2,x[i]+padding-size//2:x[i]+padding+size//2].unsqueeze(0)-0.5 for i in range(len(x))],0)

        v = (torch.max(w,h)*(2)-55).unsqueeze(0)

        z  = torch.cat([(-v/0.21)+761,(v/0.21)+761], 0).round().type(torch.int).clamp(0,len(self.downsampled_reference)-1)

        correlations = self.downsampled_reference[z]*captures[:,0]
        return z[correlations.sum((-1,-2)).argmax(0)].diag()/1568

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
    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
            z=batch["z"]

        )


    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        from ultralytics.engine.results import Results
        import numpy as np
        # zaxis = torch.cat([predn, pred_z.view(-1,1)],1)

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            zaxis=predn
        ).save_txt(file, save_conf=save_conf)

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes","z"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = [
                torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                for i in range(nb)
            ]

        return batch
    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 8) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)","Z-axis MSE","Z-axis R2")