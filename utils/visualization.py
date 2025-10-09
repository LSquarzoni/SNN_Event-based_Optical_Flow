import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Visualization:
    """
    Utility class for the visualization and storage of rendered image-like representation
    of multiple elements of the optical flow estimation and image reconstruction pipeline.
    """

    def __init__(self, kwargs, eval_id=-1, path_results=None, vis_type="gradients"):
        self.img_idx = 0
        self.px = kwargs["vis"]["px"]
        self.color_scheme = "green_red"  # gray / blue_red / green_red
        self.last_store_ts = None  # for controlling store rate
        self.store_interval = kwargs["vis"].get("store_interval", 5.0)  # seconds
        self.vis_type = vis_type
        self.store_type = kwargs["vis"].get("store_type", "image")  # 'image' or 'video'
        self.video_writers = {}

        if eval_id >= 0 and path_results is not None:
            self.store_dir = path_results + "results/"
            self.store_dir = self.store_dir + "eval_" + str(eval_id) + "/"
            if not os.path.exists(self.store_dir):
                os.makedirs(self.store_dir)
            self.store_file = None

    def update(self, inputs, flow, iwe, events_window=None, masked_window_flow=None, iwe_window=None):
        """
        Live visualization.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        """

        events = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None
        frames = inputs["frames"] if "frames" in inputs.keys() else None
        gtflow = inputs["gtflow"] if "gtflow" in inputs.keys() else None
        
        # Get dimensions from events if available, otherwise from flow or gtflow
        if events is not None:
            height = events.shape[2]
            width = events.shape[3]
        elif flow is not None:
            height = flow.shape[2]
            width = flow.shape[3]
        elif gtflow is not None:
            height = gtflow.shape[2]
            width = gtflow.shape[3]
        else:
            height, width = 256, 256  # fallback

        # input events
        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        cv2.namedWindow("Input Events", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Input Events", int(self.px), int(self.px))
        cv2.imshow("Input Events", self.events_to_image(events_npy))

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            cv2.namedWindow("Input Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Events - Eval window", int(self.px), int(self.px))
            # Show flow visualizations only if masked flow is provided
            if masked_window_flow is not None:
                mflow = masked_window_flow.detach()
                m_h, m_w = mflow.shape[2], mflow.shape[3]
                masked_window_flow_npy = mflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((m_h, m_w, 2))
                # Show gradients visualization
                masked_grad_img = self.flow_to_image(
                    masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
                )
                masked_grad_img = cv2.cvtColor(masked_grad_img, cv2.COLOR_RGB2BGR)
                cv2.namedWindow("Estimated Flow - Eval window (gradients)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Estimated Flow - Eval window (gradients)", int(self.px), int(self.px))
                cv2.imshow("Estimated Flow - Eval window (gradients)", masked_grad_img)

                # Show vectors visualization (overlay GT average arrow if available)
                gt_x = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 0] if gtflow is not None else None
                gt_y = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 1] if gtflow is not None else None
                masked_vec_img = self.flow_to_vector(
                    masked_window_flow_npy[:, :, 0],
                    masked_window_flow_npy[:, :, 1],
                    type="sparse",
                    center=True,
                    gt_flow_x=gt_x,
                    gt_flow_y=gt_y,
                    overlay_gt=True if gtflow is not None else False,
                    fixed_length=70,
                )
                cv2.namedWindow("Estimated Flow - Eval window (vectors)", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Estimated Flow - Eval window (vectors)", int(self.px), int(self.px))
                cv2.imshow("Estimated Flow - Eval window (vectors)", masked_vec_img)
            cv2.imshow("Input Events - Eval window", self.events_to_image(events_window_npy))

        # input frames
        if frames is not None:
            frame_image = np.zeros((height, 2 * width))
            frames = frames.detach()
            frames_npy = frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            frame_image[:height, 0:width] = frames_npy[:, :, 0] / 255.0
            frame_image[:height, width : 2 * width] = frames_npy[:, :, 1] / 255.0
            cv2.namedWindow("Input Frames (Prev/Curr)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Input Frames (Prev/Curr)", int(2 * self.px), int(self.px))
            cv2.imshow("Input Frames (Prev/Curr)", frame_image)

        # optical flow
        if flow is not None:
            flow = flow.detach()
            flow_h, flow_w = flow.shape[2], flow.shape[3]
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((flow_h, flow_w, 2))
            if self.vis_type == "vectors":
                # If ground-truth exists, overlay its average vector as a white arrow behind the predicted one
                gt_x = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 0] if gtflow is not None else None
                gt_y = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 1] if gtflow is not None else None
                flow_npy = self.flow_to_vector(
                    flow_npy[:, :, 0],
                    flow_npy[:, :, 1],
                    type="sparse",
                    center=True,
                    gt_flow_x=gt_x,
                    gt_flow_y=gt_y,
                    overlay_gt=True if gtflow is not None else False,
                    fixed_length=70,
                )
            else:
                flow_npy = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
                flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow", flow_npy)

        # optical flow (masked)
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_h, masked_w = masked_window_flow.shape[2], masked_window_flow.shape[3]
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((masked_h, masked_w, 2))
            if self.vis_type == "vectors":
                # Overlay white average GT arrow if ground-truth is available
                gt_x = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 0] if gtflow is not None else None
                gt_y = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 1] if gtflow is not None else None
                masked_window_flow_npy = self.flow_to_vector(
                    masked_window_flow_npy[:, :, 0],
                    masked_window_flow_npy[:, :, 1],
                    type="sparse",
                    center=True,
                    gt_flow_x=gt_x,
                    gt_flow_y=gt_y,
                    overlay_gt=True if gtflow is not None else False,
                    fixed_length=70,
                )
            else:
                masked_window_flow_npy = self.flow_to_image(
                    masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
                )
                masked_window_flow_npy = cv2.cvtColor(masked_window_flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow - Eval window", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow - Eval window", masked_window_flow_npy)

        # ground-truth optical flow
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_h, gtflow_w = gtflow.shape[2], gtflow.shape[3]
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow_h, gtflow_w, 2))
            if self.vis_type == "vectors":
                gtflow_npy = self.flow_to_vector(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1], center=True)
            else:
                gtflow_npy = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
                gtflow_npy = cv2.cvtColor(gtflow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Ground-truth Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Ground-truth Flow", int(self.px), int(self.px))
            cv2.imshow("Ground-truth Flow", gtflow_npy)

        # image of warped events
        if iwe is not None:
            iwe = iwe.detach()
            iwe_h, iwe_w = iwe.shape[2], iwe.shape[3]
            iwe_npy = iwe.cpu().numpy().transpose(0, 2, 3, 1).reshape((iwe_h, iwe_w, 2))
            iwe_npy = self.events_to_image(iwe_npy)
            cv2.namedWindow("Image of Warped Events", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events", iwe_npy)

        # image of warped events - evaluation window
        if iwe_window is not None:
            iwe_window = iwe_window.detach()
            iwe_window_h, iwe_window_w = iwe_window.shape[2], iwe_window.shape[3]
            iwe_window_npy = iwe_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((iwe_window_h, iwe_window_w, 2))
            iwe_window_npy = self.events_to_image(iwe_window_npy)
            cv2.namedWindow("Image of Warped Events - Eval window", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Image of Warped Events - Eval window", int(self.px), int(self.px))
            cv2.imshow("Image of Warped Events - Eval window", iwe_window_npy)

        cv2.waitKey(1)

    def store(self, inputs, flow, iwe, sequence, events_window=None, masked_window_flow=None, iwe_window=None, ts=None):
        """
        Store rendered images.
        :param inputs: dataloader dictionary
        :param flow: [batch_size x 2 x H x W] optical flow map
        :param iwe: [batch_size x 1 x H x W] image of warped events
        :param sequence: filename of the event sequence under analysis
        :param ts: timestamp associated with rendered files (default = None)
        """

        events = inputs["event_cnt"] if "event_cnt" in inputs.keys() else None
        frames = inputs["frames"] if "frames" in inputs.keys() else None
        gtflow = inputs["gtflow"] if "gtflow" in inputs.keys() else None
        
        # Get dimensions from events if available, otherwise from flow or gtflow
        if events is not None:
            height = events.shape[2]
            width = events.shape[3]
        elif flow is not None:
            height = flow.shape[2]
            width = flow.shape[3]
        elif gtflow is not None:
            height = gtflow.shape[2]
            width = gtflow.shape[3]
        else:
            height, width = 256, 256  # fallback
        
        # Skip saving if not enough time has passed
        if ts is not None:
            if self.last_store_ts is not None and (ts - self.last_store_ts) < self.store_interval:
                return
            self.last_store_ts = ts

        # check if new sequence
        path_to = self.store_dir + sequence + "/"
        if not os.path.exists(path_to):
            # If we have active video writers from a previous sequence, close them
            if self.video_writers:
                for _w in self.video_writers.values():
                    try:
                        _w.release()
                    except Exception:
                        pass
                self.video_writers.clear()

            os.makedirs(path_to)
            os.makedirs(path_to + "gtflow/")
            os.makedirs(path_to + "flow/")
            os.makedirs(path_to + "masked_flow_grad/")
            os.makedirs(path_to + "masked_flow_vec/")
            os.makedirs(path_to + "stitched/")
            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(path_to + "timestamps.txt", "w")
            self.img_idx = 0
            # reset store timestamp so first frames of the new sequence are stored immediately
            self.last_store_ts = None

        # input events
        event_image = np.zeros((height, width))
        events = events.detach()
        events_npy = events.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
        event_image = self.events_to_image(events_npy)
        filename = path_to + "events/%09d.png" % self.img_idx
        cv2.imwrite(filename, event_image * 255)

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            events_window_npy = self.events_to_image(events_window_npy)
            filename = path_to + "events_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, events_window_npy * 255)

        # Prepare holders for stitched output (four panels)
        flow_frame = None
        masked_grad_frame = None
        masked_vec_frame = None
        gtflow_frame = None

        # input frames
        if frames is not None:
            frames = frames.detach()
            frames_npy = frames.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, 2))
            filename = path_to + "frames/%09d.png" % self.img_idx
            cv2.imwrite(filename, frames_npy[:, :, 1])

        # full estimated flow (gradient-based)
        if flow is not None:
            flow = flow.detach()
            flow_h, flow_w = flow.shape[2], flow.shape[3]
            flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((flow_h, flow_w, 2))
            flow_grad_img = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1])
            flow_grad_img = cv2.cvtColor(flow_grad_img, cv2.COLOR_RGB2BGR)
            if self.store_type == "image":
                filename = path_to + "flow/%09d.png" % self.img_idx
                cv2.imwrite(filename, flow_grad_img)
            elif self.store_type == "video":
                if "flow" not in self.video_writers:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30
                    shape = (flow_w, flow_h)
                    self.video_writers["flow"] = cv2.VideoWriter(path_to + "flow/flow.mp4", fourcc, fps, shape)
                self.video_writers["flow"].write(flow_grad_img)

            # store frame for stitched output
            flow_frame = flow_grad_img

        # masked estimated flow (gradient and vector based)
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_h, masked_w = masked_window_flow.shape[2], masked_window_flow.shape[3]
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((masked_h, masked_w, 2))
            # Gradient-based visualization
            masked_grad_img = self.flow_to_image(
                masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
            )
            masked_grad_img = cv2.cvtColor(masked_grad_img, cv2.COLOR_RGB2BGR)
            if self.store_type == "image":
                filename = path_to + "masked_flow_grad/%09d.png" % self.img_idx
                cv2.imwrite(filename, masked_grad_img)
            elif self.store_type == "video":
                if "masked_flow_grad" not in self.video_writers:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30
                    shape = (masked_w, masked_h)
                    self.video_writers["masked_flow_grad"] = cv2.VideoWriter(path_to + "masked_flow_grad/masked_flow_grad.mp4", fourcc, fps, shape)
                self.video_writers["masked_flow_grad"].write(masked_grad_img)

            # Vector-based visualization
            gt_x = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 0] if gtflow is not None else None
            gt_y = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow.shape[2], gtflow.shape[3], 2))[:, :, 1] if gtflow is not None else None
            masked_vec_img = self.flow_to_vector(
                masked_window_flow_npy[:, :, 0],
                masked_window_flow_npy[:, :, 1],
                type="sparse",
                center=True,
                gt_flow_x=gt_x,
                gt_flow_y=gt_y,
                overlay_gt=True if gtflow is not None else False,
                fixed_length=70,
            )
            if self.store_type == "image":
                filename = path_to + "masked_flow_vec/%09d.png" % self.img_idx
                cv2.imwrite(filename, masked_vec_img)
            elif self.store_type == "video":
                if "masked_flow_vec" not in self.video_writers:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30
                    shape = (masked_w, masked_h)
                    self.video_writers["masked_flow_vec"] = cv2.VideoWriter(path_to + "masked_flow_vec/masked_flow_vec.mp4", fourcc, fps, shape)
                self.video_writers["masked_flow_vec"].write(masked_vec_img)

            # store frames for stitched output
            masked_grad_frame = masked_grad_img
            masked_vec_frame = masked_vec_img

        # ground-truth optical flow (gradient-based)
        if gtflow is not None:
            gtflow = gtflow.detach()
            gtflow_h, gtflow_w = gtflow.shape[2], gtflow.shape[3]
            gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow_h, gtflow_w, 2))
            gtflow_img = self.flow_to_image(gtflow_npy[:, :, 0], gtflow_npy[:, :, 1])
            gtflow_img = cv2.cvtColor(gtflow_img, cv2.COLOR_RGB2BGR)
            if self.store_type == "image":
                filename = path_to + "gtflow/%09d.png" % self.img_idx
                cv2.imwrite(filename, gtflow_img)
            elif self.store_type == "video":
                if "gtflow" not in self.video_writers:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30
                    shape = (gtflow_w, gtflow_h)
                    self.video_writers["gtflow"] = cv2.VideoWriter(path_to + "gtflow/gtflow.mp4", fourcc, fps, shape)
                self.video_writers["gtflow"].write(gtflow_img)

            # store frame for stitched output
            gtflow_frame = gtflow_img

        # Create stitched output (four panels side-by-side). If any panel is missing, replace it with a black placeholder.
        frames_to_stitch = [gtflow_frame, flow_frame, masked_grad_frame, masked_vec_frame]
        if any(f is not None for f in frames_to_stitch):
            # compute target heights/widths
            present = [f for f in frames_to_stitch if f is not None]
            widths = [f.shape[1] for f in present]
            heights = [f.shape[0] for f in present]
            max_h = max(heights)
            default_w = max(widths) if widths else 1

            padded = []
            for f in frames_to_stitch:
                if f is None:
                    pad = np.zeros((max_h, default_w, 3), dtype=np.uint8)
                    padded.append(pad)
                else:
                    h, w = f.shape[0], f.shape[1]
                    if h < max_h:
                        pad = np.zeros((max_h, w, 3), dtype=np.uint8)
                        y = (max_h - h) // 2
                        pad[y : y + h, :w] = f
                        padded.append(pad)
                    else:
                        padded.append(f)

            stitched_img = cv2.hconcat(padded)

            # Some codecs require even frame dimensions. Pad to even width/height if needed.
            sh, sw = stitched_img.shape[0], stitched_img.shape[1]
            pad_bottom = 0 if (sh % 2 == 0) else 1
            pad_right = 0 if (sw % 2 == 0) else 1
            if pad_bottom or pad_right:
                stitched_img = cv2.copyMakeBorder(stitched_img, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                sh, sw = stitched_img.shape[0], stitched_img.shape[1]

            if self.store_type == "image":
                filename = path_to + "stitched/%09d.png" % self.img_idx
                cv2.imwrite(filename, stitched_img)
            elif self.store_type == "video":
                # VideoWriter expects (width, height)
                if "stitched" not in self.video_writers:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30
                    shape = (sw, sh)
                    self.video_writers["stitched"] = cv2.VideoWriter(path_to + "stitched/stitched.mp4", fourcc, fps, shape)
                self.video_writers["stitched"].write(stitched_img)
                
    def close_videos(self):
        """Close all video writers if in video mode."""
        if self.store_type == "video":
            for writer in self.video_writers.values():
                writer.release()

    @staticmethod
    def flow_to_image(flow_x, flow_y):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
        flows = np.stack((flow_x, flow_y), axis=2)
        mag = np.linalg.norm(flows, axis=2)
        min_mag = np.min(mag)
        mag_range = np.max(mag) - min_mag

        ang = np.arctan2(flow_y, flow_x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3])
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0
        hsv[:, :, 2] = mag - min_mag
        if mag_range != 0.0:
            hsv[:, :, 2] /= mag_range

        flow_rgb = matplotlib.colors.hsv_to_rgb(hsv)
        return (255 * flow_rgb).astype(np.uint8)
    
    @staticmethod
    def flow_to_vector(flow_x, flow_y, type="dense", step=12, scale=6.0, min_magnitude=0.2, center=False, gt_flow_x=None, gt_flow_y=None, overlay_gt=True, fixed_length=None):
        """
        Use the optical flow to generate a matrix of vectors representing the direction 
        and magnitude of the optical flows.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :param type: "dense" for ground truth, "sparse" for estimated optical flow
        :param step: Sampling step for vector representation
        :param scale: scaling factor for vector length
        :param min_magnitude: minimum magnitude to draw a vector
        :return img: [H x W x 3] vector-encoded optical flow image
        """
        # center: if True, draw a single arrow at the image center corresponding
        # to the average flow over the whole frame (masked by min_magnitude)
        if type == "sparse":
            step = 6
            scale = 750.0
            min_magnitude = 0.01
            thickness = 3
            tip_length = 0.3
        else:
            thickness = 3
            tip_length = 0.3

        # Support both (H, W) and (H, W, 1) shapes
        if flow_x.ndim == 3:
            H, W = flow_x.shape[0], flow_x.shape[1]
            fx = flow_x[:, :, 0]
            fy = flow_y[:, :, 0]
        else:
            H, W = flow_x.shape
            fx = flow_x
            fy = flow_y

        # Create a black image
        img = np.zeros((H, W, 3), dtype=np.uint8)

        # Precompute magnitude and angular mapping for color coding
        mag = np.sqrt(fx ** 2 + fy ** 2)
        min_mag = float(np.min(mag))
        mag_range = float(np.max(mag) - min_mag)

        if center:
            # compute average flow over pixels above min_magnitude
            mask = mag >= min_magnitude
            if not np.any(mask):
                # nothing significant to draw
                return img
            avg_dx = float(np.mean(fx[mask]))
            avg_dy = float(np.mean(fy[mask]))
            avg_mag = float(np.sqrt(avg_dx ** 2 + avg_dy ** 2))

            # center coordinates (x=j, y=i)
            cx = W // 2
            cy = H // 2

            # compute relative length: scale the arrow length according to avg_mag / max_mag
            max_mag_pred = float(np.max(mag))
            # Optionally include GT in shared scaling reference
            max_mag_gt = 0.0
            if overlay_gt and gt_flow_x is not None and gt_flow_y is not None:
                if gt_flow_x.ndim == 3:
                    gx = gt_flow_x[:, :, 0]
                    gy = gt_flow_y[:, :, 0]
                else:
                    gx = gt_flow_x
                    gy = gt_flow_y
                mag_gt_full = np.sqrt(gx ** 2 + gy ** 2)
                max_mag_gt = float(np.max(mag_gt_full))
            combined_max = max(max_mag_pred, max_mag_gt)

            if combined_max <= 0 or avg_mag == 0:
                return img

            # relative fraction (0..1)
            frac = np.clip(avg_mag / combined_max, 0.0, 1.0)

            # maximum drawable length (pixels) -- keep it within frame
            max_len = int(0.45 * min(H, W))

            # If a fixed length is requested, use it; otherwise use fraction of max_len
            if fixed_length is not None:
                length_px = int(fixed_length)
            else:
                # arrow length in pixels proportional to fraction and scaled by max_len
                length_px = int(frac * max_len)

            # normalize direction and compute endpoint (invert arrow direction)
            inv_avg_dx = -avg_dx
            inv_avg_dy = -avg_dy
            dir_x = inv_avg_dx / avg_mag
            dir_y = inv_avg_dy / avg_mag
            end_x = int(cx + dir_x * length_px)
            end_y = int(cy + dir_y * length_px)

            # If requested, draw GT average arrow behind predicted one (white)
            if overlay_gt and gt_flow_x is not None and gt_flow_y is not None:
                # compute GT average with same thresholding
                if gt_flow_x.ndim == 3:
                    gx = gt_flow_x[:, :, 0]
                    gy = gt_flow_y[:, :, 0]
                else:
                    gx = gt_flow_x
                    gy = gt_flow_y
                mag_gt = np.sqrt(gx ** 2 + gy ** 2)
                # Prefer to mask GT using the predicted mask if shapes align, so
                # we compute the GT average only over pixels where the predicted
                # flow has support. Otherwise fall back to GT magnitude threshold.
                if mag.shape == mag_gt.shape:
                    mask_shared = mask & (mag_gt >= min_magnitude)
                else:
                    mask_shared = mag_gt >= min_magnitude
                if np.any(mask_shared):
                    avg_gx = float(np.mean(gx[mask_shared]))
                    avg_gy = float(np.mean(gy[mask_shared]))
                    avg_mag_gt = float(np.sqrt(avg_gx ** 2 + avg_gy ** 2))
                    if avg_mag_gt > 0:
                        # scale using the same combined_max
                        frac_gt = np.clip(avg_mag_gt / combined_max, 0.0, 1.0)
                        if fixed_length is not None:
                            length_px_gt = int(fixed_length)
                        else:
                            length_px_gt = int(frac_gt * max_len)
                        # invert GT arrow direction as well
                        inv_avg_gx = -avg_gx
                        inv_avg_gy = -avg_gy
                        dir_x_gt = inv_avg_gx / avg_mag_gt
                        dir_y_gt = inv_avg_gy / avg_mag_gt
                        end_x_gt = int(cx + dir_x_gt * length_px_gt)
                        end_y_gt = int(cy + dir_y_gt * length_px_gt)
                        # draw GT arrow first (behind): white color
                        cv2.arrowedLine(img, (cx, cy), (end_x_gt, end_y_gt), (255, 255, 255), thickness, tipLength=tip_length)

            # compute color via same HSV mapping as flow_to_image
            # Use the ORIGINAL (non-inverted) flow direction for color so
            # the color wheel represents the true flow direction even though
            # the arrow geometry is inverted for visualization.
            ang = np.arctan2(avg_dy, avg_dx) + np.pi
            ang *= 1.0 / np.pi / 2.0
            hsv = np.array([ang, 1.0, (avg_mag - min_mag)])
            if mag_range != 0.0:
                hsv[2] = hsv[2] / mag_range
            rgb = matplotlib.colors.hsv_to_rgb(hsv)
            # convert to BGR 0-255 ints for OpenCV
            arrow_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

            cv2.arrowedLine(img, (cx, cy), (end_x, end_y), arrow_color, thickness, tipLength=tip_length)
            return img

        # Sample the flow fields
        for i in range(0, H, step):
            for j in range(0, W, step):
                dx = fx[i, j]
                dy = fy[i, j]
                mag = np.sqrt(dx**2 + dy**2)

                if mag < min_magnitude:
                    continue  # skip drawing short vectors

                # invert direction
                inv_dx = -dx
                inv_dy = -dy
                if fixed_length is not None:
                    # draw sampled arrow with fixed pixel length in direction of vector
                    ux = inv_dx / mag
                    uy = inv_dy / mag
                    end_x = int(j + ux * float(fixed_length))
                    end_y = int(i + uy * float(fixed_length))
                else:
                    end_x = int(j + inv_dx * scale)
                    end_y = int(i + inv_dy * scale)

                # compute color from angle/magnitude using the ORIGINAL direction
                # (do not invert) so colors correspond to the true flow direction
                ang = np.arctan2(dy, dx) + np.pi
                ang *= 1.0 / np.pi / 2.0
                v = mag - min_mag
                if mag_range != 0.0:
                    v = v / mag_range
                else:
                    v = 0.0
                hsv = np.array([ang, 1.0, v])
                rgb = matplotlib.colors.hsv_to_rgb(hsv)
                arrow_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

                # Draw arrow
                cv2.arrowedLine(img, (j, i), (end_x, end_y), arrow_color, thickness, tipLength=tip_length)

        return img

    @staticmethod
    def minmax_norm(x):
        """
        Robust min-max normalization.
        :param x: [H x W x 1]
        :return x: [H x W x 1] normalized x
        """
        den = np.percentile(x, 99) - np.percentile(x, 1)
        if den != 0:
            x = (x - np.percentile(x, 1)) / den
        return np.clip(x, 0, 1)

    @staticmethod
    def events_to_image(event_cnt, color_scheme="green_red"):
        """
        Visualize the input events.
        :param event_cnt: [batch_size x 2 x H x W] per-pixel and per-polarity event count
        :param color_scheme: green_red/gray
        :return event_image: [H x W x 3] color-coded event image
        """
        pos = event_cnt[:, :, 0]
        neg = event_cnt[:, :, 1]
        pos_max = np.percentile(pos, 99)
        pos_min = np.percentile(pos, 1)
        neg_max = np.percentile(neg, 99)
        neg_min = np.percentile(neg, 1)
        max = pos_max if pos_max > neg_max else neg_max

        if pos_min != max:
            pos = (pos - pos_min) / (max - pos_min)
        if neg_min != max:
            neg = (neg - neg_min) / (max - neg_min)

        pos = np.clip(pos, 0, 1)
        neg = np.clip(neg, 0, 1)

        event_image = np.ones((event_cnt.shape[0], event_cnt.shape[1]))
        if color_scheme == "gray":
            event_image *= 0.5
            pos *= 0.5
            neg *= -0.5
            event_image += pos + neg

        elif color_scheme == "green_red":
            event_image = np.repeat(event_image[:, :, np.newaxis], 3, axis=2)
            event_image *= 0
            mask_pos = pos > 0
            mask_neg = neg > 0
            mask_not_pos = pos == 0
            mask_not_neg = neg == 0

            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            event_image[:, :, 2][mask_neg] = neg[mask_neg]
            event_image[:, :, 0][mask_neg] = 0
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0

        return event_image


def vis_activity(activity, activity_log):
    # start of new sequence
    if activity_log is None:
        plt.close("activity")
        activity_log = []

    # update log
    activity_log.append(activity)
    df = pd.DataFrame(activity_log)

    # retrieves fig if it exists
    fig = plt.figure("activity")
    # make axis if it doesn't exist
    if not fig.axes:
        ax = fig.add_subplot()
    else:
        ax = fig.axes[0]
    lines = ax.lines

    # plot data
    if not lines:
        for name, data in df.iteritems():
            ax.plot(data.index.to_numpy(), data.to_numpy(), label=name)
        ax.grid()
        ax.legend()
        ax.set_xlabel("step")
        ax.set_ylabel("fraction of nonzero outputs")
        plt.show(block=False)
    else:
        for line in lines:
            label = line.get_label()
            line.set_data(df[label].index.to_numpy(), df[label].to_numpy())

    # update figure
    fig.canvas.draw()
    ax.relim()
    ax.autoscale_view(True, True, True)
    fig.canvas.flush_events()

    return activity_log