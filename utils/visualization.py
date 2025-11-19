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
        # visualization brightness scale for flow images (1.0 = unchanged)
        self.v_scale = float(kwargs["vis"].get("v_scale", 1))
        # value (V) to use for uniform non-zero flow fields (0..1). This
        # prevents uniform fields from being mapped to full brightness.
        self.v_uniform = float(kwargs["vis"].get("v_uniform", 1))
        self.last_store_ts = None  # for controlling store rate
        self.store_interval = kwargs["vis"].get("store_interval", 5.0)  # seconds
        self.vis_type = vis_type
        self.store_type = kwargs["vis"].get("store_type", "image")  # 'image' or 'video'
        self.video_writers = {}
        # flow scaling factor to convert from model's internal representation to pixels
        self.flow_scaling = float(kwargs.get("metrics", {}).get("flow_scaling", 1.0))

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
                # Scale flow from model's internal representation to pixels
                masked_window_flow_npy = masked_window_flow_npy * self.flow_scaling
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
            # Scale flow from model's internal representation to pixels
            flow_npy = flow_npy * self.flow_scaling
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
                flow_npy = self._apply_v_scale(flow_npy)
                flow_npy = cv2.cvtColor(flow_npy, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Estimated Flow", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Estimated Flow", int(self.px), int(self.px))
            cv2.imshow("Estimated Flow", flow_npy)

        # optical flow (masked)
        if masked_window_flow is not None:
            masked_window_flow = masked_window_flow.detach()
            masked_h, masked_w = masked_window_flow.shape[2], masked_window_flow.shape[3]
            masked_window_flow_npy = masked_window_flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((masked_h, masked_w, 2))
            # Scale flow from model's internal representation to pixels
            masked_window_flow_npy = masked_window_flow_npy * self.flow_scaling
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
                    masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1], uniform_v=self.v_uniform
                )
                masked_window_flow_npy = self._apply_v_scale(masked_window_flow_npy)
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

        # Special handling for 8x8_vec store type: create single video with dual-arrow visualization
        if self.store_type == "8x8_vec":
            path_to = self.store_dir + sequence + "/"
            if not os.path.exists(path_to):
                # Close any active video writers from previous sequence
                if self.video_writers:
                    for _w in self.video_writers.values():
                        try:
                            _w.release()
                        except Exception:
                            pass
                    self.video_writers.clear()
                
                os.makedirs(path_to)
                if self.store_file is not None:
                    self.store_file.close()
                self.store_file = open(path_to + "timestamps.txt", "w")
                self.img_idx = 0
                self.last_store_ts = None
            
            # Extract flow and gtflow numpy arrays
            if flow is not None and gtflow is not None:
                flow = flow.detach()
                flow_h, flow_w = flow.shape[2], flow.shape[3]
                flow_npy = flow.cpu().numpy().transpose(0, 2, 3, 1).reshape((flow_h, flow_w, 2))
                # Scale flow from model's internal representation to pixels
                flow_npy = flow_npy * self.flow_scaling
                
                gtflow = gtflow.detach()
                gtflow_h, gtflow_w = gtflow.shape[2], gtflow.shape[3]
                gtflow_npy = gtflow.cpu().numpy().transpose(0, 2, 3, 1).reshape((gtflow_h, gtflow_w, 2))
                
                # Create 256x256 dual-arrow frame
                dual_arrow_frame = self._draw_dual_arrow_frame(
                    flow_npy[:, :, 0], flow_npy[:, :, 1],
                    gtflow_npy[:, :, 0], gtflow_npy[:, :, 1],
                    frame_size=256,
                    fixed_length=80
                )
                
                # Write to video
                if "8x8_vec" not in self.video_writers:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = 30
                    shape = (256, 256)
                    self.video_writers["8x8_vec"] = cv2.VideoWriter(path_to + "flow_vectors.mp4", fourcc, fps, shape)
                self.video_writers["8x8_vec"].write(dual_arrow_frame)
            
            self.img_idx += 1
            return

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
            os.makedirs(path_to + "events/")
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
        
        # Store individual event images or video
        events_img_bgr = (event_image * 255).astype(np.uint8)
        if events_img_bgr.ndim == 2:
            events_img_bgr = cv2.cvtColor(events_img_bgr, cv2.COLOR_GRAY2BGR)
        elif events_img_bgr.shape[2] == 3:
            events_img_bgr = cv2.cvtColor(events_img_bgr, cv2.COLOR_RGB2BGR)
        
        if self.store_type == "image":
            filename = path_to + "events/%09d.png" % self.img_idx
            cv2.imwrite(filename, events_img_bgr)
        elif self.store_type == "video":
            if "events" not in self.video_writers:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps = 30
                shape = (width, height)
                self.video_writers["events"] = cv2.VideoWriter(path_to + "events/events.mp4", fourcc, fps, shape)
            self.video_writers["events"].write(events_img_bgr)

        # input events
        if events_window is not None:
            events_window = events_window.detach()
            events_window_npy = events_window.cpu().numpy().transpose(0, 2, 3, 1).reshape((height, width, -1))
            events_window_npy = self.events_to_image(events_window_npy)
            filename = path_to + "events_window/%09d.png" % self.img_idx
            cv2.imwrite(filename, events_window_npy * 255)

        # Prepare holders for stitched output (four panels)
        # Convert event_image to BGR uint8 for stitching
        events_frame = (event_image * 255).astype(np.uint8)
        if events_frame.ndim == 2:
            events_frame = cv2.cvtColor(events_frame, cv2.COLOR_GRAY2BGR)
        elif events_frame.shape[2] == 3:
            events_frame = cv2.cvtColor(events_frame, cv2.COLOR_RGB2BGR)
        
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
            # Scale flow from model's internal representation to pixels
            flow_npy = flow_npy * self.flow_scaling
            flow_grad_img = self.flow_to_image(flow_npy[:, :, 0], flow_npy[:, :, 1], uniform_v=self.v_uniform)
            flow_grad_img = self._apply_v_scale(flow_grad_img)
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
            # Scale flow from model's internal representation to pixels
            masked_window_flow_npy = masked_window_flow_npy * self.flow_scaling
            # Gradient-based visualization
            masked_grad_img = self.flow_to_image(
                masked_window_flow_npy[:, :, 0], masked_window_flow_npy[:, :, 1]
            )
            masked_grad_img = self._apply_v_scale(masked_grad_img)
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
        # Order: input events → ground truth → masked flow (gradient) → masked flow (vectors with dual arrows)
        frames_to_stitch = [events_frame, gtflow_frame, masked_grad_frame, masked_vec_frame]
        frame_labels = ["Input events", "Ground truth", "Masked flow", "gt (white) vs. flow (color)"]
        
        if any(f is not None for f in frames_to_stitch):
            # compute target heights/widths
            present = [f for f in frames_to_stitch if f is not None]
            widths = [f.shape[1] for f in present]
            heights = [f.shape[0] for f in present]
            max_h = max(heights)
            default_w = max(widths) if widths else 1

            padded = []
            for idx, f in enumerate(frames_to_stitch):
                if f is None:
                    pad = np.zeros((max_h, default_w, 3), dtype=np.uint8)
                else:
                    h, w = f.shape[0], f.shape[1]
                    if h < max_h:
                        pad = np.zeros((max_h, w, 3), dtype=np.uint8)
                        y = (max_h - h) // 2
                        pad[y : y + h, :w] = f
                    else:
                        pad = f.copy()
                
                # Add text label at the top of each panel
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                text_color = (255, 255, 255)  # white text
                text = frame_labels[idx]
                
                # Get text size for background rectangle
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = 5
                text_y = 20
                
                # Draw black background rectangle for better readability
                cv2.rectangle(pad, (text_x - 2, text_y - text_size[1] - 2),
                            (text_x + text_size[0] + 2, text_y + 2),
                            (0, 0, 0), -1)
                
                # Draw text
                cv2.putText(pad, text, (text_x, text_y), font, font_scale, text_color, thickness)
                
                padded.append(pad)

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

    def _apply_v_scale(self, img):
        """Apply a simple brightness scale to an RGB uint8 image.

        img: HxWx3 uint8 RGB image coming from flow_to_image (0-255)
        Returns scaled uint8 image clipped to [0,255].
        """
        if not hasattr(self, "v_scale") or self.v_scale == 1.0:
            return img
        try:
            f = float(self.v_scale)
        except Exception:
            return img
        if f <= 0:
            return np.zeros_like(img, dtype=np.uint8)
        scaled = (img.astype(np.float32) * f).clip(0, 255).astype(np.uint8)
        return scaled

    @staticmethod
    def _draw_dual_arrow_frame(flow_x, flow_y, gt_flow_x, gt_flow_y, frame_size=256, fixed_length=80):
        """
        Create a single frame showing two centered arrows: GT (white) and predicted (colored).
        Arrow lengths are proportional to flow magnitudes with power-law scaling for better visibility.
        Designed for global/small flow outputs (e.g., [B,2,1,1] expanded to [B,2,H,W]).
        
        :param flow_x: [H x W] predicted horizontal optical flow component
        :param flow_y: [H x W] predicted vertical optical flow component
        :param gt_flow_x: [H x W] ground-truth horizontal optical flow component
        :param gt_flow_y: [H x W] ground-truth vertical optical flow component
        :param frame_size: output frame size (e.g., 256x256)
        :param fixed_length: base arrow length in pixels (scales with magnitude)
        :return frame_bgr: [frame_size x frame_size x 3] BGR uint8 frame
        """
        # Create black canvas
        img = np.zeros((frame_size, frame_size, 3), dtype=np.uint8)
        
        # Compute average predicted flow (across all pixels, or use the constant value)
        avg_pred_dx = float(np.mean(flow_x))
        avg_pred_dy = float(np.mean(flow_y))
        avg_pred_mag = float(np.sqrt(avg_pred_dx ** 2 + avg_pred_dy ** 2))
        
        # Compute average GT flow
        avg_gt_dx = float(np.mean(gt_flow_x))
        avg_gt_dy = float(np.mean(gt_flow_y))
        avg_gt_mag = float(np.sqrt(avg_gt_dx ** 2 + avg_gt_dy ** 2))
        
        # Combined max magnitude for consistent scaling across both arrows
        combined_max = max(avg_pred_mag, avg_gt_mag, 1e-6)
        
        cx = frame_size // 2
        cy = frame_size // 2
        
        # Maximum drawable length (pixels) -- keep it within frame
        max_len = int(0.45 * frame_size)
        
        # Power-law scaling (exponent 0.6) to compress range while preserving differences
        # This makes arrows closer but still distinguishable
        power = 0.6
        
        # Draw predicted flow arrow first (white, behind GT)
        if avg_pred_mag > 0:
            # Scale arrow length using power-law to compress large magnitude differences
            normalized_pred = avg_pred_mag / combined_max
            scaled_pred = np.power(normalized_pred, power)
            length_px_pred = int(scaled_pred * max_len)
            
            # Invert direction for visualization
            inv_avg_pred_dx = -avg_pred_dx
            inv_avg_pred_dy = -avg_pred_dy
            dir_x_pred = inv_avg_pred_dx / avg_pred_mag
            dir_y_pred = inv_avg_pred_dy / avg_pred_mag
            end_x_pred = int(cx + dir_x_pred * length_px_pred)
            end_y_pred = int(cy + dir_y_pred * length_px_pred)
            cv2.arrowedLine(img, (cx, cy), (end_x_pred, end_y_pred), (255, 255, 255), 3, tipLength=0.3)
        
        # Draw GT arrow (colored, on top)
        if avg_gt_mag > 0:
            # Scale arrow length using power-law to compress large magnitude differences
            normalized_gt = avg_gt_mag / combined_max
            scaled_gt = np.power(normalized_gt, power)
            length_px_gt = int(scaled_gt * max_len)
            
            # Invert direction for visualization
            inv_avg_gt_dx = -avg_gt_dx
            inv_avg_gt_dy = -avg_gt_dy
            dir_x_gt = inv_avg_gt_dx / avg_gt_mag
            dir_y_gt = inv_avg_gt_dy / avg_gt_mag
            end_x_gt = int(cx + dir_x_gt * length_px_gt)
            end_y_gt = int(cy + dir_y_gt * length_px_gt)
            
            # Compute color from angle/magnitude using ORIGINAL (non-inverted) direction
            ang = np.arctan2(avg_gt_dy, avg_gt_dx) + np.pi
            ang *= 1.0 / np.pi / 2.0
            v = np.clip(normalized_gt, 0.0, 1.0)
            hsv = np.array([ang, 1.0, v])
            rgb = matplotlib.colors.hsv_to_rgb(hsv)
            arrow_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            
            cv2.arrowedLine(img, (cx, cy), (end_x_gt, end_y_gt), arrow_color, 3, tipLength=0.3)
        
        # Add text annotations with flow values
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        text_color = (255, 255, 255)  # white text
        
        # GT flow text on top
        gt_text = f"GT: ({avg_gt_dx:.2f}, {avg_gt_dy:.2f})"
        text_size = cv2.getTextSize(gt_text, font, font_scale, thickness)[0]
        text_pos_gt = (10, 30)
        cv2.rectangle(img, (text_pos_gt[0] - 5, text_pos_gt[1] - text_size[1] - 5),
                      (text_pos_gt[0] + text_size[0] + 5, text_pos_gt[1] + 5),
                      (0, 0, 0), -1)  # black background
        cv2.putText(img, gt_text, text_pos_gt, font, font_scale, text_color, thickness)
        
        # Predicted flow text on bottom
        pred_text = f"Flow: ({avg_pred_dx:.2f}, {avg_pred_dy:.2f})"
        text_size = cv2.getTextSize(pred_text, font, font_scale, thickness)[0]
        text_pos_flow = (10, frame_size - 15)
        cv2.rectangle(img, (text_pos_flow[0] - 5, text_pos_flow[1] - text_size[1] - 5),
                      (text_pos_flow[0] + text_size[0] + 5, text_pos_flow[1] + 5),
                      (0, 0, 0), -1)  # black background
        cv2.putText(img, pred_text, text_pos_flow, font, font_scale, text_color, thickness)
        
        return img

    @staticmethod
    def flow_to_image(flow_x, flow_y, uniform_v=None):
        """
        Use the optical flow color scheme from the supplementary materials of the paper 'Back to Event
        Basics: Self-Supervised Image Reconstruction for Event Cameras via Photometric Constancy',
        Paredes-Valles et al., CVPR'21.
        :param flow_x: [H x W x 1] horizontal optical flow component
        :param flow_y: [H x W x 1] vertical optical flow component
        :return flow_rgb: [H x W x 3] color-encoded optical flow
        """
        flows = np.stack((flow_x, flow_y), axis=2)
        mag = np.linalg.norm(flows, axis=2).astype(float)
        min_mag = float(np.min(mag))
        max_mag = float(np.max(mag))
        mag_range = max_mag - min_mag

        ang = np.arctan2(flow_y, flow_x) + np.pi
        ang *= 1.0 / np.pi / 2.0

        hsv = np.zeros([flow_x.shape[0], flow_x.shape[1], 3], dtype=float)
        hsv[:, :, 0] = ang
        hsv[:, :, 1] = 1.0

        # Value channel (brightness) - robust to uniform fields
        if mag_range > 0.0:
            # Normal case: spread values linearly between min and max
            hsv[:, :, 2] = (mag - min_mag) / mag_range
        else:
            # Uniform magnitude across the field. If magnitude is non-zero,
            # show it with a scaled brightness so it isn't full-white by
            # default (use uniform_v to control). If zero everywhere, leave as black.
            if max_mag > 0.0:
                v = mag / max_mag
                if uniform_v is not None:
                    v = v * float(uniform_v)
                hsv[:, :, 2] = v
            else:
                hsv[:, :, 2] = 0.0

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
        max_mag = float(np.max(mag))
        mag_range = float(max_mag - min_mag)

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
            # Robustly compute the value (brightness) channel. If the field is
            # uniform (mag_range == 0) but non-zero, show relative brightness
            # as avg_mag / max_mag so arrows are visible instead of black.
            if mag_range != 0.0:
                v = (avg_mag - min_mag) / mag_range
            else:
                v = (avg_mag / max_mag) if max_mag > 0.0 else 0.0
            hsv = np.array([ang, 1.0, v])
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
                # Robust brightness normalization similar to flow_to_image.
                if mag_range != 0.0:
                    v = (mag - min_mag) / mag_range
                else:
                    v = (mag / max_mag) if max_mag > 0.0 else 0.0
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

            # Positive events in green (channel 1 in RGB)
            event_image[:, :, 0][mask_pos] = 0
            event_image[:, :, 1][mask_pos] = pos[mask_pos]
            event_image[:, :, 2][mask_pos * mask_not_neg] = 0
            # Negative events in red (channel 0 in RGB)
            event_image[:, :, 0][mask_neg] = neg[mask_neg]
            event_image[:, :, 1][mask_neg * mask_not_pos] = 0
            event_image[:, :, 2][mask_neg] = 0

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