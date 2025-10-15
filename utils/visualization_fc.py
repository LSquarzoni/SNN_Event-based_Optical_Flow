import os
import cv2
import numpy as np
import matplotlib

class FCVisualization:
    """
    Minimal visualization and storage for Fully Connected models at fixed 8x8 resolution.
    - Live/video only (no images)
    - Ground truth: gradient color wheel
    - Predicted flow: single center arrow
    - Masked flow (eval window): single center arrow
    """

    def __init__(self, kwargs, eval_id=-1, path_results=None):
        self.img_idx = 0
        self.px = kwargs["vis"].get("px", 256)
        self.store_interval = kwargs["vis"].get("store_interval", 0.0)
        self.last_store_ts = None
        self.store_type = "video"  # Only video output
        self.video_writers = {}
        self.fixed_h = 8
        self.fixed_w = 8
        # Center-arrow parameters (configurable)
        self.fc_center_fixed_len = int(kwargs["vis"].get("fc_center_fixed_len", 3))
        self.fc_center_thickness = int(kwargs["vis"].get("fc_center_thickness", 2))
        self.fc_center_tip_length = float(kwargs["vis"].get("fc_center_tip_length", 0.3))
        self.fc_center_min_mag = float(kwargs["vis"].get("fc_center_min_mag", 0.0))
        self.fc_center_scale_by_mag = bool(kwargs["vis"].get("fc_center_scale_by_mag", False))

        if eval_id >= 0 and path_results is not None:
            self.store_dir = os.path.join(path_results, "results", f"eval_{eval_id}")
            os.makedirs(self.store_dir, exist_ok=True)
            self.store_file = None
        else:
            self.store_dir = None
            self.store_file = None

    def update(self, inputs, flow, iwe, events_window=None, masked_window_flow=None, iwe_window=None):
        """Live visualization at 8x8: center arrow for flows, gradient for GT."""
        gtflow = inputs["gtflow"] if "gtflow" in inputs else None
        # Full flow (center arrow)
        if flow is not None:
            fh, fw = flow.shape[2], flow.shape[3]
            fn = flow.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((fh, fw, 2))
            vec = self.flow_to_vector_center(
                fn[:, :, 0], fn[:, :, 1],
                fixed_len=self.fc_center_fixed_len,
                thickness=self.fc_center_thickness,
                tip_length=self.fc_center_tip_length,
                min_magnitude=self.fc_center_min_mag,
                scale_by_mag=self.fc_center_scale_by_mag,
            )
            cv2.namedWindow("Flow (vectors)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Flow (vectors)", int(self.px), int(self.px))
            cv2.imshow("Flow (vectors)", vec)
        # Masked flow (center arrow)
        if masked_window_flow is not None:
            mh, mw = masked_window_flow.shape[2], masked_window_flow.shape[3]
            mn = masked_window_flow.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((mh, mw, 2))
            mvec = self.flow_to_vector_center(
                mn[:, :, 0], mn[:, :, 1],
                fixed_len=self.fc_center_fixed_len,
                thickness=self.fc_center_thickness,
                tip_length=self.fc_center_tip_length,
                min_magnitude=self.fc_center_min_mag,
                scale_by_mag=self.fc_center_scale_by_mag,
            )
            cv2.namedWindow("Masked Flow (vectors)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Masked Flow (vectors)", int(self.px), int(self.px))
            cv2.imshow("Masked Flow (vectors)", mvec)
        # GT gradient
        if gtflow is not None:
            gh, gw = gtflow.shape[2], gtflow.shape[3]
            gn = gtflow.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((gh, gw, 2))
            gimg = self.flow_to_image(gn[:, :, 0], gn[:, :, 1])
            gimg = cv2.cvtColor(gimg, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("GT (gradient)", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("GT (gradient)", int(self.px), int(self.px))
            cv2.imshow("GT (gradient)", gimg)
        cv2.waitKey(1)

    def store(self, inputs, flow, iwe, sequence, events_window=None, masked_window_flow=None, iwe_window=None, ts=None):
        """Store video with panels: [GT gradient | Flow center arrow | Masked center arrow]."""
        if self.store_dir is None:
            return
        if ts is not None:
            if self.last_store_ts is not None and (ts - self.last_store_ts) < self.store_interval:
                return
            self.last_store_ts = ts

        # New sequence setup
        path_to = os.path.join(self.store_dir, sequence)
        if not os.path.exists(path_to):
            self._close_all_writers()
            os.makedirs(path_to, exist_ok=True)
            os.makedirs(os.path.join(path_to, "gtflow"), exist_ok=True)
            os.makedirs(os.path.join(path_to, "flow_vec"), exist_ok=True)
            os.makedirs(os.path.join(path_to, "masked_flow_vec"), exist_ok=True)
            os.makedirs(os.path.join(path_to, "stitched"), exist_ok=True)
            if self.store_file is not None:
                self.store_file.close()
            self.store_file = open(os.path.join(path_to, "timestamps.txt"), "w")
            self.img_idx = 0
            self.last_store_ts = None

        gtflow = inputs["gtflow"] if "gtflow" in inputs else None

        # Prepare panels
        gt_panel = None
        flow_panel = None
        masked_panel = None

        if gtflow is not None:
            gh, gw = gtflow.shape[2], gtflow.shape[3]
            gn = gtflow.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((gh, gw, 2))
            gt_panel = self.flow_to_image(gn[:, :, 0], gn[:, :, 1])
            gt_panel = cv2.cvtColor(gt_panel, cv2.COLOR_RGB2BGR)
            self._write_video(os.path.join(path_to, "gtflow", "gtflow.mp4"), "gtflow", gt_panel)

        if flow is not None:
            fh, fw = flow.shape[2], flow.shape[3]
            fn = flow.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((fh, fw, 2))
            flow_panel = self.flow_to_vector_center(
                fn[:, :, 0], fn[:, :, 1],
                fixed_len=self.fc_center_fixed_len,
                thickness=self.fc_center_thickness,
                tip_length=self.fc_center_tip_length,
                min_magnitude=self.fc_center_min_mag,
                scale_by_mag=self.fc_center_scale_by_mag,
            )
            self._write_video(os.path.join(path_to, "flow_vec", "flow_vec.mp4"), "flow_vec", flow_panel)

        if masked_window_flow is not None:
            mh, mw = masked_window_flow.shape[2], masked_window_flow.shape[3]
            mn = masked_window_flow.detach().cpu().numpy().transpose(0, 2, 3, 1).reshape((mh, mw, 2))
            masked_panel = self.flow_to_vector_center(
                mn[:, :, 0], mn[:, :, 1],
                fixed_len=self.fc_center_fixed_len,
                thickness=self.fc_center_thickness,
                tip_length=self.fc_center_tip_length,
                min_magnitude=self.fc_center_min_mag,
                scale_by_mag=self.fc_center_scale_by_mag,
            )
            self._write_video(os.path.join(path_to, "masked_flow_vec", "masked_flow_vec.mp4"), "masked_flow_vec", masked_panel)

        # Stitch: [GT | Flow center | Masked center]
        panels = [gt_panel, flow_panel, masked_panel]
        stitched = self._stitch(panels)
        self._write_video(os.path.join(path_to, "stitched", "stitched.mp4"), "stitched", stitched)

        self.img_idx += 1

    def close_videos(self):
        self._close_all_writers()

    # Helpers
    def _close_all_writers(self):
        for w in self.video_writers.values():
            try:
                w.release()
            except Exception:
                pass
        self.video_writers.clear()

    def _write_video(self, out_path, key, frame_bgr, fps=30):
        h, w = frame_bgr.shape[:2]
        # pad even sizes for codec
        pad_bottom = 1 if (h % 2) else 0
        pad_right = 1 if (w % 2) else 0
        if pad_bottom or pad_right:
            frame_bgr = cv2.copyMakeBorder(frame_bgr, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            h, w = frame_bgr.shape[:2]
        if key not in self.video_writers:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writers[key] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        self.video_writers[key].write(frame_bgr)

    def _stitch(self, frames):
        present = [f for f in frames if f is not None]
        if not present:
            return np.zeros((self.fixed_h, self.fixed_w, 3), dtype=np.uint8)
        widths = [f.shape[1] for f in present]
        heights = [f.shape[0] for f in present]
        max_h = max(heights)
        default_w = max(widths)
        padded = []
        for f in frames:
            if f is None:
                padded.append(np.zeros((max_h, default_w, 3), dtype=np.uint8))
            else:
                h, w = f.shape[:2]
                if h < max_h:
                    pad = np.zeros((max_h, w, 3), dtype=np.uint8)
                    y = (max_h - h) // 2
                    pad[y:y + h, :w] = f
                    padded.append(pad)
                else:
                    padded.append(f)
        return cv2.hconcat(padded)

    @staticmethod
    def flow_to_image(flow_x, flow_y):
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
    def flow_to_vector_center(flow_x, flow_y, fixed_len=3, thickness=2, tip_length=0.3, min_magnitude=0.0, scale_by_mag=False):
        """Draw a single arrow at center using average flow direction; color encodes direction via HSV."""
        H, W = flow_x.shape if flow_x.ndim == 2 else (flow_x.shape[0], flow_x.shape[1])
        fx = flow_x[:, :] if flow_x.ndim == 2 else flow_x[:, :, 0]
        fy = flow_y[:, :] if flow_y.ndim == 2 else flow_y[:, :, 0]
        img = np.zeros((H, W, 3), dtype=np.uint8)

        mag = np.sqrt(fx ** 2 + fy ** 2)
        mask = mag >= min_magnitude
        if not np.any(mask):
            return img
        avg_dx = float(np.mean(fx[mask]))
        avg_dy = float(np.mean(fy[mask]))
        avg_mag = float(np.sqrt(avg_dx ** 2 + avg_dy ** 2))
        if avg_mag == 0.0:
            return img

        cx, cy = W // 2, H // 2

        if scale_by_mag:
            max_mag = float(np.max(mag))
            if max_mag > 0:
                frac = np.clip(avg_mag / max_mag, 0.0, 1.0)
                max_len = int(0.45 * min(H, W))
                length_px = max(1, int(frac * max_len))
            else:
                length_px = fixed_len
        else:
            length_px = fixed_len

        inv_dx, inv_dy = -avg_dx, -avg_dy
        dir_x, dir_y = inv_dx / avg_mag, inv_dy / avg_mag
        end_x = int(cx + dir_x * length_px)
        end_y = int(cy + dir_y * length_px)

        ang = np.arctan2(avg_dy, avg_dx) + np.pi
        ang *= 1.0 / np.pi / 2.0
        min_mag_all = float(np.min(mag))
        mag_range = float(np.max(mag) - min_mag_all)
        v = (avg_mag - min_mag_all)
        v = (v / mag_range) if mag_range != 0.0 else 0.0
        hsv = np.array([ang, 1.0, v])
        rgb = matplotlib.colors.hsv_to_rgb(hsv)
        color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

        cv2.arrowedLine(img, (cx, cy), (end_x, end_y), color, thickness, tipLength=tip_length)
        return img
    
