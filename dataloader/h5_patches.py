import os
import h5py
import hdf5plugin  # noqa: F401 (ensures plugins are registered when reading compressed datasets)
import numpy as np

import torch

from .base import BaseDataLoader
from .encodings import binary_search_array


class PatchH5Loader(BaseDataLoader):
    """
    Events-only dataloader that extracts fixed-size spatial patches and
    feeds them following their temporal flow. For a given patch position
    (e.g., (0,0)), it will yield consecutive temporal windows from the
    start of the sequence to the end before moving to the next patch.

    Notes
    - Training only (events mode); frames/gtflow not supported here.
    - Uses non-overlapping patches by default (stride == patch size).
    - If full resolution is not perfectly divisible by patch size, the
      border remainder is ignored.
    """

    def __init__(self, config, num_bins, round_encoding=False):
        super().__init__(config, num_bins, round_encoding)

        if self.config["data"]["mode"] != "events":
            raise ValueError("PatchH5Loader supports only data.mode == 'events'")

        # Patch settings
        self.patch_size = tuple(self.config["loader"].get("patch_size", [64, 64]))  # (H, W)
        self.patch_stride = tuple(self.config["loader"].get("patch_stride", list(self.patch_size)))

        # Full sensor resolution (H, W) as defined in config
        self.full_resolution = tuple(self.config["loader"].get("std_resolution", self.config["loader"]["resolution"]))

        # Force the encoding resolution to be the patch size (overrides BaseDataLoader setting)
        self.resolution = self.patch_size

        # Reinitialize hot pixel buffers if enabled (shape changed)
        if self.config["hot_filter"]["enabled"]:
            self.hot_idx = [0 for _ in range(self.config["loader"]["batch_size"])]
            self.hot_events = [torch.zeros(self.resolution) for _ in range(self.config["loader"]["batch_size"])]

        # Build list of files
        self.files = []
        for root, _, files in os.walk(self.config["data"]["path"]):
            for file in files:
                if file.endswith(".h5"):
                    self.files.append(os.path.join(root, file))

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .h5 files found under {self.config['data']['path']}")

        # Open first files (one per batch slot)
        self.open_files = []
        for b in range(self.config["loader"]["batch_size"]):
            self.open_files.append(h5py.File(self.files[b % len(self.files)], "r"))

        # Build patch grid (list of (y0, y1, x0, x1)) for full resolution
        self.patch_grid = self._build_patch_grid(self.full_resolution, self.patch_size, self.patch_stride)
        self.num_patches = len(self.patch_grid)
        if self.num_patches == 0:
            raise ValueError("Patch grid is empty. Check patch_size/stride vs full resolution.")

        # Per-batch temporal index in the global event stream (search pointer)
        self.batch_row = [0 for _ in range(self.config["loader"]["batch_size"])]

        # Per-batch current patch index (linear over self.patch_grid)
        self.batch_patch_idx = [0 for _ in range(self.config["loader"]["batch_size"])]

        # For cycling files over time
        self.batch_idx = [i for i in range(self.config["loader"]["batch_size"])]

        # Tracks last processed timestamp (for dt_input computation)
        self.last_proc_timestamp = 0.0

    @staticmethod
    def _build_patch_grid(full_res, patch_size, stride):
        H, W = full_res
        ph, pw = patch_size
        sh, sw = stride
        grid = []
        # Non-overlapping by default (sh==ph, sw==pw). If different stride is given, allow overlap.
        for y0 in range(0, H - ph + 1, sh):
            for x0 in range(0, W - pw + 1, sw):
                grid.append((y0, y0 + ph, x0, x0 + pw))
        return grid

    def __len__(self):
        # Unbounded like original; training loop controls epochs.
        return 1000

    def __getitem__(self, index):
        while True:
            b = index % self.config["loader"]["batch_size"]

            # Current patch region for this batch slot
            patch_idx = self.batch_patch_idx[b] % self.num_patches
            y0, y1, x0, x1 = self.patch_grid[patch_idx]

            # Try to gather exactly window events within this patch, scanning forward in time
            xs, ys, ts, ps, reached_eof = self._get_events_in_patch(
                self.open_files[b], b, self.config["data"]["window"], y0, y1, x0, x1
            )

            # If too few events for this window or we reached EOF for this patch, move to next patch/file
            need_switch_patch = xs.shape[0] < self.config["data"]["window"]
            if need_switch_patch or reached_eof:
                # If we collected almost nothing, treat as finished for this patch
                self._switch_to_next_patch_or_file(b)
                continue  # try again with updated state

            # Handle very few events edge case (keep as empty tensors)
            if xs.shape[0] <= 10:
                xs = np.empty([0])
                ys = np.empty([0])
                ts = np.empty([0])
                ps = np.empty([0])

            # event formatting and timestamp normalization
            dt_input = np.asarray(0.0)
            if ts.shape[0] > 0:
                dt_input = np.asarray(ts[-1] - ts[0])
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # augmentation (flips and polarity) operates within patch coordinates
            xs, ys, ps = self.augment_events(xs, ys, ps, b)

            # encodings
            event_cnt = self.create_cnt_encoding(xs, ys, ps)
            event_mask = self.create_mask_encoding(xs, ys, ps)
            event_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
            event_list = self.create_list_encoding(xs, ys, ts, ps)
            event_list_pol_mask = self.create_polarity_mask(ps)

            # optional temporal cnt encoding: convert [2,H,W] -> [2,H,W] with
            # channel0 = current pos-neg, channel1 = previous pos-neg
            event_cnt = self.apply_temporal_cnt(event_cnt, b)

            # hot pixel removal (if enabled)
            if self.config["hot_filter"]["enabled"]:
                hot_mask = self.create_hot_mask(event_cnt, b)
                hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
                hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
                event_voxel = event_voxel * hot_mask_voxel
                event_cnt = event_cnt * hot_mask_cnt
                event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))

            # Advance temporal pointer within the same patch for next call
            # (already advanced in _get_events_in_patch by setting self.batch_row[b])

            # Prepare output dictionary (events-only)
            output = {
                "event_cnt": event_cnt,
                "event_voxel": event_voxel,
                "event_mask": event_mask,
                "event_list": event_list,
                "event_list_pol_mask": event_list_pol_mask,
                "dt_gt": torch.from_numpy(np.asarray(0.0)),  # not used in training
                "dt_input": torch.from_numpy(dt_input),
            }

            break

        return output

    def _switch_to_next_patch_or_file(self, b):
        """Advance to next patch; if we finished all patches, move to next file.
        Also mark new_seq so the training loop resets model states.
        """
        reset_on_patch = self.config["loader"].get("reset_on_patch_switch", False)
        # Only signal new_seq (and reset) on patch change if configured
        if reset_on_patch:
            self.new_seq = True
            if self.config["hot_filter"]["enabled"]:
                self.hot_idx[b] = 0
                self.hot_events[b] = torch.zeros(self.resolution)

        # Reset temporal pointer for the next patch
        self.batch_row[b] = 0

        # Next patch
        self.batch_patch_idx[b] += 1

        if self.batch_patch_idx[b] >= self.num_patches:
            # Move to next file and reset patch idx
            self.batch_patch_idx[b] = 0
            self.batch_idx[b] = max(self.batch_idx) + 1

            # Full sequence reset when changing files (increments seq_num and resamples augmentation)
            self.new_seq = True
            self.reset_sequence(b)

            # Swap file
            try:
                self.open_files[b].close()
            except Exception:
                pass
            next_file = self.files[self.batch_idx[b] % len(self.files)]
            self.open_files[b] = h5py.File(next_file, "r")

    def _get_events_in_patch(self, file, batch, target_num_events, y0, y1, x0, x1):
        """
        Collect exactly target_num_events events within the patch (y0:y1, x0:x1),
        scanning forward from the current search pointer self.batch_row[batch].

        Returns
        - xs, ys, ts, ps: numpy arrays (patch-local coordinates)
        - reached_eof: boolean indicating end-of-file reached while searching
        """
        # Current scan position in the global event arrays
        current_idx = int(self.batch_row[batch])
        N_total = len(file["events/xs"])  # total number of events in the file

        collected = 0
        # Start with a moderate chunk; adapt if too sparse
        chunk_size = max(1024, target_num_events // 2)

        xs_out, ys_out, ts_out, ps_out = [], [], [], []

        # Safety to avoid infinite loop if patch is extremely sparse
        max_search_events = max(target_num_events * 20, 50000)
        searched_events = 0
        reached_eof = False
        last_selected_global_index = None

        # Bounds for quick filtering
        y0_i, y1_i, x0_i, x1_i = int(y0), int(y1), int(x0), int(x1)

        while collected < target_num_events and searched_events < max_search_events:
            if current_idx >= N_total:
                reached_eof = True
                break

            start_idx = current_idx
            end_idx = min(current_idx + chunk_size, N_total)

            xs_chunk = file["events/xs"][current_idx:end_idx]
            ys_chunk = file["events/ys"][current_idx:end_idx]
            ts_chunk = file["events/ts"][current_idx:end_idx]
            ps_chunk = file["events/ps"][current_idx:end_idx]

            if len(xs_chunk) > 0:
                # Filter by patch bounds
                spatial_mask = (
                    (ys_chunk >= y0_i)
                    & (ys_chunk < y1_i)
                    & (xs_chunk >= x0_i)
                    & (xs_chunk < x1_i)
                )

                valid_idx = np.where(spatial_mask)[0]
                if valid_idx.size > 0:
                    needed = target_num_events - collected
                    take = min(needed, valid_idx.size)
                    sel = valid_idx[:take]

                    xs_out.extend(xs_chunk[sel] - x0_i)
                    ys_out.extend(ys_chunk[sel] - y0_i)
                    # normalize timestamps later; here keep absolute then subtract t0
                    ts_out.extend(ts_chunk[sel])
                    ps_out.extend(ps_chunk[sel])

                    collected += take
                    # Update the last selected global index (temporal pointer) to the last event we consumed
                    last_selected_global_index = start_idx + int(sel[-1])

            current_idx = end_idx
            searched_events += (end_idx - start_idx)

            # If we're finding too few, increase chunk size up to a cap
            if collected < target_num_events * 0.5:
                chunk_size = min(chunk_size * 2, max(8192, target_num_events * 4))

        # Convert to numpy arrays
        xs = np.asarray(xs_out, dtype=np.int32)
        ys = np.asarray(ys_out, dtype=np.int32)
        ts = np.asarray(ts_out, dtype=np.float64)
        ps = np.asarray(ps_out, dtype=np.int8)

        # Adjust timestamps to start at t0 (sequence start)
        if ts.size > 0:
            ts = ts - file.attrs["t0"]
            self.last_proc_timestamp = ts[-1]

        # Bounds check (should already be within 0..patch-1)
        if xs.size > 0 or ys.size > 0:
            ph, pw = self.patch_size
            valid = (ys >= 0) & (ys < ph) & (xs >= 0) & (xs < pw)
            if not np.all(valid):
                xs = xs[valid]
                ys = ys[valid]
                ts = ts[valid]
                ps = ps[valid]

        # Update scan pointer for next call: continue right after the last event we consumed
        if last_selected_global_index is not None:
            self.batch_row[batch] = last_selected_global_index + 1
        else:
            # If we didn't consume any events (extremely sparse), move to where we stopped scanning
            self.batch_row[batch] = current_idx

        return xs, ys, ts, ps, reached_eof

    # Keep a helper for completeness; not used in this loader but here for parity with original
    @staticmethod
    def find_ts_index(file, timestamp):
        return binary_search_array(file["events/ts"], timestamp)
