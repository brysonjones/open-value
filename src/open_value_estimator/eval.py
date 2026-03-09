"""Evaluation visualization tool for value estimator.

Creates a video visualization with:
- Left: Original video frames from the episode in a camera grid
- Right: Combined ground-truth and predicted value plot

The plot updates in sync with the video frames.

Uses pure OpenCV/numpy for plot rendering instead of matplotlib,
pre-rendering static elements once and only drawing dynamic elements per frame.
"""

import logging
import math
import random
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from open_value_estimator.config import Config
from open_value_estimator.dataset import ValueDataset
from open_value_estimator.utils import preprocess_batch

# Layout constants
TARGET_WIDTH = 1000  # Target video width in pixels
TITLE_HEIGHT = 50  # Height of title bar
FPS = 10  # Output video framerate
BG_COLOR = (30, 26, 46)  # Dark background color (BGR)

# Computer Modern fonts (bundled with matplotlib)
_FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
FONT_REGULAR = str(_FONT_DIR / "cmr10.ttf")
FONT_BOLD = str(_FONT_DIR / "cmb10.ttf")


def pil_text(
    canvas: np.ndarray,
    text: str,
    pos: tuple[int, int],
    font_path: str = FONT_REGULAR,
    font_size: int = 14,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw text on an OpenCV BGR canvas using PIL for TrueType font support.

    Renders text onto the canvas in-place. The color should be in BGR format
    (matching OpenCV convention); it is converted to RGB internally for PIL.

    Args:
        canvas: BGR numpy array (H, W, 3), modified in-place.
        text: Text string to render.
        pos: (x, y) position for the top-left of the text bounding box.
        font_path: Path to a TTF/OTF font file.
        font_size: Font size in pixels.
        color: Text color in BGR format.
    """
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    # Render text onto a small RGBA image
    txt_img = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    draw = ImageDraw.Draw(txt_img)
    rgb_color = (color[2], color[1], color[0])  # BGR -> RGB
    draw.text((-bbox[0], -bbox[1]), text, font=font, fill=(*rgb_color, 255))

    txt_arr = np.array(txt_img)
    alpha = txt_arr[:, :, 3:4].astype(np.float32) / 255.0
    rgb = txt_arr[:, :, :3]
    # Convert RGB to BGR for OpenCV
    bgr = rgb[:, :, ::-1]

    # Composite onto canvas
    x, y = pos
    h, w = txt_arr.shape[:2]
    # Clip to canvas bounds
    y1, y2 = max(y, 0), min(y + h, canvas.shape[0])
    x1, x2 = max(x, 0), min(x + w, canvas.shape[1])
    sy, sx = y1 - y, x1 - x
    sh, sw = y2 - y1, x2 - x1
    if sh <= 0 or sw <= 0:
        return

    region = canvas[y1:y2, x1:x2].astype(np.float32)
    a = alpha[sy : sy + sh, sx : sx + sw]
    fg = bgr[sy : sy + sh, sx : sx + sw].astype(np.float32)
    canvas[y1:y2, x1:x2] = (fg * a + region * (1.0 - a)).astype(np.uint8)


def measure_text(
    text: str,
    font_path: str = FONT_REGULAR,
    font_size: int = 14,
) -> tuple[int, int]:
    """Measure text dimensions using PIL.

    Returns:
        (width, height) of the text bounding box in pixels.
    """
    font = ImageFont.truetype(font_path, font_size)
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def compute_grid_layout(n_cameras: int) -> tuple[int, int]:
    """Compute grid layout (cols, rows) for a given number of cameras."""
    if n_cameras == 1:
        return 1, 1
    elif n_cameras == 2:
        return 2, 1
    elif n_cameras <= 4:
        return 2, 2
    elif n_cameras <= 6:
        return 3, 2
    else:
        n_cols = 4
        return n_cols, (n_cameras + n_cols - 1) // n_cols


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    """Convert a hex color string to a BGR tuple for OpenCV."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 1,
    dash_length: int = 8,
    gap_length: int = 5,
) -> None:
    """Draw a dashed line on an image in-place."""
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = np.hypot(dx, dy)
    if dist < 1:
        return
    step = dash_length + gap_length
    num_dashes = int(dist / step) + 1
    for i in range(num_dashes):
        start_frac = i * step / dist
        end_frac = min((i * step + dash_length) / dist, 1.0)
        if start_frac >= 1.0:
            break
        start = (int(pt1[0] + dx * start_frac), int(pt1[1] + dy * start_frac))
        end = (int(pt1[0] + dx * end_frac), int(pt1[1] + dy * end_frac))
        cv2.line(img, start, end, color, thickness, cv2.LINE_AA)


class CVPlotRenderer:
    """Fast OpenCV-based plot renderer with multi-series support.

    Pre-renders all static elements (background, grid, axes, labels, faded trajectories,
    legend) into a base canvas once. Per-frame rendering only draws the solid line
    prefixes, cursor markers, and vertical cursor line on a copy of the base.
    """

    MARGIN_LEFT = 72
    MARGIN_RIGHT = 15
    MARGIN_TOP = 20
    MARGIN_BOTTOM = 52

    def __init__(
        self,
        series: list[dict],
        y_range: tuple[float, float] = (-1.0, 0.0),
        fps: float = 30.0,
        bg_color: str = "#1a1a2e",
        width: int = 400,
        height: int = 250,
    ):
        """Initialize the plot renderer.

        Args:
            series: List of dicts, each with "values" (list[float]),
                    "color" (hex string), and "label" (str).
            y_range: (min, max) for the y-axis data range.
            fps: Frame rate for converting frame indices to seconds on x-axis.
            bg_color: Background hex color.
            width: Plot width in pixels.
            height: Plot height in pixels.
        """
        self.fps = fps
        self.width = width
        self.height = height
        self.bg_bgr = hex_to_bgr(bg_color)

        # Process each series
        self.series_data = []
        for s in series:
            line_color = hex_to_bgr(s["color"])
            faded_color = tuple(
                int(c * 0.3 + bg * 0.7) for c, bg in zip(line_color, self.bg_bgr)
            )
            self.series_data.append({
                "values": s["values"],
                "label": s["label"],
                "line_color": line_color,
                "faded_color": faded_color,
            })

        # Plot area bounds (pixel coordinates)
        self.plot_x0 = self.MARGIN_LEFT
        self.plot_x1 = width - self.MARGIN_RIGHT
        self.plot_y0 = self.MARGIN_TOP
        self.plot_y1 = height - self.MARGIN_BOTTOM

        # Data bounds (x-axis in seconds)
        self.total_frames = max(len(s["values"]) for s in series)
        self.data_x_min = 0.0
        self.data_x_max = float(max(self.total_frames - 1, 1)) / self.fps
        self.data_y_min = y_range[0]
        self.data_y_max = y_range[1]

        # Pre-compute pixel coordinates for each series (frame index -> seconds)
        for sd in self.series_data:
            sd["all_points"] = np.array(
                [self._data_to_pixel(i / self.fps, v) for i, v in enumerate(sd["values"])],
                dtype=np.int32,
            )

        self.base_canvas = self._build_base()

    def _data_to_pixel(self, data_x: float, data_y: float) -> tuple[int, int]:
        """Map data coordinates to pixel coordinates."""
        px = int(
            self.plot_x0
            + (data_x - self.data_x_min)
            / (self.data_x_max - self.data_x_min)
            * (self.plot_x1 - self.plot_x0)
        )
        # Y is inverted (higher values = lower pixel y)
        py = int(
            self.plot_y0
            + (1.0 - (data_y - self.data_y_min) / (self.data_y_max - self.data_y_min))
            * (self.plot_y1 - self.plot_y0)
        )
        return (px, py)

    @staticmethod
    def _choose_nice_integer_step(x_max: float, target_ticks: int = 6) -> int:
        """Choose a whole-second tick step that scales with episode length."""
        if x_max <= 0:
            return 1

        rough_step = x_max / max(target_ticks, 1)
        rough_step = max(1.0, rough_step)
        base_exp = int(math.floor(math.log10(rough_step)))

        candidates: list[int] = []
        for exp in (base_exp - 1, base_exp, base_exp + 1):
            magnitude = 10 ** exp
            for multiplier in (1, 2, 3, 5, 10):
                step = int(multiplier * magnitude)
                if step >= 1:
                    candidates.append(step)

        candidates = sorted(set(candidates))
        return min(candidates, key=lambda s: abs((x_max / s) - target_ticks))

    def _get_x_ticks(self) -> list[int]:
        """Return whole-second x-axis ticks with an even, readable spacing."""
        if self.data_x_max <= 0:
            return [0]

        step = self._choose_nice_integer_step(self.data_x_max)
        max_whole_second = int(math.floor(self.data_x_max))
        return list(range(0, max_whole_second + 1, step))

    def _build_base(self) -> np.ndarray:
        """Build the static base canvas with grid, axes, labels, legend, and faded trajectories."""
        canvas = np.full((self.height, self.width, 3), self.bg_bgr, dtype=np.uint8)

        gray = (136, 136, 136)
        light_gray = (204, 204, 204)

        # --- Grid lines (dashed, gray) ---
        y_tick_count = 5
        y_data_range = self.data_y_max - self.data_y_min
        x_ticks = self._get_x_ticks()
        for i in range(y_tick_count + 1):
            y_data = self.data_y_min + i * y_data_range / y_tick_count
            _, py = self._data_to_pixel(0, y_data)
            if self.plot_y0 <= py <= self.plot_y1:
                draw_dashed_line(
                    canvas, (self.plot_x0, py), (self.plot_x1, py),
                    gray, thickness=1, dash_length=6, gap_length=4,
                )

        for x_tick in x_ticks:
            px, _ = self._data_to_pixel(float(x_tick), self.data_y_min)
            if self.plot_x0 <= px <= self.plot_x1:
                draw_dashed_line(
                    canvas, (px, self.plot_y0), (px, self.plot_y1),
                    gray, thickness=1, dash_length=6, gap_length=4,
                )

        # --- Axis spines ---
        cv2.line(canvas, (self.plot_x0, self.plot_y1), (self.plot_x1, self.plot_y1), gray, 1, cv2.LINE_AA)
        cv2.line(canvas, (self.plot_x0, self.plot_y0), (self.plot_x0, self.plot_y1), gray, 1, cv2.LINE_AA)

        # --- Tick labels (Computer Modern, small) ---
        tick_font_size = 16

        for i in range(y_tick_count + 1):
            y_data = self.data_y_min + i * y_data_range / y_tick_count
            _, py = self._data_to_pixel(0, y_data)
            if self.plot_y0 <= py <= self.plot_y1:
                label = f"{y_data:.1f}"
                tw, th = measure_text(label, FONT_REGULAR, tick_font_size)
                pil_text(canvas, label, (self.plot_x0 - tw - 6, py - th // 2),
                         FONT_REGULAR, tick_font_size, gray)

        for x_tick in x_ticks:
            px, _ = self._data_to_pixel(float(x_tick), self.data_y_min)
            if self.plot_x0 <= px <= self.plot_x1:
                label = str(x_tick)
                tw, th = measure_text(label, FONT_REGULAR, tick_font_size)
                pil_text(canvas, label, (px - tw // 2, self.plot_y1 + 6),
                         FONT_REGULAR, tick_font_size, gray)

        # --- Axis labels (Computer Modern, medium) ---
        axis_font_size = 20

        x_label = "Time (s)"
        xlw, xlh = measure_text(x_label, FONT_REGULAR, axis_font_size)
        x_label_x = (self.plot_x0 + self.plot_x1) // 2 - xlw // 2
        pil_text(canvas, x_label, (x_label_x, self.height - xlh - 4),
                 FONT_REGULAR, axis_font_size, light_gray)

        y_label = "Value"
        ylw, ylh = measure_text(y_label, FONT_REGULAR, axis_font_size)
        # Render y-label onto a small canvas, then rotate 90 degrees CCW
        y_label_canvas = np.full((ylh + 4, ylw + 4, 3), self.bg_bgr, dtype=np.uint8)
        pil_text(y_label_canvas, y_label, (2, 2), FONT_REGULAR, axis_font_size, light_gray)
        y_label_canvas = cv2.rotate(y_label_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
        lh, lw = y_label_canvas.shape[:2]
        ly = (self.plot_y0 + self.plot_y1) // 2 - lh // 2
        lx = 2
        if ly >= 0 and ly + lh <= self.height and lx + lw <= self.width:
            canvas[ly : ly + lh, lx : lx + lw] = y_label_canvas

        # --- Legend (top-left inside plot area, Computer Modern) ---
        legend_font_size = 18
        legend_line_length = 20
        legend_spacing = 26
        legend_padding = 8

        legend_entries = []
        for sd in self.series_data:
            tw, th = measure_text(sd["label"], FONT_REGULAR, legend_font_size)
            legend_entries.append((tw, th))

        legend_width = max(legend_line_length + 8 + tw for tw, _ in legend_entries) + 2 * legend_padding
        legend_height = len(legend_entries) * legend_spacing + 2 * legend_padding

        lx0 = self.plot_x0 + 8
        ly0 = self.plot_y0 + 5
        lx1 = lx0 + legend_width
        ly1 = ly0 + legend_height

        # Darken the legend area for contrast
        canvas[ly0:ly1, lx0:lx1] = (canvas[ly0:ly1, lx0:lx1].astype(np.float32) * 0.5).astype(np.uint8)

        for i, sd in enumerate(self.series_data):
            entry_y = ly0 + legend_padding + i * legend_spacing + legend_spacing // 2
            line_x0 = lx0 + legend_padding
            line_x1 = line_x0 + legend_line_length
            cv2.line(canvas, (line_x0, entry_y), (line_x1, entry_y), sd["line_color"], 2, cv2.LINE_AA)
            tw, th = legend_entries[i]
            pil_text(canvas, sd["label"], (line_x1 + 8, entry_y - th // 2),
                     FONT_REGULAR, legend_font_size, (255, 255, 255))

        # --- Faded full trajectory lines ---
        for sd in self.series_data:
            if len(sd["all_points"]) > 1:
                cv2.polylines(
                    canvas, [sd["all_points"]], isClosed=False,
                    color=sd["faded_color"], thickness=1, lineType=cv2.LINE_AA,
                )

        return canvas

    def render_frame(self, current_idx: int) -> np.ndarray:
        """Render the plot for a specific frame index.

        Copies the pre-built base canvas and draws the solid line prefixes,
        scatter markers, and vertical cursor line for all series.
        """
        frame = self.base_canvas.copy()

        if current_idx < 0:
            return frame

        # Vertical cursor line (dashed white) — use first series for x position
        first_pts = self.series_data[0]["all_points"]
        cursor_idx = min(current_idx, len(first_pts) - 1)
        px_cursor = first_pts[cursor_idx][0]
        draw_dashed_line(
            frame, (px_cursor, self.plot_y0), (px_cursor, self.plot_y1),
            (255, 255, 255), thickness=1, dash_length=6, gap_length=4,
        )

        # Draw solid lines and markers for each series
        for sd in self.series_data:
            if len(sd["all_points"]) == 0:
                continue
            idx = min(current_idx, len(sd["all_points"]) - 1)
            pts = sd["all_points"][:idx + 1]
            if len(pts) > 1:
                cv2.polylines(
                    frame, [pts], isClosed=False,
                    color=sd["line_color"], thickness=2, lineType=cv2.LINE_AA,
                )
            cx, cy = sd["all_points"][idx]
            cv2.circle(frame, (cx, cy), 5, sd["line_color"], -1, cv2.LINE_AA)

        return frame


def create_title_bar(
    text: str,
    width: int,
    height: int = TITLE_HEIGHT,
    bg_color: tuple[int, int, int] = (30, 26, 46),  # BGR
    text_color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    """Create a title bar image.

    Args:
        text: Title text.
        width: Bar width in pixels.
        height: Bar height in pixels.
        bg_color: Background color (BGR).
        text_color: Text color (BGR).

    Returns:
        BGR image array of shape [height, width, 3].
    """
    bar = np.full((height, width, 3), bg_color, dtype=np.uint8)

    title_font_size = 22
    tw, th = measure_text(text, FONT_BOLD, title_font_size)
    text_x = (width - tw) // 2
    text_y = (height - th) // 2
    pil_text(bar, text, (text_x, text_y), FONT_BOLD, title_font_size, text_color)

    return bar


def create_camera_grid(
    images: torch.Tensor,
    target_width: int,
    bg_color: tuple[int, int, int] = BG_COLOR,
) -> np.ndarray:
    """Arrange multiple camera images into a centered grid.

    Args:
        images: Tensor of shape [N, C, H, W] with N camera images in [0, 1].
        target_width: Target width for the entire grid.
        bg_color: Background color for padding (BGR).

    Returns:
        BGR image array with all cameras arranged in a grid.
    """
    n_cameras = images.shape[0]
    _, img_h, img_w = images[0].shape
    aspect_ratio = img_h / img_w

    n_cols, n_rows = compute_grid_layout(n_cameras)

    cell_width = target_width // n_cols
    cell_height = int(cell_width * aspect_ratio)

    grid_width = target_width
    grid_height = cell_height * n_rows

    grid = np.full((grid_height, grid_width, 3), bg_color, dtype=np.uint8)

    # Batch convert: [N, C, H, W] -> [N, H, W, C], float -> uint8, RGB -> BGR
    imgs_np = (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    imgs_bgr = np.ascontiguousarray(imgs_np[:, :, :, ::-1])  # RGB -> BGR, single contiguous copy

    for i in range(n_cameras):
        row = i // n_cols
        col = i % n_cols

        # Center incomplete last row
        images_in_last_row = n_cameras - (n_rows - 1) * n_cols
        if row == n_rows - 1 and images_in_last_row < n_cols:
            offset = (n_cols - images_in_last_row) * cell_width // 2
            x_start = col * cell_width + offset
        else:
            x_start = col * cell_width

        y_start = row * cell_height

        img_bgr = cv2.resize(
            imgs_bgr[i], (cell_width, cell_height), interpolation=cv2.INTER_LINEAR
        )

        grid[y_start : y_start + cell_height, x_start : x_start + cell_width] = img_bgr

    return grid


class EpisodeSequentialSampler(torch.utils.data.Sampler):
    """Sampler that yields frame indices for a single episode in sequential order."""

    def __init__(self, start_idx: int, end_idx: int):
        self.indices = list(range(start_idx, end_idx))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def get_available_camera_view_keys(dataset: ValueDataset) -> list[str]:
    """Get available camera feature keys from dataset metadata."""
    feature_keys: list[str] = []

    if hasattr(dataset.meta, "features") and dataset.meta.features is not None:
        feature_keys = list(dataset.meta.features.keys())
    elif hasattr(dataset, "hf_dataset"):
        feature_keys = list(dataset.hf_dataset.column_names)

    return [k for k in feature_keys if "observation.images" in k and k != "observation.images"]


def resolve_camera_view_keys(
    requested_views: list[str] | None,
    available_keys: list[str],
) -> list[str] | None:
    """Resolve requested camera views to dataset keys.

    Accepts either full feature keys (e.g., "observation.images.front")
    or suffix names (e.g., "front").
    """
    if not requested_views:
        return None

    available_set = set(available_keys)
    suffix_to_keys: dict[str, list[str]] = {}
    for key in available_keys:
        suffix = key.split("observation.images.", 1)[1] if key.startswith("observation.images.") else key
        suffix_to_keys.setdefault(suffix, []).append(key)

    resolved: list[str] = []
    unknown: list[str] = []
    ambiguous: dict[str, list[str]] = {}

    for view in requested_views:
        if view in available_set:
            resolved.append(view)
            continue

        matches = suffix_to_keys.get(view, [])
        if len(matches) == 1:
            resolved.append(matches[0])
        elif len(matches) > 1:
            ambiguous[view] = matches
        else:
            unknown.append(view)

    if unknown or ambiguous:
        parts: list[str] = []
        if unknown:
            parts.append(f"Unknown camera views: {unknown}.")
        if ambiguous:
            ambiguous_str = "; ".join(f"{k} -> {v}" for k, v in ambiguous.items())
            parts.append(f"Ambiguous camera views: {ambiguous_str}.")
        parts.append(f"Available camera views: {available_keys}")
        raise ValueError(" ".join(parts))

    # Deduplicate while preserving user order
    deduped: list[str] = []
    seen: set[str] = set()
    for key in resolved:
        if key not in seen:
            deduped.append(key)
            seen.add(key)

    return deduped


def create_eval_dataloader(
    dataset: ValueDataset,
    episode_idx: int,
    batch_size: int = 16,
    num_workers: int = 4,
) -> tuple[torch.utils.data.DataLoader, int, int]:
    """Create a dataloader for sequential evaluation of a single episode.

    Args:
        dataset: The ValueDataset instance.
        episode_idx: Episode index to evaluate.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.

    Returns:
        Tuple of (dataloader, start_idx, end_idx).
    """
    ep_meta = dataset.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    end_idx = ep_meta["dataset_to_index"]

    sampler = EpisodeSequentialSampler(start_idx, end_idx)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return dataloader, start_idx, end_idx


def _stack_recording_camera_images(
    batch: dict,
    camera_views: list[str] | None = None,
) -> torch.Tensor:
    """Stack camera images for video rendering without changing model inputs."""
    all_image_keys = [k for k in batch if "observation.images" in k and k != "observation.images"]
    selected_image_keys = camera_views or all_image_keys

    if camera_views is not None:
        available = set(all_image_keys)
        missing = [k for k in camera_views if k not in available]
        if missing:
            raise ValueError(
                f"Requested camera view keys not found in batch: {missing}. "
                f"Available: {sorted(all_image_keys)}"
            )

    images: list[torch.Tensor] = []
    for key in selected_image_keys:
        tensor = batch[key]
        if isinstance(tensor, torch.Tensor) and tensor.dim() >= 4:
            if tensor.dim() == 5:
                tensor = tensor[:, -1, ...]  # [B, T, C, H, W] -> [B, C, H, W]
            images.append(tensor)

    if not images:
        if camera_views:
            raise ValueError(f"No image tensors found for requested camera views: {camera_views}")
        raise ValueError("No camera image tensors found in batch.")

    return torch.stack(images, dim=1)  # [B, N, C, H, W]


@torch.no_grad()
def run_episode_predictions(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    start_idx: int,
    end_idx: int,
    value_targets: torch.Tensor,
    device: torch.device,
    state_stats: dict[str, torch.Tensor] | None = None,
    camera_views: list[str] | None = None,
) -> tuple[list[float], list[float], list[torch.Tensor], str]:
    """Run predictions for all frames in an episode using batched inference.

    Args:
        model: The value estimator model.
        dataloader: Sequential dataloader for the episode.
        start_idx: Start frame index in dataset.
        end_idx: End frame index in dataset.
        value_targets: Precomputed value targets tensor.
        device: Torch device.
        state_stats: Optional normalization stats for observation.state.
        camera_views: Optional ordered list of camera feature keys to render.
            This only affects video output, not model inference inputs.

    Returns:
        Tuple of (predictions, gt_values, frame_images, task) where frame_images
        is a list of per-frame image tensors [N, C, H, W] on CPU.
    """
    model.eval()
    predictions = []
    gt_values = value_targets[start_idx:end_idx].tolist()
    frame_images = []
    task = ""

    for raw_batch in dataloader:
        # Build video frames from selected cameras, but keep full camera inputs for model inference.
        recording_images = _stack_recording_camera_images(raw_batch, camera_views)

        batch = preprocess_batch(
            raw_batch,
            device,
            state_stats=state_stats,
        )

        logits = model(batch)
        batch_predictions = model.get_expected_value(logits).tolist()
        predictions.extend(batch_predictions)

        # Bulk transfer images to CPU (single operation per batch)
        batch_images = recording_images.cpu()  # [B, N, C, H, W]
        for i in range(batch_images.shape[0]):
            frame_images.append(batch_images[i])  # [N, C, H, W]

        # Capture task from first batch
        if not task and "task" in batch:
            if isinstance(batch["task"], list):
                task = batch["task"][0]
            else:
                task = batch["task"]

    return predictions, gt_values, frame_images, task


def create_evaluation_video(
    model: torch.nn.Module,
    dataset: ValueDataset,
    output_path: Path,
    episode_idx: int | None = None,
    device: torch.device | None = None,
    batch_size: int = 16,
    num_workers: int = 4,
    camera_views: list[str] | None = None,
    show_ground_truth_reward: bool = True,
) -> Path:
    """Create an evaluation video for an episode.

    Args:
        model: The trained value estimator.
        dataset: The ValueDataset.
        output_path: Path to save the output video.
        episode_idx: Episode index (random if None).
        device: Torch device.
        batch_size: Batch size for inference.
        num_workers: Number of data loading workers.
        camera_views: Optional ordered list of camera feature keys to render.
            This only affects video output, not model inference inputs.
        show_ground_truth_reward: Whether to include the ground-truth target
            line in the rollout plot.

    Returns:
        Path to the saved video file.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if episode_idx is None:
        episode_idx = random.randint(0, dataset.meta.total_episodes - 1)

    logging.info(f"Creating evaluation video for episode {episode_idx}")
    if camera_views:
        logging.info(f"Using selected camera views for recording: {camera_views}")

    # Create eval dataloader for this episode
    dataloader, start_idx, end_idx = create_eval_dataloader(
        dataset, episode_idx, batch_size=batch_size, num_workers=num_workers
    )
    num_frames = end_idx - start_idx
    logging.info(f"Episode has {num_frames} frames")

    # Run batched predictions (returns only image tensors, not full batch dicts)
    predictions, gt_values, frame_images, task = run_episode_predictions(
        model, dataloader, start_idx, end_idx, dataset.value_targets, device,
        state_stats=dataset.state_stats,
        camera_views=camera_views,
    )

    if not frame_images:
        raise ValueError(f"Episode {episode_idx} has no frames (start={start_idx}, end={end_idx}).")

    # Determine camera grid dimensions from first frame
    first_images = frame_images[0]  # [N, C, H, W]
    n_cameras = first_images.shape[0]
    _, img_h, img_w = first_images[0].shape
    aspect_ratio = img_h / img_w

    n_cols, n_rows = compute_grid_layout(n_cameras)

    cell_width = TARGET_WIDTH // n_cols
    cell_height = int(cell_width * aspect_ratio)
    video_height = cell_height * n_rows

    # Plot: same height as camera grid, 4:5 (height:width) aspect ratio
    plot_height = video_height
    plot_width = int(plot_height * 5 / 4)

    total_width = TARGET_WIDTH + plot_width
    total_height = TITLE_HEIGHT + video_height

    logging.info(
        f"Video layout: {n_cameras} cameras in {n_rows}x{n_cols} grid, "
        f"frame size {total_width}x{total_height}"
    )

    # Set up video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (total_width, total_height))

    # Title bar spans full width
    task_display = task[:80] + "..." if len(task) > 80 else task
    title_text = f"Task: {task_display}" if task_display else "Task"
    title_bar = create_title_bar(title_text, total_width)

    plot_series = [
        {"values": predictions, "color": "#3498db", "label": "Predicted"},
    ]
    if show_ground_truth_reward:
        plot_series.insert(0, {"values": gt_values, "color": "#e74c3c", "label": "Ground Truth"})

    # Single plot renderer with predicted series and optional ground-truth overlay
    renderer = CVPlotRenderer(
        series=plot_series,
        y_range=(-1.0, 0.0),
        fps=dataset.fps,
        width=plot_width,
        height=plot_height,
    )

    # Generate frames: camera grid (left) + plot (right), title bar on top
    for frame_idx, images in enumerate(frame_images):
        camera_grid = create_camera_grid(images, TARGET_WIDTH)
        plot = renderer.render_frame(frame_idx)
        content_row = np.hstack([camera_grid, plot])
        full_frame = np.vstack([title_bar, content_row])
        writer.write(full_frame)

    writer.release()
    logging.info(f"Saved evaluation video to {output_path}")

    return output_path


def evaluate(
    model: torch.nn.Module,
    cfg: Config,
    output_dir: Path,
    step: int,
    device: torch.device,
    dataset: ValueDataset | None = None,
) -> Path:
    """Run evaluation and create visualization video.

    Args:
        model: The value estimator model.
        cfg: Config object.
        output_dir: Output directory for videos.
        step: Current training step (for filename).
        device: Torch device.
        dataset: Optional pre-loaded dataset to avoid redundant construction.

    Returns:
        Path to the saved video.
    """
    if dataset is None:
        dataset = ValueDataset(cfg.data)

    episode_idx = random.randint(0, dataset.meta.total_episodes - 1)
    video_path = output_dir / "eval_videos" / f"eval_step_{step}_ep{episode_idx}.mp4"

    return create_evaluation_video(
        model=model,
        dataset=dataset,
        output_path=video_path,
        episode_idx=episode_idx,
        device=device,
        show_ground_truth_reward=cfg.eval.show_ground_truth_reward,
    )


def main() -> None:
    """CLI entry point for standalone evaluation.

    Usage:
        python -m open_value_estimator.eval \
            --checkpoint path/to/model.safetensors \
            --dataset lerobot/dataset_repo_id \
            --episodes 0 5 10 42 \
            --camera-views front wrist
    """
    import argparse

    from open_value_estimator.config import DataConfig
    from open_value_estimator.value_estimator import OpenValueEstimator

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Evaluate a value estimator checkpoint on specific episodes.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the safetensors checkpoint file.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo ID (e.g. lerobot/my_dataset).")
    parser.add_argument("--data-root", type=str, default=None, help="Local root path for the dataset.")
    parser.add_argument("--episodes", type=int, nargs="+", required=True, help="Episode indices to evaluate.")
    parser.add_argument("--output-dir", type=str, default="./eval_outputs", help="Directory to save output videos.")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect cuda/cpu).")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference.")
    parser.add_argument(
        "--camera-views",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Camera views to render. Accepts full feature keys "
            "(e.g. observation.images.front) or suffix names (e.g. front). "
            "This only affects the saved video, not model inputs."
        ),
    )
    parser.add_argument(
        "--hide-ground-truth-reward",
        dest="show_ground_truth_reward",
        action="store_false",
        default=True,
        help="Hide the ground-truth target line from the rollout plot.",
    )
    parser.add_argument("--no-ema", action="store_true", help="Skip EMA weights, use regular checkpoint.")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    data_cfg = DataConfig(repo_id=args.dataset, root=args.data_root or args.dataset)
    logging.info(f"Loading dataset: {data_cfg.repo_id}")
    dataset = ValueDataset(data_cfg)
    available_camera_views = get_available_camera_view_keys(dataset)
    if not available_camera_views:
        parser.error("No camera view columns found in dataset (expected keys like 'observation.images.<view>').")

    try:
        selected_camera_views = resolve_camera_view_keys(args.camera_views, available_camera_views)
    except ValueError as e:
        parser.error(str(e))

    if selected_camera_views:
        logging.info(f"Selected camera views for recording: {selected_camera_views}")
    else:
        logging.info(f"Using all camera views: {available_camera_views}")

    logging.info(f"Loading model from {args.checkpoint}")
    model = OpenValueEstimator.from_pretrained(
        path=args.checkpoint,
        device=device,
        use_ema=not args.no_ema,
    )

    total_episodes = dataset.meta.total_episodes
    invalid = [ep for ep in args.episodes if ep < 0 or ep >= total_episodes]
    if invalid:
        raise ValueError(
            f"Invalid episode indices {invalid}. Dataset has {total_episodes} episodes (0-{total_episodes - 1})."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx in args.episodes:
        output_path = output_dir / f"eval_ep{ep_idx}.mp4"
        logging.info(f"Evaluating episode {ep_idx} -> {output_path}")
        create_evaluation_video(
            model=model,
            dataset=dataset,
            output_path=output_path,
            episode_idx=ep_idx,
            device=device,
            batch_size=args.batch_size,
            camera_views=selected_camera_views,
            show_ground_truth_reward=args.show_ground_truth_reward,
        )

    logging.info(f"Done. Saved {len(args.episodes)} videos to {output_dir}")


if __name__ == "__main__":
    main()
