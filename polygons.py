"""4-corner polygon helpers: clean up Gemini's polygons, compare them, and straighten crops.

Adapted from Parth's parthswork/polygons branch (detect_shelves.py and shelf_metrics.py).

A polygon is 4 corners, each [y, x] normalised to 0-1000, ordered top-left, top-right,
bottom-right, bottom-left. Shelves seen at an angle are slanted quadrilaterals, so
cropping them straight (a perspective warp) gives the model an upright, enlarged view.
"""
import cv2
import numpy as np

Polygon = list[list[int]]

MIN_AREA = 200  # in 0-1000 units squared; smaller polygons are treated as noise
MIN_VISIBLE = 0.15  # drop units with less than this fraction left after nearer units
MIN_HIDDEN = 0.005  # a nearer unit covering less than this fraction doesn't count
GRID = 1000  # resolution of the normalised masks used to work out what is hidden
IOU_GRID = 200  # coarser grid for comparing polygons; plenty for an overlap ratio
HIDDEN_GREY = 128  # fill for parts of a crop covered by a nearer unit or outside the frame
MIN_LONG_SIDE = 1024  # scale straightened crops to at least this many pixels on the long side
MAX_UPSCALE = 4.0


def order_corners(xy: np.ndarray) -> np.ndarray:
    """Sort 4 (x, y) points clockwise starting from the top-left one."""
    centre = xy.mean(axis=0)
    xy = xy[np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))]
    return np.roll(xy, -int(np.argmin(xy.sum(axis=1))), axis=0)


def clean_polygon(polygon: Polygon) -> Polygon | None:
    """Clamp to 0-1000, force 4 corners in TL/TR/BR/BL order, and drop tiny shapes.

    Anything that isn't a convex quadrilateral is replaced by its tightest rotated
    rectangle, so it can always be straightened.
    """
    xy = np.array([[p[1], p[0]] for p in polygon if len(p) == 2], np.float32)
    if len(xy) < 3:
        return None
    xy = np.clip(xy, 0, 1000)
    quad = order_corners(xy) if len(xy) == 4 else None
    if quad is None or not cv2.isContourConvex(quad):
        quad = order_corners(np.clip(cv2.boxPoints(cv2.minAreaRect(xy)), 0, 1000))
    if cv2.contourArea(quad) < MIN_AREA:
        return None
    return [[round(float(y)), round(float(x))] for x, y in quad]


def to_pixels(polygon: Polygon, width: int, height: int) -> np.ndarray:
    """[[y, x], ...] in 0-1000 -> float32 array of (x, y) pixel coordinates."""
    return np.array([[x * width / 1000, y * height / 1000] for y, x in polygon], np.float32)


def polygon_mask(polygon: Polygon, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [to_pixels(polygon, width, height).round().astype(np.int32)], 255)
    return mask


def polygon_area(polygon: Polygon) -> float:
    """Area as a fraction of the whole frame (0-1)."""
    return float(cv2.contourArea(to_pixels(polygon, GRID, GRID))) / (GRID * GRID)


def clean_levels(levels: list[Polygon], unit: Polygon) -> list[Polygon]:
    """Clean shelf-row polygons, keep those centred inside their unit, sort top to bottom."""
    outline = to_pixels(unit, GRID, GRID)
    kept = []
    for level in levels:
        level = clean_polygon(level)
        if level is None:
            continue
        cx, cy = to_pixels(level, GRID, GRID).mean(axis=0)
        if cv2.pointPolygonTest(outline, (float(cx), float(cy)), False) >= 0:
            kept.append(level)
    return sorted(kept, key=lambda p: sum(y for y, _ in p))


def polygon_iou(a: Polygon, b: Polygon) -> float:
    ma, mb = polygon_mask(a, IOU_GRID, IOU_GRID), polygon_mask(b, IOU_GRID, IOU_GRID)
    union = np.count_nonzero(cv2.bitwise_or(ma, mb))
    return np.count_nonzero(cv2.bitwise_and(ma, mb)) / union if union else 0.0


def left_to_right(region: dict) -> tuple[int, int]:
    return min(x for _, x in region["polygon"]), min(y for y, _ in region["polygon"])


def link_and_number(regions: list[dict]) -> list[dict]:
    """Work out which nearer regions hide part of each one, then number them s1, s2, ...

    Each region needs `polygon` and `depth` (1 = nearest the camera). Regions that are
    almost entirely behind nearer ones are dropped. Adds `id` and `hidden_by` (ids of the
    nearer regions standing in front of it), numbered left to right.
    """
    masks = [polygon_mask(r["polygon"], GRID, GRID) for r in regions]
    kept = []
    for i, region in enumerate(regions):
        area = np.count_nonzero(masks[i])
        hidden = np.zeros_like(masks[i])
        fronts = []
        for j, other in enumerate(regions):
            if other["depth"] < region["depth"]:
                overlap = cv2.bitwise_and(masks[i], masks[j])
                if np.count_nonzero(overlap) > MIN_HIDDEN * area:
                    fronts.append(j)
                    hidden |= overlap
        if area == 0 or 1 - np.count_nonzero(hidden) / area < MIN_VISIBLE:
            continue
        kept.append((i, fronts))

    kept.sort(key=lambda item: left_to_right(regions[item[0]]))
    ids = {i: f"s{n}" for n, (i, _) in enumerate(kept, 1)}
    out = []
    for i, fronts in kept:
        out.append({**regions[i], "id": ids[i], "hidden_by": [ids[j] for j in fronts if j in ids]})
    return out


def straighten(image: np.ndarray, polygon: Polygon, front_polygons: list[Polygon],
               pad: float, levels: list[Polygon] = ()) -> np.ndarray:
    """Warp the polygon to an upright rectangle, enlarged for detail, with nearer regions
    in front of it (and anything outside the frame) filled flat grey. Shelf-row polygons
    in `levels` are outlined and labelled L1, L2, ... from the top."""
    h, w = image.shape[:2]
    if front_polygons:
        image = image.copy()
        for front in front_polygons:
            cv2.fillPoly(image, [to_pixels(front, w, h).round().astype(np.int32)],
                         (HIDDEN_GREY,) * 3)

    quad = to_pixels(polygon, w, h)
    centre = quad.mean(axis=0)
    quad = centre + (quad - centre) * (1 + 2 * pad)
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = min(max(MIN_LONG_SIDE / max(width, height), 1.0), MAX_UPSCALE)
    out_w, out_h = max(round(width * scale), 1), max(round(height * scale), 1)
    warp = cv2.getPerspectiveTransform(
        quad.astype(np.float32), np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]]))
    out = cv2.warpPerspective(image, warp, (out_w, out_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(HIDDEN_GREY,) * 3)

    # Draw after warping so outlines stay thin relative to the products.
    for n, level in enumerate(levels, 1):
        pts = cv2.perspectiveTransform(to_pixels(level, w, h).reshape(-1, 1, 2), warp)
        pts = pts.reshape(-1, 2).round().astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        org = (int(pts[0][0]) + 4, int(pts[0][1]) + 22)
        cv2.putText(out, f"L{n}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"L{n}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out
