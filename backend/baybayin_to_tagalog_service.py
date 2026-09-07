import io
import os
import uuid

import cv2
import numpy as np
from PIL import Image as PILImage, ImageOps
from skimage.feature import hog

TEMP_ROOT = 'temp_crops'


def decode_grayscale_exif_corrected(image_bytes):
    pil_img = PILImage.open(io.BytesIO(image_bytes))
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert('L')
    return np.array(pil_img)


def ensure_dirs():
    os.makedirs(TEMP_ROOT, exist_ok=True)


def preprocess_image(gray_img):
    _, thresh = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh


def segment_glyphs(bin_img, pad=0, min_area=40, merge_kernel=(5, 35)):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, merge_kernel)
    dilated = cv2.dilate(bin_img, kernel, iterations=1)
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    height, width = bin_img.shape[:2]
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width * box_height < min_area:
            continue
        boxes.append((
            max(0, x - pad), max(0, y - pad),
            min(width, x + box_width + pad),
            min(height, y + box_height + pad),
        ))

    if not boxes:
        return []

    boxes.sort(key=lambda box: box[1])
    rows = [[boxes[0]]]
    for box in boxes[1:]:
        previous = rows[-1][-1]
        previous_center = (previous[1] + previous[3]) / 2
        current_center = (box[1] + box[3]) / 2
        row_height = previous[3] - previous[1]
        if abs(current_center - previous_center) < row_height * 0.5:
            rows[-1].append(box)
        else:
            rows.append([box])

    return [box for row in rows for box in sorted(row, key=lambda item: item[0])]


def tighten_boxes(bin_img, boxes, pad=0):
    height, width = bin_img.shape[:2]
    tightened = []
    for x0, y0, x1, y1 in boxes:
        sub = bin_img[y0:y1, x0:x1]
        ys, xs = np.where(sub > 0)
        if len(xs) == 0:
            tightened.append((x0, y0, x1, y1))
            continue
        tightened.append((
            max(0, x0 + xs.min() - pad),
            max(0, y0 + ys.min() - pad),
            min(width, x0 + xs.max() + 1 + pad),
            min(height, y0 + ys.max() + 1 + pad),
        ))
    return tightened


def tight_crop_glyph(crop_bin):
    """
    Tightly crops to the glyph's ink bounding box but does NOT resize.
    Resizing now happens per-component (base vs diacritic) inside
    classify_glyph, AFTER separation - matching how the training
    images were built (separate first at native resolution, then
    resize each piece independently).
    """
    points = cv2.findNonZero(crop_bin)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return crop_bin[y:y + height, x:x + width]


def tight_crop_glyph_with_offset(crop_bin):
    """
    Same as tight_crop_glyph, but also returns the (x, y) offset of the
    crop's top-left corner relative to crop_bin's own origin, so a
    caller can add it to crop_bin's own absolute position and keep
    track of exactly where this glyph sits in the ORIGINAL image -
    needed for drawing bounding boxes on the source photo later.
    Returns (None, None) if crop_bin has no ink.
    """
    points = cv2.findNonZero(crop_bin)
    if points is None:
        return None, None
    x, y, width, height = cv2.boundingRect(points)
    return (x, y), crop_bin[y:y + height, x:x + width]


GLYPH_REFINE_MERGE_KERNEL = (18, 18)
GLYPH_REFINE_MIN_AREA = 20


def find_glyph_clusters(binary_img, kernel_size=GLYPH_REFINE_MERGE_KERNEL, min_area=GLYPH_REFINE_MIN_AREA):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    dilated = cv2.dilate(binary_img, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_area:
            continue
        boxes.append((x, y, x + w, y + h))

    boxes.sort(key=lambda b: (b[0], b[1]))  # left-to-right within a single glyph box
    return boxes


def tighten_to_ink(binary_img, box):
    """
    Shrinks a cluster's box down to just its actual ink, undoing the
    dilation used only for finding the cluster boundary. Returns the
    new ABSOLUTE box (in binary_img's coordinate space, same space
    `box` was already given in) alongside the cropped pixels, so
    callers can keep drawing bounding boxes correctly instead of
    losing track of position after this extra crop.
    """
    x0, y0, x1, y1 = box
    sub = binary_img[y0:y1, x0:x1]
    ys, xs = np.where(sub > 0)
    if len(xs) == 0:
        return box, sub
    new_box = (
        x0 + int(xs.min()), y0 + int(ys.min()),
        x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1,
    )
    tightened = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return new_box, tightened


def split_into_single_glyphs(crop_bin):

    clusters = find_glyph_clusters(crop_bin)
    if len(clusters) <= 1:
        h, w = crop_bin.shape[:2]
        return [((0, 0, w, h), crop_bin)]
    return [tighten_to_ink(crop_bin, box) for box in clusters]


def crop_and_pad_to_square(img, pad_frac=0.15):
    """
    Crops tightly to the foreground's bounding box, then pads back out
    to a square canvas (centered, black padding) so a mark's true
    aspect ratio survives the later resize instead of being distorted.
    Mirrors the same helper used in the training script's diacritic path.
    """
    coords = cv2.findNonZero(img)
    if coords is None:
        return img
    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y:y + h, x:x + w]
    side = max(w, h)
    pad = max(int(side * pad_frac), 1)
    side_padded = side + 2 * pad
    square = np.zeros((side_padded, side_padded), dtype=img.dtype)
    y_offset = (side_padded - h) // 2
    x_offset = (side_padded - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped
    return square


def group_boxes_into_words(boxes):
    if not boxes:
        return []

    ordered = sorted(boxes, key=lambda box: (box[1] + box[3], box[0]))
    rows = [[ordered[0]]]
    for box in ordered[1:]:
        previous = rows[-1][-1]
        previous_center = (previous[1] + previous[3]) / 2
        current_center = (box[1] + box[3]) / 2
        row_height = max(1, previous[3] - previous[1])
        if abs(current_center - previous_center) < row_height * 0.5:
            rows[-1].append(box)
        else:
            rows.append([box])

    words = []
    for row in rows:
        row = sorted(row, key=lambda box: box[0])
        reference_width = float(np.median([box[2] - box[0] for box in row]))
        word_gap = max(10.0, reference_width * 0.9)
        current_word = [row[0]]
        for box in row[1:]:
            previous = current_word[-1]
            gap = box[0] - previous[2]
            if gap > word_gap:
                words.append(current_word)
                current_word = [box]
            else:
                current_word.append(box)
        words.append(current_word)
    return words


def _class_name(classes, prediction):
    return classes[int(prediction)] if not isinstance(prediction, str) else prediction


def _predict_with_confidence(model, features):
    prediction = model.predict(features)[0]
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        return prediction, float(np.max(probabilities))
    return prediction, 1.0


# A break in a single stroke caused by thresholding/anti-aliasing sits at
# a near-constant tiny gap (~2px) essentially independent of how big the
# glyph was drawn/scanned - it's a rendering artifact, not a stylistic
# pen-lift. A genuine diacritic mark (dot, bar, cross) is a deliberate
# pen-lift and measures several times that, whether the glyph is a 24px
# crop or a 128px one. That's why gap size in raw pixels, not a ratio of
# component size or bounding-box overlap, is the reliable signal here.
SAME_STROKE_GAP_THRESHOLD = 1.9


def _distance_transform_gap(labels, mask_label_ids, candidate_label_id):
    """Approximate nearest-pixel distance from `candidate_label_id`'s ink
    to the union of components in `mask_label_ids`, via a distance
    transform (no scipy dependency needed)."""
    mask = np.isin(labels, mask_label_ids).astype(np.uint8) * 255
    inverted = np.where(mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    return float(dist[labels == candidate_label_id].min())


def separate_base_and_diacritic(labels, stats, num_labels,
                                 gap_threshold=SAME_STROKE_GAP_THRESHOLD):
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda t: t[1], reverse=True)

    base_idx_list = [areas[0][0]]
    remaining = [idx for idx, _ in areas[1:]]

    keep_merging = True
    while keep_merging and remaining:
        keep_merging = False
        still_remaining = []
        for idx in remaining:
            gap = _distance_transform_gap(labels, base_idx_list, idx)
            if gap <= gap_threshold:
                base_idx_list.append(idx)
                keep_merging = True
            else:
                still_remaining.append(idx)
        remaining = still_remaining

    dia_idx = None
    if remaining:
        remaining_by_area = sorted(
            remaining, key=lambda idx: stats[idx, cv2.CC_STAT_AREA], reverse=True
        )
        dia_idx = remaining_by_area[0]

    return base_idx_list, dia_idx


# Threshold for the ROTATION-AWARE bar check below. Unlike the old
# axis-aligned dw/dh test, this is measured along the mark's own
# minimum-area rectangle, so a curved or diagonally-drawn dash still
# reads as "long and thin" instead of being penalized for not being
# perfectly horizontal. Tune this after checking real samples (see the
# debug hook in classify_glyph).
BAR_ROTATED_ASPECT_THRESHOLD = 2.2

# Below this pixel AREA (dw * dh), a diacritic crop is considered too
# small for shape-based analysis (solidity, aspect ratio) to be
# trustworthy - a handful of pixels can look "concave" purely from
# thresholding jaggedness, not because it's genuinely an X. Marks this
# small default straight to Dot instead of risking a false Bar/X
# override. Tune from real small-dot samples if dots are still being
# misclassified, or if genuinely tiny X marks start getting missed.
MIN_PIXELS_FOR_SHAPE_ANALYSIS = 16


def classify_glyph(native_crop, base_model, dia_model, base_classes, dia_classes,
                   solidity_threshold=0.80,
                   bar_rotated_aspect_threshold=BAR_ROTATED_ASPECT_THRESHOLD,
                   min_pixels_for_shape_analysis=MIN_PIXELS_FOR_SHAPE_ANALYSIS,
                   debug=False):
    """
    native_crop: tight-cropped binary glyph at ITS ORIGINAL resolution
    (NOT yet resized to 56x56). Splitting into base/diacritic happens
    here, on the native-resolution image, so each piece can be resized
    independently afterward - matching the training pipeline's order.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(native_crop)
    predicted_base_name = 'Unknown'
    predicted_dia_name = 'None'
    position = 'None'
    base_confidence = 0.0
    dia_confidence = 0.0

    if num_labels <= 1:
        return predicted_base_name, predicted_dia_name, '', 0.0

    # ---- WHOLE-CROP FALLBACK CLASSIFICATION ----
    # Classify the ENTIRE native_crop as a single base character, with
    # no diacritic split attempted. Standalone vowels (A/EI/OU) can have
    # a natural stroke gap (e.g. a zigzag pen-lift) that the diacritic-
    # separation logic below would otherwise misread as "base +
    # diacritic". Computing this fallback up front lets us catch and
    # correct that case afterward, since a real diacritic split should
    # never legitimately resolve to a standalone vowel.
    whole_mask = np.where(native_crop > 0, 255, 0).astype(np.uint8)
    whole_coords = cv2.findNonZero(whole_mask)
    wx, wy, ww, wh = cv2.boundingRect(whole_coords)
    whole_crop = whole_mask[wy:wy + wh, wx:wx + ww]
    whole_norm = cv2.resize(whole_crop, (56, 56)).astype(np.float32) / 255.0
    hog_whole = hog(
        whole_norm, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, visualize=False,
    ).reshape(1, -1)
    whole_prediction, whole_confidence = _predict_with_confidence(base_model, hog_whole)
    whole_base_name = _class_name(base_classes, whole_prediction)

    base_idx_list, dia_idx = separate_base_and_diacritic(labels, stats, num_labels)

    if dia_idx is None:
        # No component sits cleanly outside the base's vertical span -
        # this whole native_crop IS the base glyph (possibly reunited
        # from multiple disconnected pieces of the same stroke). The
        # whole-crop classification above already covers exactly this
        # case, so reuse it directly instead of reclassifying.
        predicted_base_name = whole_base_name
        base_confidence = whole_confidence
    else:
        # ---- BASE: union of every component that overlaps the anchor
        # component's vertical span (reunites a base whose stroke isn't
        # fully pixel-connected), then crop tightly to ITS OWN combined
        # bounding box (not the full glyph canvas) before resizing.
        base_mask_full = np.isin(labels, base_idx_list).astype(np.uint8) * 255
        coords = cv2.findNonZero(base_mask_full)
        bx, by, bw, bh = cv2.boundingRect(coords)
        base_crop = base_mask_full[by:by + bh, bx:bx + bw]
        base_norm = cv2.resize(base_crop, (56, 56)).astype(np.float32) / 255.0

        # ---- DIACRITIC: crop tightly to its own bbox, then pad to a
        # square canvas before resizing, so its aspect ratio (dot vs
        # dash) survives - mirrors the training script's diacritic fix.
        dx, dy, dw, dh, _ = stats[dia_idx]
        if dw == 0 or dh == 0:
            predicted_base_name = whole_base_name
            base_confidence = whole_confidence
            return predicted_base_name, predicted_dia_name, predicted_base_name, base_confidence

        dia_mask_full = (labels == dia_idx).astype(np.uint8) * 255
        dia_crop = dia_mask_full[dy:dy + dh, dx:dx + dw]
        dia_padded = crop_and_pad_to_square(dia_crop)

        # ---- SOLIDITY: upscale with CUBIC (not nearest-neighbor) so
        # jagged single-pixel edges from a tiny native crop get smoothed
        # out rather than amplified into hard, artificial notches.
        # Re-threshold afterward since cubic interpolation introduces
        # gray values into what must stay a binary shape.
        dia_upscaled = cv2.resize(dia_crop, (40, 40), interpolation=cv2.INTER_CUBIC)
        _, dia_upscaled = cv2.threshold(dia_upscaled, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            dia_upscaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            hull_area = cv2.contourArea(cv2.convexHull(largest_contour))
            solidity = (
                cv2.contourArea(largest_contour) / hull_area
                if hull_area > 0 else 1.0
            )
        else:
            solidity = 1.0

        dia_norm = cv2.resize(dia_padded, (56, 56)).astype(np.float32) / 255.0

        hog_base = hog(
            base_norm, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, visualize=False,
        ).reshape(1, -1)
        hog_dia = hog(
            dia_norm, orientations=9, pixels_per_cell=(4, 4),
            cells_per_block=(1, 1), transform_sqrt=True, visualize=False,
        ).reshape(1, -1)
        base_prediction, base_confidence = _predict_with_confidence(base_model, hog_base)
        dia_prediction, dia_confidence = _predict_with_confidence(dia_model, hog_dia)
        predicted_base_name = _class_name(base_classes, base_prediction)
        predicted_dia_name = _class_name(dia_classes, dia_prediction)
        svm_raw_dia_prediction = predicted_dia_name  # snapshot before geometric override

        # position is relative to the (possibly multi-component) base's
        # combined centroid, not just the single largest base piece.
        base_centroid_y = by + (bh / 2.0)
        position = 'Above' if centroids[dia_idx][1] < base_centroid_y else 'Below'

        # ---- ROTATION-AWARE BAR CHECK ----
        dia_points = cv2.findNonZero(dia_crop)
        if dia_points is not None:
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(dia_points)
            long_side = max(rect_w, rect_h)
            short_side = max(min(rect_w, rect_h), 1e-6)  # avoid divide-by-zero
            rotated_aspect_ratio = long_side / short_side
        else:
            rotated_aspect_ratio = float(dw) / float(dh)

        is_thin_bar = rotated_aspect_ratio >= bar_rotated_aspect_threshold
        is_too_small_for_shape_analysis = (dw * dh) < min_pixels_for_shape_analysis

        # ---- MINIMUM-SIZE GUARD ----
        if is_too_small_for_shape_analysis:
            predicted_dia_name = 'Dot'
        elif is_thin_bar:
            predicted_dia_name = 'Bar'
        elif solidity < solidity_threshold:
            available_classes = [str(value).lower() for value in dia_classes]
            if 'x' in available_classes:
                predicted_dia_name = 'X'
            elif 'cross' in available_classes:
                predicted_dia_name = 'Cross'
        else:
            predicted_dia_name = 'Dot'

        # ---- VOWEL-SPLIT CORRECTION ----
        # A real diacritic split should never legitimately resolve to a
        # standalone vowel (A/EI/OU) - those never take diacritics. If
        # the whole-crop fallback confidently says this IS a vowel, the
        # "diacritic" piece that got split off was almost certainly part
        # of the vowel's own natural stroke gap, not a real mark. Prefer
        # the whole-crop reading whenever it's at least as confident as
        # the split-based base reading, so this only overrides genuinely
        # weaker/incorrect split results rather than every vowel-shaped
        # coincidence.
        if whole_base_name in ('A', 'EI', 'OU') and whole_confidence >= base_confidence:
            if debug:
                print(f"  [debug] VOWEL-SPLIT CORRECTION: split gave "
                      f"'{svm_raw_dia_prediction}' diacritic on base "
                      f"'{predicted_base_name}', but whole-crop reading "
                      f"'{whole_base_name}' ({whole_confidence:.2f}) is a standalone "
                      f"vowel -> using whole-crop result instead")
            predicted_base_name = whole_base_name
            predicted_dia_name = 'None'
            position = 'None'
            base_confidence = whole_confidence
            dia_confidence = 0.0

        if debug:
            print(f"  [debug] dw={dw}, dh={dh}, area={dw * dh}, "
                  f"rotated_aspect_ratio={rotated_aspect_ratio:.2f} "
                  f"(threshold={bar_rotated_aspect_threshold}), solidity={solidity:.2f}, "
                  f"too_small={is_too_small_for_shape_analysis}, "
                  f"split_svm_pred='{svm_raw_dia_prediction}', "
                  f"whole_crop_pred='{whole_base_name}' ({whole_confidence:.2f}), "
                  f"position='{position}'")

    final_output_text = predicted_base_name
    if predicted_base_name not in ['A', 'EI', 'OU']:
        dia_clean = predicted_dia_name.lower()
        base_root = predicted_base_name[:-1]
        if 'cross' in dia_clean or 'x' in dia_clean:
            final_output_text = base_root
        elif position == 'Above' and 'bar' in dia_clean:
            final_output_text = base_root + 'e'
        elif position == 'Above' and 'dot' in dia_clean:
            final_output_text = base_root + 'i'
        elif position == 'Below' and 'dot' in dia_clean:
            final_output_text = base_root + 'o'
        elif position == 'Below' and 'bar' in dia_clean:
            final_output_text = base_root + 'u'

    confidence = base_confidence
    if predicted_dia_name != 'None':
        confidence = min(base_confidence, dia_confidence)
    return predicted_base_name, predicted_dia_name, final_output_text, confidence


def preprocess_and_predict(image_bytes, session_id, base_model, dia_model, base_classes, dia_classes):
    if base_model is None or dia_model is None:
        raise ValueError('Baybayin base and diacritic models not loaded')

    try:
        image = decode_grayscale_exif_corrected(image_bytes)
    except Exception:
        image = None
    if image is None:
        return 'Error', 0.0, [], {'width': 0, 'height': 0}

    image_height, image_width = image.shape[:2]

    blur_size = min(101, max(15, (min(image.shape) // 3) | 1))
    background = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
    normalized = cv2.divide(image, background, scale=255)
    binary = preprocess_image(normalized)
    boxes = tighten_boxes(binary, segment_glyphs(binary), pad=0)
    if not boxes:
        return 'No characters detected', 0.0, [], {'width': image_width, 'height': image_height}
    word_groups = group_boxes_into_words(boxes)

    session_dir = os.path.join(TEMP_ROOT, f'session_{session_id}')
    os.makedirs(session_dir, exist_ok=True)
    results = []
    output_parts = []
    noise_confidence_threshold = 0.20

    crop_index = 0
    for word_group in word_groups:
        word_parts = []
        for x0, y0, x1, y1 in word_group:
            # Tight crop only - NO resize yet. Track the offset so we
            # can map back to absolute image coordinates afterward.
            crop_offset, crop = tight_crop_glyph_with_offset(binary[y0:y1, x0:x1])
            if crop is None:
                continue
            # Absolute top-left of `crop` in the ORIGINAL image.
            tight_abs_x = x0 + crop_offset[0]
            tight_abs_y = y0 + crop_offset[1]

            # Safety net: segment_glyphs' merge_kernel can occasionally
            # pull two nearby letters into one box. Re-check here and
            # split back apart into individual glyphs if that happened,
            # instead of silently classifying only one of them. Each
            # split-out piece carries its own local_box relative to `crop`.
            single_glyph_crops = split_into_single_glyphs(crop)

            for local_box, glyph_crop in single_glyph_crops:
                base_name, dia_name, final_text, confidence = classify_glyph(
                    glyph_crop, base_model, dia_model, base_classes, dia_classes
                )
                if base_name == 'Unknown':
                    continue
                if confidence < noise_confidence_threshold:
                    continue

                # Translate local_box (relative to `crop`) all the way
                # back to absolute pixel coordinates in the original image.
                lx0, ly0, lx1, ly1 = local_box
                abs_x0 = tight_abs_x + lx0
                abs_y0 = tight_abs_y + ly0
                abs_x1 = tight_abs_x + lx1
                abs_y1 = tight_abs_y + ly1

                crop_path = os.path.join(
                    session_dir, f'{final_text}_{crop_index}_{uuid.uuid4().hex[:6]}.jpg'
                )
                cv2.imwrite(crop_path, glyph_crop)
                results.append({
                    'char': final_text,
                    'base': base_name,
                    'diacritic': dia_name,
                    'confidence': round(confidence * 100, 2),
                    'is_eligible': True,
                    'temp_path': crop_path,
                    'bbox': {
                        'x0': int(abs_x0),
                        'y0': int(abs_y0),
                        'x1': int(abs_x1),
                        'y1': int(abs_y1),
                    },
                })
                word_parts.append(final_text)
                crop_index += 1
        if word_parts:
            output_parts.append(''.join(word_parts))

    average_confidence = (
        round(float(np.mean([item['confidence'] for item in results])), 2)
        if results else 0.0
    )
    image_dims = {'width': image_width, 'height': image_height}
    return ' '.join(output_parts).strip().capitalize(), average_confidence, results, image_dims