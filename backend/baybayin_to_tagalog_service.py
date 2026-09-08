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
    """
    points = cv2.findNonZero(crop_bin)
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    return crop_bin[y:y + height, x:x + width]


def tight_crop_glyph_with_offset(crop_bin):
    """
    Same as tight_crop_glyph, but also returns the (x, y) offset of the
    crop's top-left corner relative to crop_bin's own origin.
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

    boxes.sort(key=lambda b: (b[0], b[1]))
    return boxes


def tighten_to_ink(binary_img, box):
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


def _distance_transform_gap(labels, mask_label_ids, candidate_label_id):
    """Approximate nearest-pixel distance from `candidate_label_id`'s ink
    to the union of components in `mask_label_ids`, via a distance
    transform (no scipy dependency needed)."""
    mask = np.isin(labels, mask_label_ids).astype(np.uint8) * 255
    inverted = np.where(mask > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inverted, cv2.DIST_L2, 5)
    return float(dist[labels == candidate_label_id].min())


def _estimate_stroke_thickness(binary_img):
    """
    Median-based (not 90th percentile) distance-transform thickness
    estimate - more robust against self-intersecting loop crossings
    that spike thickness locally. Used only by the PEN preset's
    adaptive gap threshold below.
    """
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    nonzero_dists = dist[binary_img > 0]
    if nonzero_dists.size == 0:
        return 1.0
    typical_half_width = float(np.percentile(nonzero_dists, 50))
    return max(typical_half_width * 2.0, 1.0)


# ---- MARKER / PENTEL PEN: fixed gap threshold ----
# Confirmed working as-is for thicker marker/felt-tip strokes. A break
# in a single stroke caused by thresholding/anti-aliasing sits at a
# near-constant tiny gap (~2px) independent of glyph size - a genuine
# diacritic pen-lift measures several times that.
MARKER_SAME_STROKE_GAP_THRESHOLD = 1.9

# ---- PEN: adaptive gap threshold ----
# Confirmed working for thin ballpoint/gel pen strokes, where a fixed
# pixel threshold is too rigid. Scales with the glyph's OWN measured
# stroke thickness instead.
PEN_GAP_THICKNESS_MULTIPLIER = 1.5
PEN_MAX_GAP_THRESHOLD_PIXELS = 6.0


def separate_base_and_diacritic_fixed(labels, stats, num_labels,
                                       gap_threshold=MARKER_SAME_STROKE_GAP_THRESHOLD):
    """MARKER / PENTEL PEN preset: fixed pixel gap threshold."""
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

    return base_idx_list, dia_idx, gap_threshold


def separate_base_and_diacritic_adaptive(labels, stats, num_labels, native_crop,
                                          gap_thickness_multiplier=PEN_GAP_THICKNESS_MULTIPLIER,
                                          max_gap_threshold_pixels=PEN_MAX_GAP_THRESHOLD_PIXELS):
    """PEN preset: gap threshold scaled to the glyph's own stroke
    thickness, capped so a self-intersection spike can't blow it out."""
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    areas.sort(key=lambda t: t[1], reverse=True)

    base_idx_list = [areas[0][0]]
    remaining = [idx for idx, _ in areas[1:]]

    largest_mask = (labels == areas[0][0]).astype(np.uint8) * 255
    stroke_thickness = _estimate_stroke_thickness(largest_mask)
    gap_threshold = min(stroke_thickness * gap_thickness_multiplier, max_gap_threshold_pixels)

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

    return base_idx_list, dia_idx, gap_threshold


INPUT_TYPE_PRESETS = {
    'marker': {
        'separation_mode': 'fixed',
        'gap_threshold': MARKER_SAME_STROKE_GAP_THRESHOLD,
        'min_pixels_for_shape_analysis': 16,
        'bar_rotated_aspect_threshold': 2.2,
        'solidity_threshold': 0.80,
    },
    'pentel_pen': {
        'separation_mode': 'fixed',
        'gap_threshold': MARKER_SAME_STROKE_GAP_THRESHOLD,
        'min_pixels_for_shape_analysis': 16,
        'bar_rotated_aspect_threshold': 2.2,
        'solidity_threshold': 0.80,
    },
    'pen': {
        'separation_mode': 'adaptive',
        'gap_thickness_multiplier': PEN_GAP_THICKNESS_MULTIPLIER,
        'max_gap_threshold_pixels': PEN_MAX_GAP_THRESHOLD_PIXELS,
        'min_pixels_for_shape_analysis': 16,
        'bar_rotated_aspect_threshold': 2.2,
        'solidity_threshold': 0.80,
    },
}
DEFAULT_INPUT_TYPE = 'marker'


BAR_ROTATED_ASPECT_THRESHOLD = 2.2
MIN_PIXELS_FOR_SHAPE_ANALYSIS = 16


def classify_glyph(native_crop, base_model, dia_model, base_classes, dia_classes,
                   input_type=DEFAULT_INPUT_TYPE,
                   debug=False):
    """
    native_crop: tight-cropped binary glyph at ITS ORIGINAL resolution
    (NOT yet resized to 56x56).

    input_type: 'marker', 'pentel_pen', or 'pen' - selects which
    separate_base_and_diacritic implementation and threshold set to use,
    per INPUT_TYPE_PRESETS. Each preset reflects a version that has been
    tested and confirmed working for that specific ink type - marker and
    pentel_pen currently share settings (a fixed gap threshold);
    ballpoint/gel pen needs the adaptive, stroke-thickness-scaled gap
    threshold instead.
    """
    preset = INPUT_TYPE_PRESETS.get(input_type, INPUT_TYPE_PRESETS[DEFAULT_INPUT_TYPE])
    min_pixels_for_shape_analysis = preset['min_pixels_for_shape_analysis']
    bar_rotated_aspect_threshold = preset['bar_rotated_aspect_threshold']
    solidity_threshold = preset['solidity_threshold']

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(native_crop)
    predicted_base_name = 'Unknown'
    predicted_dia_name = 'None'
    position = 'None'
    base_confidence = 0.0
    dia_confidence = 0.0

    if num_labels <= 1:
        return predicted_base_name, predicted_dia_name, '', 0.0

    # ---- WHOLE-CROP FALLBACK CLASSIFICATION ----
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

    # ---- SEPARATION: dispatch to the preset's chosen implementation ----
    if preset['separation_mode'] == 'adaptive':
        base_idx_list, dia_idx, gap_threshold_used = separate_base_and_diacritic_adaptive(
            labels, stats, num_labels, native_crop,
            gap_thickness_multiplier=preset['gap_thickness_multiplier'],
            max_gap_threshold_pixels=preset['max_gap_threshold_pixels'],
        )
    else:
        base_idx_list, dia_idx, gap_threshold_used = separate_base_and_diacritic_fixed(
            labels, stats, num_labels,
            gap_threshold=preset['gap_threshold'],
        )

    if dia_idx is None:
        predicted_base_name = whole_base_name
        base_confidence = whole_confidence
        if debug:
            print(f"  [debug] input_type={input_type}, no diacritic split "
                  f"(gap_threshold_used={gap_threshold_used:.2f}); "
                  f"whole_crop_pred='{whole_base_name}' ({whole_confidence:.2f})")
    else:
        base_mask_full = np.isin(labels, base_idx_list).astype(np.uint8) * 255
        coords = cv2.findNonZero(base_mask_full)
        bx, by, bw, bh = cv2.boundingRect(coords)
        base_crop = base_mask_full[by:by + bh, bx:bx + bw]
        base_norm = cv2.resize(base_crop, (56, 56)).astype(np.float32) / 255.0

        dx, dy, dw, dh, _ = stats[dia_idx]
        if dw == 0 or dh == 0:
            predicted_base_name = whole_base_name
            base_confidence = whole_confidence
            return predicted_base_name, predicted_dia_name, predicted_base_name, base_confidence

        dia_mask_full = (labels == dia_idx).astype(np.uint8) * 255
        dia_crop = dia_mask_full[dy:dy + dh, dx:dx + dw]
        dia_padded = crop_and_pad_to_square(dia_crop)

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
        svm_raw_dia_prediction = predicted_dia_name

        base_centroid_y = by + (bh / 2.0)
        position = 'Above' if centroids[dia_idx][1] < base_centroid_y else 'Below'

        dia_points = cv2.findNonZero(dia_crop)
        if dia_points is not None:
            (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(dia_points)
            long_side = max(rect_w, rect_h)
            short_side = max(min(rect_w, rect_h), 1e-6)
            rotated_aspect_ratio = long_side / short_side
        else:
            rotated_aspect_ratio = float(dw) / float(dh)

        is_thin_bar = rotated_aspect_ratio >= bar_rotated_aspect_threshold
        is_too_small_for_shape_analysis = (dw * dh) < min_pixels_for_shape_analysis

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

        # ---- SPLIT-VS-WHOLE CORRECTION ----
        # Vowels ALWAYS get corrected if whole-crop is at least as
        # confident, since a real diacritic can never legitimately
        # attach to A/EI/OU. Non-vowels only get corrected when the
        # split-off piece was ALSO too small to trust geometrically -
        # catches stray stroke fragments (tails/serifs) without
        # overriding genuinely visible diacritics on consonants.
        force_for_vowel = whole_base_name in ('A', 'EI', 'OU') and whole_confidence >= base_confidence
        prefer_for_fragment = whole_confidence >= base_confidence and is_too_small_for_shape_analysis

        if force_for_vowel or prefer_for_fragment:
            if debug:
                reason = 'vowel' if force_for_vowel else 'small stray fragment'
                print(f"  [debug] SPLIT CORRECTION ({reason}): split gave "
                      f"'{svm_raw_dia_prediction}' diacritic on base "
                      f"'{predicted_base_name}' ({base_confidence:.2f}), but whole-crop "
                      f"reading '{whole_base_name}' ({whole_confidence:.2f}) is at least "
                      f"as confident -> using whole-crop result instead")
            predicted_base_name = whole_base_name
            predicted_dia_name = 'None'
            position = 'None'
            base_confidence = whole_confidence
            dia_confidence = 0.0

        if debug:
            print(f"  [debug] input_type={input_type}, dw={dw}, dh={dh}, area={dw * dh}, "
                  f"rotated_aspect_ratio={rotated_aspect_ratio:.2f} "
                  f"(threshold={bar_rotated_aspect_threshold}), solidity={solidity:.2f}, "
                  f"too_small={is_too_small_for_shape_analysis}, "
                  f"gap_threshold_used={gap_threshold_used:.2f}, "
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


def preprocess_and_predict(image_bytes, session_id, base_model, dia_model, base_classes, dia_classes,
                           input_type=DEFAULT_INPUT_TYPE):
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
            crop_offset, crop = tight_crop_glyph_with_offset(binary[y0:y1, x0:x1])
            if crop is None:
                continue
            tight_abs_x = x0 + crop_offset[0]
            tight_abs_y = y0 + crop_offset[1]

            single_glyph_crops = split_into_single_glyphs(crop)

            for local_box, glyph_crop in single_glyph_crops:
                base_name, dia_name, final_text, confidence = classify_glyph(
                    glyph_crop, base_model, dia_model, base_classes, dia_classes,
                    input_type=input_type,
                )
                if base_name == 'Unknown':
                    continue
                if confidence < noise_confidence_threshold:
                    continue

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