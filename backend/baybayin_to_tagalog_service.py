import os
import uuid

import cv2
import numpy as np
from skimage.feature import hog

TEMP_ROOT = 'temp_crops'


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


def classify_glyph(crop_bin, base_model, dia_model, base_classes, dia_classes,
                   bar_aspect_threshold=1.4, solidity_threshold=0.80):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_bin)
    predicted_base_name = 'Unknown'
    predicted_dia_name = 'None'
    position = 'None'

    if num_labels <= 1:
        return predicted_base_name, predicted_dia_name, '', 0.0

    if num_labels == 2:
        base_mask = (labels == 1).astype(np.uint8) * 255
        base_norm = cv2.resize(base_mask, (56, 56)).astype(np.float32) / 255.0
        hog_base = hog(
            base_norm, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, visualize=False,
        ).reshape(1, -1)
        base_prediction, base_confidence = _predict_with_confidence(base_model, hog_base)
        predicted_base_name = _class_name(base_classes, base_prediction)
    else:
        valid_indices = sorted(
            range(1, num_labels),
            key=lambda index: stats[index, cv2.CC_STAT_AREA],
            reverse=True,
        )
        base_idx = valid_indices[0]
        dia_idx = valid_indices[1]
        base_mask = (labels == base_idx).astype(np.uint8) * 255
        dia_mask = (labels == dia_idx).astype(np.uint8) * 255
        dx, dy, dw, dh, _ = stats[dia_idx]
        if dw == 0 or dh == 0:
            return predicted_base_name, predicted_dia_name, '', 0.0
        dia_crop = dia_mask[dy:dy + dh, dx:dx + dw]

        dia_upscaled = cv2.resize(dia_crop, (40, 40), interpolation=cv2.INTER_NEAREST)
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

        base_norm = cv2.resize(base_mask, (56, 56)).astype(np.float32) / 255.0
        dia_norm = cv2.resize(dia_crop, (56, 56)).astype(np.float32) / 255.0
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
        position = 'Above' if centroids[dia_idx][1] < centroids[base_idx][1] else 'Below'

        if float(dw) / float(dh) >= bar_aspect_threshold:
            predicted_dia_name = 'Bar'
        elif solidity < solidity_threshold:
            available_classes = [str(value).lower() for value in dia_classes]
            if 'x' in available_classes:
                predicted_dia_name = 'X'
            elif 'cross' in available_classes:
                predicted_dia_name = 'Cross'
        else:
            predicted_dia_name = 'Dot'

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

    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return 'Error', 0.0, []

    binary = preprocess_image(image)
    boxes = tighten_boxes(binary, segment_glyphs(binary), pad=0)
    if not boxes:
        return 'No characters detected', 0.0, []
    word_groups = group_boxes_into_words(boxes)

    session_dir = os.path.join(TEMP_ROOT, f'session_{session_id}')
    os.makedirs(session_dir, exist_ok=True)
    results = []
    output_parts = []
    noise_confidence_threshold = 0.40

    crop_index = 0
    for word_group in word_groups:
        word_parts = []
        for x0, y0, x1, y1 in word_group:
            crop = binary[y0:y1, x0:x1]
            base_name, dia_name, final_text, confidence = classify_glyph(
                crop, base_model, dia_model, base_classes, dia_classes
            )
            if base_name == 'Unknown':
                continue
            if confidence < noise_confidence_threshold:
                continue
            crop_path = os.path.join(
                session_dir, f'{final_text}_{crop_index}_{uuid.uuid4().hex[:6]}.jpg'
            )
            cv2.imwrite(crop_path, crop)
            results.append({
                'char': final_text,
                'base': base_name,
                'diacritic': dia_name,
                'confidence': round(confidence * 100, 2),
                'is_eligible': True,
                'temp_path': crop_path,
            })
            word_parts.append(final_text)
            crop_index += 1
        if word_parts:
            output_parts.append(''.join(word_parts))

    average_confidence = (
        round(float(np.mean([item['confidence'] for item in results])), 2)
        if results else 0.0
    )
    return ' '.join(output_parts).strip().capitalize(), average_confidence, results
