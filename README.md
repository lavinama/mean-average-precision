# mean-average-precision
Calculating mean Average Precision (mAP) only using numpy


Steps to calculate mAP for multiple IoU thresholds:
- For each given IoU threshold:
    - For each class calculate the AP:
        - Determine the IoU threshold to choose
        - Calculate the IoU for each image
        - Calculate the AP given the IoU threshold
            - Sort the predicted boxes in descending order (lowest scoring boxes first) for each image
            - Loop over the models’ score thresholds
                - Loop over images
                    - Find the first index of a box with a score greater than the threshold
                    - Remove the boxes with scores lower than the threshold
                    - Recalculate the number of false positives, true positives, false negatives
                - Calculate precision and recall for each image
            - Smooth the interpolated precision recall curve

#### `calc_precision_recall_numpy`
```python
def calc_precision_recall_numpy(true_values, predictions):
    # true_values = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
    # predictions = np.array([[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]])

    TP = ((predictions == 1) & (true_values == 1)).sum()
    FP = ((predictions == 1) & (true_values == 0)).sum()
    FN = ((predictions == 0) & (true_values == 1)).sum()
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    return precision, recall
```

#### `calc_precision_recall_curve_numpy`
```python
def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    for threshold in thresholds:
        # Convert prediction scores to class labels
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
        # Calculate precision and recall
        precision, recall = calc_precision_recall_numpy(true_values, predictions)
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls
```

#### `calculate_average_precision_numpy`
```python
def average_precision_numpy(precisions, recalls):
    precisions.append(1)
    recalls.append(0)
    ap = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return ap
```

#### `calculater_intersection_over_union`
```python
def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
    
    iou = intersection / union

    return iou, intersection, union
```