import numpy
import sklearn

def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0]+gt_box[2], pred_box[0]+pred_box[2]), min(gt_box[1]+gt_box[3], pred_box[1]+pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection
    
    iou = intersection / union

    return iou, intersection, union

def convert_scores_to_labels(pred_scores, threshold):
    y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]   
    return y_pred

def calculate_confusion_precision_recall(y_pred, y_true):
    r = numpy.flip(sklearn.metrics.confusion_matrix(y_true, y_pred))
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
    return r, precision, recall

def calc_precision_recall_numpy(true_values, predictions):
    # true_values = np.array([[1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0]])
    # predictions = np.array([[1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]])

    TP = ((predictions == 1) & (true_values == 1)).sum()
    FP = ((predictions == 1) & (true_values == 0)).sum()
    FN = ((predictions == 0) & (true_values == 1)).sum()
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    return precision, recall

def precision_recall_curve(y_true, pred_scores, thresholds):
    precisions = []
    recalls = []
    for threshold in thresholds:
        # Convert prediction scores to class labels
        y_pred = ["positive" if score >= threshold else "negative" for score in pred_scores]
        # Calculate precision and recall
        precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_pred, pos_label="positive")
        precisions.append(precision)
        recalls.append(recall)
    return precisions, recalls

def f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def average_precision_for_loop(precisions, recalls):
    precisions.append(1)
    recalls.append(0)
    ap = 0
    for i in range(len(precisions)):
        ap += precisions[i] * (recalls[i] - recalls[i-1])
    return ap

def average_precision_numpy(precisions, recalls):
    precisions.append(1)
    recalls.append(0)
    ap = numpy.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
    return ap

if __name__=="__main__":
    pred_scores = [0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.4, 0.2, 0.4, 0.3]
    y_true = ["positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive", "negative", "positive"]
    thresholds=numpy.arange(start=0.2, stop=0.7, step=0.05)
    precisions, recalls = precision_recall_curve(y_true, pred_scores, thresholds)
    precisions = numpy.array(precisions)
    recalls = numpy.array(recalls)