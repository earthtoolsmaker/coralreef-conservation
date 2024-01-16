import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

LABEL_TO_CLASS_MAPPING: dict[str, int] = {"soft_coral": 1, "hard_coral": 0, "other": 2}
CLASS_TO_LABEL_MAPPING: dict[int, str] = {
    v: k for k, v in LABEL_TO_CLASS_MAPPING.items()
}


def iou(bmask1, bmask2):
    """Returns the iou metric (float between 0 and 1) of the Intersection over
    Union of bmask1 and bmask2."""
    intersection_mask = bmask1 & bmask2
    union_mask = bmask1 | bmask2
    N = np.sum(intersection_mask)
    D = np.sum(union_mask)
    epsilon = 1
    return N / (D + epsilon)  # Avoid division by zero


def dice(bmask1, bmask2):
    """Returns the dice coefficient (float between 0 and 1)  of bmask1 and
    bmask2."""
    intersection_mask = bmask1 & bmask2
    N = 2 * np.sum(intersection_mask)
    D = np.sum(bmask1) + np.sum(bmask2)
    epsilon = 1
    return N / (D + epsilon)  # Avoid divide by zero


def confusion_matrix_to_precision_recall_f1(
    confusion_matrix: np.ndarray,
    class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING,
) -> dict:
    num_classes = confusion_matrix.shape[0]
    assert num_classes == 3, "should be soft corals, hard corals and others."

    metrics = ["precision", "recall", "f1"]
    result = {m: {} for m in metrics}

    for c in range(0, num_classes):
        tp = confusion_matrix[c, c]
        fp = np.sum([confusion_matrix[i, c] for i in set(range(0, 3)).difference({c})])
        fn = np.sum([confusion_matrix[c, i] for i in set(range(0, 3)).difference({c})])
        epsilon = 0.0001
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        result["precision"][class_to_label_mapping.get(c)] = precision
        result["recall"][class_to_label_mapping.get(c)] = recall
        result["f1"][class_to_label_mapping.get(c)] = f1

    # Computing the means
    for m in metrics:
        result[m]["mean"] = (
            np.sum(
                [result[m][class_to_label_mapping.get(i)] for i in range(num_classes)]
            )
            / num_classes
        )

    return result


def confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    n_classes: int = 3,
) -> np.ndarray:
    cf_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for a, p in zip(y_true, y_pred):
        cf_matrix[a][p] += 1
    return cf_matrix


def plot_confusion_matrix(
    cf_matrix: np.ndarray,
    normalize=False,
    class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING,
    save_path: Optional[Path] = None,
) -> None:
    """Given a confusion matrix `cf_matrix`, it displays it as a pyplot
    graph."""
    cm = cf_matrix.copy()
    ax = None
    if normalize:
        epsilon = 0.0001
        cm = cm.astype(np.float64) / (cm.sum(axis=1)[:, np.newaxis] + epsilon)
        ax = sns.heatmap(cm, annot=True, fmt=".2%", cmap="Blues")
    else:
        ax = sns.heatmap(cm, annot=True, cmap="Blues")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    ## Ticket labels - List must be in alphabetical order
    labels = sorted([class_to_label_mapping[i] for i in range(3)])
    labels = [class_to_label_mapping[i] for i in range(3)]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    # Save the confusion matrix
    if save_path:
        figname = "confusion_matrix_normalized" if normalize else "confusion_matrix"
        fp = save_path / f"{figname}.png"
        logging.info(f"saving confusion matrix in {fp}")
        plt.savefig(str(fp))
    else:
        plt.show()
    plt.close()


def confusion_matrix_to_dice_scores(
    confusion_matrix: np.ndarray, class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING
) -> dict:
    """Returns the dice scores as a dictionnary of labels to scores given the
    `conf_matrix`."""
    num_classes = confusion_matrix.shape[0]
    dice_scores = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives
        tp = confusion_matrix[i, i]
        # False positives - sum of column for class i, minus TP
        fp = np.sum(confusion_matrix[:, i]) - tp
        # False negatives - sum of row for class i, minus TP
        fn = np.sum(confusion_matrix[i, :]) - tp

        # IoU for class i
        dice = 2 * tp / (2 * tp + fp + fn)
        dice_scores[i] = dice

    return {
        **{class_to_label_mapping.get(i): dice_scores[i] for i in range(num_classes)},
        "mean": np.sum(dice_scores) / num_classes,
    }


def confusion_matrix_to_iou_scores(
    confusion_matrix: np.ndarray, class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING
) -> dict:
    """Returns the IoU scores as a dictionnary of labels to scores given the
    `conf_matrix`."""
    num_classes = confusion_matrix.shape[0]
    iou_scores = np.zeros(num_classes)

    for i in range(num_classes):
        # True positives
        tp = confusion_matrix[i, i]
        # False positives - sum of column for class i, minus TP
        fp = np.sum(confusion_matrix[:, i]) - tp
        # False negatives - sum of row for class i, minus TP
        fn = np.sum(confusion_matrix[i, :]) - tp

        # IoU for class i
        iou = tp / (tp + fp + fn)
        iou_scores[i] = iou

    return {
        **{class_to_label_mapping.get(i): iou_scores[i] for i in range(num_classes)},
        "mean": np.sum(iou_scores) / num_classes,
    }


def confusion_matrix_to_evaluation_metrics(
    confusion_matrix: np.ndarray,
    class_to_label_mapping: dict = CLASS_TO_LABEL_MAPPING,
) -> dict:
    result_iou_scores = confusion_matrix_to_iou_scores(
        confusion_matrix=confusion_matrix,
        class_to_label_mapping=class_to_label_mapping,
    )

    result_dice_scores = confusion_matrix_to_dice_scores(
        confusion_matrix=confusion_matrix,
        class_to_label_mapping=class_to_label_mapping,
    )

    result_precision_recall_f1 = confusion_matrix_to_precision_recall_f1(
        confusion_matrix=confusion_matrix,
        class_to_label_mapping=class_to_label_mapping,
    )

    return {
        "iou": {**result_iou_scores},
        "dice": {**result_dice_scores},
        **result_precision_recall_f1,
    }


def accumulate_confusion_matrices(
    cf_matrices: list[np.ndarray], n_classes=3
) -> np.ndarray:
    cf_matrix = np.zeros((n_classes, n_classes))
    if not cf_matrices:
        return cf_matrix
    else:
        # Accumulate the confusion matrices
        for cm in cf_matrices:
            cf_matrix += cm
        return cf_matrix
