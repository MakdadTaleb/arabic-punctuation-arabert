from sklearn.metrics import classification_report


def generate_classification_report(true_labels, pred_labels, label_names):
    return classification_report(
        true_labels,
        pred_labels,
        target_names=label_names,
        digits=4
    )