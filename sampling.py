import torch


def get_sample_probabilities(gt: torch.Tensor, offset: float = 0.05, eps: float = 1e-9) -> torch.Tensor:
    """
    find sampling weights for input samples according to the multilabel ground truth data
    :param gt: binary matrix of ground truth labels. every row corresponds to a sample, every column correspond to label
    :param offset: small float to compensate for samples that have no labels annotated
    :param eps: small float to compensate after the computation
    """
    n_classes = gt.size(-1)
    #sample_weights = np.linalg.pinv(gt.T + offset) @ np.ones((n_classes,))
    sample_weights = torch.linalg.pinv(gt.t() + offset) @ torch.ones(n_classes, dtype=gt.dtype)
    sample_weights[sample_weights < eps] = eps
    sample_weights = sample_weights / torch.sum(sample_weights)
    return sample_weights


def get_label_weighted_sample_probabilities(gt: torch.Tensor, offset: float = 100.0) -> torch.Tensor:
    n_samples, n_cls = gt.shape
    label_occurrences = torch.sum(gt, dim=0)

    class_weights = 1 / (label_occurrences + offset)
    p_cls = class_weights / torch.sum(class_weights)

    sample_weights = torch.zeros(n_samples, dtype=p_cls.dtype)

    for k in range(n_samples):
        for c in torch.where(gt[k] > 0)[0]:
            sample_weights[k] += p_cls[c] / label_occurrences[c]

    return sample_weights
