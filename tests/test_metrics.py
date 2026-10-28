import numpy as np
import torch
from sklearn.metrics import f1_score

from diabetes_ml.common import evaluate, fast_macro_f1, decision_predict, tune_offsets, paired_bootstrap
from diabetes_ml.neural import classification_loss


def test_one_vs_rest_specificity_and_undefined_predictions():
    # Confusion [[2,0,0],[1,0,1],[0,0,2]]; no prediabetes predictions.
    y = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 0, 0, 2, 2, 2])
    p = np.eye(3)[predicted]
    metrics, classes, cm, pred = evaluate(y, p)
    np.testing.assert_array_equal(cm, [[2, 0, 0], [1, 0, 1], [0, 0, 2]])
    np.testing.assert_allclose(classes.specificity, [.75, 1., .75])
    assert classes.loc[1, "precision"] == classes.loc[1, "recall"] == 0
    assert metrics["balanced_accuracy"] == 2 / 3
    assert fast_macro_f1(y, predicted) == f1_score(y, predicted, average="macro")


def test_focal_loss_uses_unweighted_pt_and_correct_alpha():
    logits = torch.tensor([[1., 2., -1.], [2., .5, 1.]], requires_grad=True)
    y = torch.tensor([1, 0])
    alpha = torch.tensor([.5, 3., 2.])
    probabilities = torch.softmax(logits, dim=1)[torch.arange(2), y]
    expected = (-torch.log(probabilities) * (1 - probabilities) ** 2 * alpha[y]).mean()
    actual = classification_loss(logits, y, alpha, gamma=2.)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(classification_loss(logits.detach(), y, gamma=0), torch.nn.functional.cross_entropy(logits.detach(), y))


def test_decision_tuning_changes_labels_without_changing_raw_scores():
    p = np.array([[.8, .1, .1], [.5, .3, .2], [.3, .1, .6]])
    copy = p.copy()
    y = np.array([0, 1, 2])
    offsets, score = tune_offsets(y, p)
    np.testing.assert_array_equal(p, copy)
    assert score >= fast_macro_f1(y, p.argmax(axis=1))
    np.testing.assert_array_equal(decision_predict(p, offsets), y)


def test_paired_bootstrap_preserves_perfect_pairing():
    y = np.array([0, 1, 2] * 20)
    baseline = np.array([0, 0, 2] * 20)
    result = paired_bootstrap(y, baseline, baseline, n=50)
    assert result["macro_f1_delta_95_ci"] == [0., 0.]
    better = paired_bootstrap(y, baseline, y, n=50)
    assert better["macro_f1_delta_95_ci"][0] > 0
