from md_emg.metrics import compute_metrics


def test_known_confusion_matrix():
    # TN=2, FP=1, FN=1, TP=2
    m=compute_metrics([0,0,0,1,1,1],[0,0,1,0,1,1])
    assert abs(m["sensitivity"]-2/3)<1e-12
    assert abs(m["specificity"]-2/3)<1e-12
    assert abs(m["balanced_accuracy"]-2/3)<1e-12
