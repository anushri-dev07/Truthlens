def test_metrics_confusion():
    from truthlens.eval.metrics import (
        confusion_matrix,
        false_positive_rate,
        precision_recall_f1,
    )

    yt = [1, 1, 0, 0]
    yp = [0.9, 0.4, 0.8, 0.1]
    c = confusion_matrix(yt, yp, 0.5)
    assert c == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    prf = precision_recall_f1(yt, yp, 0.5)
    assert prf["precision"] == 0.5
    assert prf["recall"] == 0.5
    assert false_positive_rate(yt, yp, 0.5) == 0.5


def test_best_threshold():
    from truthlens.eval.metrics import best_threshold

    yt = [1, 1, 1, 0, 0, 0]
    yp = [0.9, 0.8, 0.7, 0.2, 0.3, 0.1]
    res = best_threshold(yt, yp)
    assert 0.3 <= res["threshold"] <= 0.7
    assert res["youden_j"] == 1.0


def test_evaluate_predictions_report():
    from truthlens.eval.harness import evaluate_predictions

    yt = [1, 1, 0, 0]
    yp = [0.9, 0.8, 0.1, 0.2]
    report = evaluate_predictions(yt, yp)
    assert report["accuracy"] == 1.0
    assert report["auc"] == 1.0
    assert "prf@0.5" in report


def test_read_manifest_csv(tmp_path):
    from truthlens.eval.harness import read_manifest

    p = tmp_path / "m.csv"
    p.write_text("path,label\na.mp4,1\nb.mp4,0\n", encoding="utf-8")
    items = read_manifest(p)
    assert items == [("a.mp4", 1), ("b.mp4", 0)]


def test_evaluate_videos_with_decider():
    from truthlens.eval.harness import evaluate_videos

    items = [("a.mp4", 1), ("b.mp4", 0)]
    preds = {"a.mp4": 0.9, "b.mp4": 0.1}
    report = evaluate_videos(
        items,
        predictor=lambda p: preds[p],
        decider=lambda p: "likely_ai" if p >= 0.5 else "likely_real",
    )
    assert report["coverage"] == 1.0
    assert report["selective_accuracy"] == 1.0
