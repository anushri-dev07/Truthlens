from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from truthlens.eval.metrics import _auc_roc


def build_embedding_model(src: str, dst: str) -> str:
    import onnx

    model = onnx.load(src)
    existing = {o.name for o in model.graph.output}
    target = None
    for node in model.graph.node:
        if node.op_type == "Gather" and node.output and node.output[0] == "/Gather_output_0":
            target = node.output[0]
            break
    if target is None:
        for node in model.graph.node:
            if node.op_type == "Gather":
                target = node.output[0]
    if target is None:
        raise RuntimeError("no Gather node found")
    if target not in existing:
        model.graph.output.append(onnx.ValueInfoProto(name=target))
    onnx.save(model, dst)
    return target


class Embedder:
    def __init__(
        self,
        model_path: str,
        input_size: tuple[int, int],
        preprocessor_config: str | Path | None = None,
    ):
        import onnxruntime as ort

        from truthlens.models.frame_clf import ImagePreprocessor

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.emb_name = [o.name for o in self.sess.get_outputs() if "logits" not in o.name][0]
        self.preprocessor = ImagePreprocessor(
            preprocessor_config, fallback_size=input_size
        )

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        x = self.preprocessor(img_rgb)
        out = self.sess.run(None, {self.input_name: x})
        for o in self.sess.get_outputs():
            if o.name == self.emb_name:
                return out[self.sess.get_outputs().index(o)].reshape(-1)
        return out[1].reshape(-1)


def extract(model_path, size, files, frames, root, preprocessor_config=None):
    import cv2

    from truthlens.data.faces import extract_faces

    emb = Embedder(model_path, size, preprocessor_config)
    X, groups, ys = [], [], []
    for idx, (path, label) in enumerate(files):
        cap = cv2.VideoCapture(path)
        n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        step = max(1, int(n // (frames + 1))) if n else 1
        got = 0
        f = 0
        while got < frames:
            ok, frame = cap.read()
            if not ok:
                break
            if f % step == 0:
                faces = extract_faces(frame)
                crop = frame
                if faces:
                    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
                    crop = frame[y : y + h, x : x + w]
                vec = emb.embed(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                X.append(vec)
                groups.append(idx)
                ys.append(label)
                got += 1
            f += 1
        cap.release()
        print(f"  {idx + 1}/{len(files)} {os.path.basename(path)}", end="\r")
    return np.array(X), np.array(groups), np.array(ys)


def main():
    parser = argparse.ArgumentParser(description="Linear probe on frozen ViT embeddings")
    parser.add_argument("--model", default="artifacts/models/community_forensics/model.onnx")
    parser.add_argument("--out", default="artifacts/emb_probe")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--root", default="data/eval")
    parser.add_argument(
        "--size",
        type=int,
        default=384,
        help="fallback input size when the model has no preprocessor_config.json",
    )
    parser.add_argument("--test-size", type=float, default=0.3,
                        help="fraction of VIDEYS held out for testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reuse-embeddings", action="store_true",
                        help="reuse cached embeddings.npz instead of re-extracting")
    parser.add_argument("--fit-all", action="store_true",
                        help="after reporting held-out metrics, refit the shipped probe on ALL videos")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb_model = out / "embed_model.onnx"
    if not emb_model.exists():
        build_embedding_model(args.model, str(emb_model))
    root = Path(args.root)
    files = [(p, 0) for p in sorted(glob.glob(str(root / "real" / "*.mp4")))]
    files += [(p, 1) for p in sorted(glob.glob(str(root / "ai" / "*.mp4")))]
    print(f"videos: {len(files)}")
    cache_npz = out / "embeddings.npz"
    preprocessor_path = Path(args.model).parent / "preprocessor_config.json"
    preprocessor_config = (
        json.loads(preprocessor_path.read_text(encoding="utf-8"))
        if preprocessor_path.exists()
        else {"size": {"width": args.size, "height": args.size}}
    )
    (out / "preprocessor_config.json").write_text(
        json.dumps(preprocessor_config, indent=2), encoding="utf-8"
    )
    cache_metadata = {
        "model": str(Path(args.model).resolve()),
        "model_size_bytes": Path(args.model).stat().st_size
        if Path(args.model).exists()
        else None,
        "preprocessor_config": preprocessor_config,
        "fallback_input_size": args.size,
        "frames_per_video": args.frames,
        "videos": [(str(Path(path).resolve()), label) for path, label in files],
    }
    if args.reuse_embeddings and cache_npz.exists():
        with np.load(cache_npz) as d:
            if "metadata" not in d.files:
                raise ValueError(
                    "Embedding cache has no preprocessing metadata; rerun without "
                    "--reuse-embeddings to rebuild it."
                )
            cached_metadata = json.loads(str(d["metadata"].item()))
            if cached_metadata != cache_metadata:
                raise ValueError(
                    "Embedding cache settings do not match this run; rerun without "
                    "--reuse-embeddings to rebuild it."
                )
            X, groups, y = d["X"], d["groups"], d["y"]
        print(f"reusing cached embeddings: {X.shape}")
    else:
        X, groups, y = extract(
            str(emb_model),
            (args.size, args.size),
            files,
            args.frames,
            root,
            preprocessor_config=out / "preprocessor_config.json",
        )
        np.savez(
            cache_npz,
            X=X,
            groups=groups,
            y=y,
            metadata=json.dumps(cache_metadata, sort_keys=True),
        )
    print("\nX", X.shape, "pos", int(y.sum()))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    def make_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        )

    def video_scores(prob, groups, y):
        """Average per-frame probabilities into one score per video."""
        acc: dict = {}
        for g, p, t in zip(groups, prob, y):
            acc.setdefault((int(g), int(t)), []).append(p)
        labels = [k[1] for k in acc]
        vals = [float(np.mean(v)) for v in acc.values()]
        return labels, vals

    # ---- Honest split: a video is never in both train and test -------------
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    frame_idx = np.arange(len(y))
    train_frame_idx, test_frame_idx = next(splitter.split(frame_idx, y, groups=groups))
    train_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_frame_idx] = True
    test_mask[test_frame_idx] = True

    train_vid_set = set(int(g) for g in groups[train_mask])
    test_vid_set = set(int(g) for g in groups[test_mask])
    overlap = train_vid_set & test_vid_set
    if overlap:
        raise RuntimeError(f"video overlap between train and test: {sorted(overlap)}")
    n_train_v = len(train_vid_set)
    n_test_v = len(test_vid_set)
    print(
        f"\nvideo-disjoint split: {n_train_v} train videos / {n_test_v} test videos "
        f"(frames {int(train_mask.sum())} / {int(test_mask.sum())})"
    )

    # Cross-validated estimate over the training videos only.
    clf_cv = make_clf()
    n_splits = min(5, max(2, n_train_v))
    oof = cross_val_predict(
        clf_cv,
        X[train_mask],
        y[train_mask],
        groups=groups[train_mask],
        cv=GroupKFold(n_splits=n_splits),
        method="predict_proba",
    )[:, 1]
    cv_frame_auc = _auc_roc(list(y[train_mask]), list(oof))
    cv_vy, cv_vp = video_scores(oof, groups[train_mask], y[train_mask])
    cv_video_auc = _auc_roc(cv_vy, cv_vp)
    print(f"train CV frame-level AUC : {cv_frame_auc:.3f}")
    print(f"train CV video-level AUC : {cv_video_auc:.3f}")

    # Fit on the training videos only, then evaluate once on the held-out videos.
    clf = make_clf()
    clf.fit(X[train_mask], y[train_mask])
    test_prob = clf.predict_proba(X[test_mask])[:, 1]
    test_frame_auc = _auc_roc(list(y[test_mask]), list(test_prob))
    ty, tp = video_scores(test_prob, groups[test_mask], y[test_mask])
    test_video_auc = _auc_roc(ty, tp)
    print(f"HELD-OUT frame-level AUC : {test_frame_auc:.3f}")
    print(f"HELD-OUT video-level AUC : {test_video_auc:.3f}")

    in_prob = clf.predict_proba(X[train_mask])[:, 1]
    ins_frame_auc = _auc_roc(list(y[train_mask]), list(in_prob))
    iy, ip = video_scores(in_prob, groups[train_mask], y[train_mask])
    ins_video_auc = _auc_roc(iy, ip)
    print(f"in-sample (train) video AUC : {ins_video_auc:.3f}  <- optimistic, not a metric")

    import joblib

    train_video_paths = [files[i][0] for i in sorted(train_vid_set)]
    test_video_paths = [files[i][0] for i in sorted(test_vid_set)]

    if args.fit_all:
        # Deployment model: refit on every labelled video. The headline metrics
        # above are still the held-out ones; refitting only changes how much
        # data the shipped weights have seen, not the reported generalisation.
        clf = make_clf()
        clf.fit(X, y)
        print("\n--fit-all: refitting the shipped probe on all labelled videos")
    joblib.dump(clf, out / "probe.pkl")
    (out / "result.json").write_text(
        json.dumps(
            {
                "held_out_frame_auc": round(test_frame_auc, 4),
                "held_out_video_auc": round(test_video_auc, 4),
                "cv_frame_auc": round(cv_frame_auc, 4),
                "cv_video_auc": round(cv_video_auc, 4),
                "in_sample_video_auc": round(ins_video_auc, 4),
                "n_videos": len(files),
                "n_train_videos": n_train_v,
                "n_test_videos": n_test_v,
                "frames": len(y),
                "fallback_input_size": args.size,
                "preprocessor_config_source": str(preprocessor_path.resolve()),
                "preprocessing": preprocessor_config,
                "seed": args.seed,
                "test_size": args.test_size,
                "fit_all": bool(args.fit_all),
                "note": (
                    "held_out_* metrics come from videos never seen in training. "
                    "in_sample_video_auc is optimistic and must never be quoted "
                    "as the detector's accuracy."
                ),
            },
            indent=2,
        )
    )
    (out / "split.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "test_size": args.test_size,
                "train_videos": train_video_paths,
                "test_videos": test_video_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved probe -> {out / 'probe.pkl'}")


if __name__ == "__main__":
    main()
