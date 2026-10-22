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
    def __init__(self, model_path: str, input_size: tuple[int, int]):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.emb_name = [o.name for o in self.sess.get_outputs() if "logits" not in o.name][0]
        self.size = input_size

    def embed(self, img_rgb: np.ndarray) -> np.ndarray:
        import cv2

        img = cv2.resize(img_rgb, self.size, interpolation=cv2.INTER_CUBIC)
        x = img.astype(np.float32) / 255.0
        x = (x - np.array([0.48145466, 0.4578275, 0.40821073], np.float32)) / np.array(
            [0.26862954, 0.26130258, 0.27577711], np.float32
        )
        x = x.transpose(2, 0, 1)[None, ...]
        out = self.sess.run(None, {self.input_name: x})
        for o in self.sess.get_outputs():
            if o.name == self.emb_name:
                return out[self.sess.get_outputs().index(o)].reshape(-1)
        return out[1].reshape(-1)


def extract(model_path, size, files, frames, root):
    import cv2

    from truthlens.data.faces import extract_faces

    emb = Embedder(model_path, size)
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
    parser.add_argument("--size", type=int, default=384)
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
    X, groups, y = extract(str(emb_model), (args.size, args.size), files, args.frames, root)
    np.savez(out / "embeddings.npz", X=X, groups=groups, y=y)
    print("\nX", X.shape, "pos", int(y.sum()))

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
    )
    n_splits = min(5, len(set(groups)))
    oof = cross_val_predict(
        clf, X, y, groups=groups, cv=GroupKFold(n_splits=n_splits), method="predict_proba"
    )[:, 1]
    print(f"frame-level OOF AUC: {_auc_roc(list(y), list(oof)):.3f}")
    vid = {}
    for g, p, t in zip(groups, oof, y):
        vid.setdefault((int(g), int(t)), []).append(p)
    vy = [k[1] for k in vid]
    vp = [float(np.mean(v)) for v in vid.values()]
    print(f"video-level OOF AUC: {_auc_roc(vy, vp):.3f}")
    clf.fit(X, y)
    import joblib

    joblib.dump(clf, out / "probe.pkl")
    (out / "result.json").write_text(
        json.dumps(
            {
                "frame_auc": _auc_roc(list(y), list(oof)),
                "video_auc": _auc_roc(vy, vp),
                "n_videos": len(files),
                "frames": len(y),
                "size": args.size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
