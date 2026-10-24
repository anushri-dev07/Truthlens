from types import SimpleNamespace

import numpy as np

from truthlens.models.frame_clf import FrameClassifier, ImagePreprocessor


def test_probe_training_and_runtime_preprocessing_match(monkeypatch):
    from scripts.train_probe import Embedder

    captured = {}

    class FakeSession:
        def get_inputs(self):
            return [SimpleNamespace(name="pixels")]

        def get_outputs(self):
            return [SimpleNamespace(name="embedding")]

        def run(self, _outputs, inputs):
            captured["training_input"] = inputs["pixels"]
            return [np.ones((1, 4), dtype=np.float32)]

    monkeypatch.setitem(
        __import__("sys").modules,
        "onnxruntime",
        SimpleNamespace(
            InferenceSession=lambda *_args, **_kwargs: FakeSession()
        ),
    )

    config_path = "artifacts/models/community_probe/preprocessor_config.json"
    image = np.arange(500 * 500 * 3, dtype=np.uint8).reshape(500, 500, 3)
    embedder = Embedder("unused.onnx", (384, 384), config_path)
    embedder.embed(image)

    runtime_classifier = FrameClassifier.__new__(FrameClassifier)
    runtime_classifier.preprocessor = ImagePreprocessor(config_path)
    runtime_input = runtime_classifier._preprocess(image)

    np.testing.assert_array_equal(captured["training_input"], runtime_input)
    assert runtime_input.shape == (1, 3, 384, 384)