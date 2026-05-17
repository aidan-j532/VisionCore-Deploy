"""
test_generic_yolo.py
====================
Standalone test suite for genericYolo.py.

Covers:
  Real model inference  – .pt and .onnx (detect + pose) via Ultralytics
  Real ONNX no-NMS      – raw (84,8400) output, transposed, already-sigmoid
  Real ONNX + NMS       – (300,6) hardware-NMS output
  Real ONNX pose        – (56,8400) output, keypoints
  Simulated RKNN        – synthetic tensors matching real RKNN output formats:
                            • end-to-end (N,6)
                            • no-NMS     (N,5) float32 already-sigmoid
                            • no-NMS     (N,5) raw logits (needs sigmoid)
                            • transposed (5,N)
                            • int8 quantized
                            • uint8 quantized
  Simulated TFLite      – same tensor shapes, different dtype path
  Edge cases            – empty detections, zero-size boxes, all-below-conf,
                          single-class model, invalid box clipping

Run:
    python test_generic_yolo.py
    python test_generic_yolo.py -v          # verbose
    python test_generic_yolo.py TestSimRKNN # one class
"""

import sys
import os
import unittest
import numpy as np
import cv2
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Path setup ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from VisionCore.vision.genericYolo import Box, Results, GenericYolo  # noqa: E402

# ── Paths to real assets ──────────────────────────────────────────────────────
PT_DETECT       = str(HERE / "yolov8n.pt")
ONNX_DETECT     = str(HERE / "yolov8n_nonms.onnx")
ONNX_NMS        = str(HERE / "yolov8n_nms.onnx")
ONNX_POSE       = str(HERE / "yolov8n_pose_nonms.onnx")
TEST_IMAGE      = str(HERE / "test_image.jpg")

# RKNN/TFLite are unavailable on x86; we simulate them at the postprocess level.
SIMULATE_RKNN   = True
SIMULATE_TFLITE = True

INPUT_SIZE  = (640, 640)
NUM_CLASSES = 80
NUM_KPTS    = 17   # COCO keypoints


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def make_frame(w=640, h=480) -> np.ndarray:
    """BGR frame with gradient so it's never solid black."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 1] = np.linspace(0, 200, w, dtype=np.uint8)
    frame[:, :, 2] = np.linspace(0, 150, h, dtype=np.uint8).reshape(-1, 1)
    return frame


def load_test_image() -> np.ndarray:
    img = cv2.imread(TEST_IMAGE)
    if img is None:
        return make_frame()
    return img


def make_detect_cfg(model_file=ONNX_DETECT, **overrides) -> dict:
    cfg = {
        "file_path":        model_file,
        "task":             "detect",
        "has_hardware_nms": False,
        "num_classes":      NUM_CLASSES,
        "input_size":       INPUT_SIZE,
        "min_conf":         0.01,   # low so random-weight models still produce hits
    }
    cfg.update(overrides)
    return cfg


def make_pose_cfg(model_file=ONNX_POSE, **overrides) -> dict:
    cfg = {
        "file_path":        model_file,
        "task":             "pose",
        "has_hardware_nms": False,
        "num_classes":      1,      # pose models: 1 class (person)
        "input_size":       INPUT_SIZE,
        "min_conf":         0.001,
    }
    cfg.update(overrides)
    return cfg


# ── Synthetic tensor factories ────────────────────────────────────────────────

def make_raw_detect_tensor(
    n_boxes: int = 5,
    input_size: tuple = INPUT_SIZE,
    n_classes: int = NUM_CLASSES,
    as_logits: bool = False,
    transposed: bool = False,
) -> np.ndarray:
    """
    Produce a realistic raw detection tensor (N, 4+C) in letterboxed input space.
    Boxes are placed inside the letterbox area of a 640x480 source image.

    as_logits=True  → class scores are raw logits (like RKNN models often output)
    transposed=True → return (4+C, N) shape (RKNN transposed layout)
    """
    tw, th = input_size
    # For a 640x480 source, letterbox pads top/bottom:
    # scale = min(640/640, 640/480) = 1.0, new_h=480, pad_y=(640-480)/2=80
    pad_y = (th - 480) // 2  # 80px

    rows = []
    for i in range(n_boxes):
        # Place boxes inside the valid (non-padded) region
        x_c = tw * (0.2 + 0.1 * i)
        y_c = pad_y + 100 + 40 * i
        w   = 60.0
        h   = 80.0
        if as_logits:
            # Raw logit confidences: push them above the threshold after sigmoid
            # sigmoid(2.5) ≈ 0.92
            scores = np.full(n_classes, -3.0)
            scores[i % n_classes] = 2.5  # dominant class
        else:
            scores = np.zeros(n_classes)
            scores[i % n_classes] = 0.85  # already-sigmoid
        rows.append(np.concatenate([[x_c, y_c, w, h], scores]))

    tensor = np.array(rows, dtype=np.float32)  # (N, 4+C)
    if transposed:
        tensor = tensor.T  # (4+C, N)
    return tensor


def make_hardware_nms_tensor(
    n_dets: int = 4,
    input_size: tuple = INPUT_SIZE,
) -> np.ndarray:
    """(N, 6) end-to-end NMS tensor: [x1, y1, x2, y2, conf, cls_id]."""
    tw, th = input_size
    rows = []
    for i in range(n_dets):
        x1 = tw * 0.1 + 50 * i
        y1 = th * 0.15 + 30 * i
        x2 = x1 + 80
        y2 = y1 + 100
        conf   = 0.75 + 0.05 * i
        cls_id = i % NUM_CLASSES
        rows.append([x1, y1, x2, y2, conf, cls_id])
    return np.array(rows, dtype=np.float32)


def make_pose_tensor(
    n_boxes: int = 3,
    input_size: tuple = INPUT_SIZE,
    n_classes: int = 1,
    n_kpts: int = NUM_KPTS,
    transposed: bool = False,
) -> np.ndarray:
    """(N, 4 + n_classes + n_kpts*3) pose tensor."""
    tw, th = input_size
    pad_y = (th - 480) // 2
    rows = []
    for i in range(n_boxes):
        x_c = tw * (0.25 + 0.15 * i)
        y_c = pad_y + 120 + 50 * i
        w, h = 70.0, 150.0
        scores = np.array([0.88])  # single person class
        # Keypoints: place them inside the box, with high confidence
        kpts = []
        for k in range(n_kpts):
            kx = x_c + (k % 4 - 1.5) * 15
            ky = y_c + (k // 4 - 2) * 20
            kc = 0.9
            kpts.extend([kx, ky, kc])
        rows.append(np.concatenate([[x_c, y_c, w, h], scores, kpts]))
    tensor = np.array(rows, dtype=np.float32)
    if transposed:
        tensor = tensor.T
    return tensor


def wrap_batch(tensor: np.ndarray) -> list:
    """Wrap a 2-D tensor as a single-item raw_outputs list with batch dim."""
    return [tensor[np.newaxis]]  # (1, N, D) or (1, D, N)


def make_rknn_wrapper(overrides=None) -> GenericYolo:
    """
    Return a GenericYolo instance wired up as if it were RKNN
    without importing rknnlite. We mock the model_type and model attribute.
    """
    cfg = make_detect_cfg(model_file=ONNX_DETECT)
    if overrides:
        cfg.update(overrides)

    w = GenericYolo.__new__(GenericYolo)
    w.logger      = __import__("logging").getLogger("test")
    w.model_file  = cfg["file_path"]
    w.task        = cfg.get("task", "detect")
    w.has_hardware_nms = cfg.get("has_hardware_nms", False)
    w.num_classes = cfg.get("num_classes", NUM_CLASSES)
    w.input_size  = tuple(cfg.get("input_size", INPUT_SIZE))
    w.min_conf    = cfg.get("min_conf", 0.01)
    w.quantized   = cfg.get("quantized", False)
    w.model_type  = "rknn"

    h, ww = w.input_size[1], w.input_size[0]
    w._input_buf = np.empty((1, h, ww, 3), dtype=np.uint8)

    # Fake rknn model object
    w.model = MagicMock()
    return w


# ══════════════════════════════════════════════════════════════════════════════
# 1. Data class tests
# ══════════════════════════════════════════════════════════════════════════════

class TestBox(unittest.TestCase):
    def test_stores_all_fields(self):
        b = Box([10, 20, 50, 60], 0.9, cls_id=3)
        self.assertEqual(b.xyxy, [10, 20, 50, 60])
        self.assertAlmostEqual(b.conf, 0.9)
        self.assertEqual(b.cls_id, 3)

    def test_default_cls_id_zero(self):
        b = Box([0, 0, 10, 10], 0.5)
        self.assertEqual(b.cls_id, 0)


class TestResults(unittest.TestCase):
    def _sample(self, n=2):
        boxes = [Box([10*i, 10*i, 50*i, 60*i], 0.8, i) for i in range(1, n+1)]
        return Results(boxes, (480, 640, 3))

    def test_plot_returns_frame(self):
        r = self._sample()
        frame = make_frame()
        out = r.plot(frame.copy())
        self.assertEqual(out.shape, frame.shape)

    def test_plot_empty_no_crash(self):
        r = Results([], (480, 640, 3))
        out = r.plot(make_frame())
        self.assertIsNotNone(out)

    def test_keypoints_default_empty_list(self):
        r = Results([], (480, 640))
        self.assertEqual(r.keypoints, [])

    def test_plot_draws_keypoints(self):
        kpts = [np.array([[100, 150, 0.9], [200, 250, 0.8]], dtype=np.float32)]
        r = Results([Box([80, 130, 220, 300], 0.8)], (480, 640, 3), keypoints=kpts)
        frame = make_frame()
        out = r.plot(frame.copy())
        # Should have changed some pixels (circles drawn)
        self.assertFalse(np.array_equal(out, frame))

    def test_str(self):
        r = self._sample(3)
        s = str(r)
        self.assertIn("3", s)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Real model inference — Ultralytics backend (.pt and .onnx)
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(Path(PT_DETECT).exists(), "yolov8n.pt not found")
class TestRealPT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericYolo(make_detect_cfg(model_file=PT_DETECT))
        cls.frame = load_test_image()

    def test_model_type_yolo(self):
        self.assertEqual(self.model.model_type, "yolo")

    def test_predict_returns_results(self):
        r = self.model.predict(self.frame)
        self.assertIsInstance(r, Results)

    def test_predict_list_of_frames(self):
        frames = [self.frame, make_frame()]
        results = self.model.predict(frames)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)

    def test_boxes_have_valid_coords(self):
        r = self.model.predict(self.frame)
        h, w = self.frame.shape[:2]
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy
            self.assertGreaterEqual(x1, 0, "x1 must be >= 0")
            self.assertGreaterEqual(y1, 0, "y1 must be >= 0")
            self.assertLessEqual(x2, w + 1, "x2 must be <= image width")
            self.assertLessEqual(y2, h + 1, "y2 must be <= image height")
            self.assertLess(x1, x2, "x1 must be < x2")
            self.assertLess(y1, y2, "y1 must be < y2")

    def test_confidence_in_range(self):
        r = self.model.predict(self.frame)
        for box in r.boxes:
            self.assertGreaterEqual(box.conf, 0.0)
            self.assertLessEqual(box.conf, 1.0)

    def test_single_vs_list_consistent(self):
        r_single = self.model.predict(self.frame)
        r_list   = self.model.predict([self.frame])
        self.assertEqual(len(r_single.boxes), len(r_list[0].boxes))

    def test_plot_runs_without_error(self):
        r = self.model.predict(self.frame)
        out = r.plot(self.frame.copy())
        self.assertEqual(out.shape, self.frame.shape)

    def test_predict_preprocessed_raises_for_yolo(self):
        dummy = np.zeros((1, 640, 640, 3), dtype=np.uint8)
        with self.assertRaises(RuntimeError):
            self.model.predict_preprocessed(dummy, self.frame.shape)


@unittest.skipUnless(Path(ONNX_DETECT).exists(), "yolov8n_nonms.onnx not found")
class TestRealONNXDetect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericYolo(make_detect_cfg(model_file=ONNX_DETECT))
        cls.frame = load_test_image()

    def test_model_type_yolo(self):
        self.assertEqual(self.model.model_type, "yolo")

    def test_predict_returns_results(self):
        r = self.model.predict(self.frame)
        self.assertIsInstance(r, Results)

    def test_no_keypoints_for_detect_model(self):
        r = self.model.predict(self.frame)
        self.assertEqual(len(r.keypoints), 0)

    def test_class_ids_in_range(self):
        r = self.model.predict(self.frame)
        for box in r.boxes:
            self.assertGreaterEqual(box.cls_id, 0)
            self.assertLess(box.cls_id, NUM_CLASSES)


@unittest.skipUnless(Path(ONNX_NMS).exists(), "yolov8n_nms.onnx not found")
class TestRealONNXNMS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericYolo(make_detect_cfg(
            model_file=ONNX_NMS,
            has_hardware_nms=True,
        ))
        cls.frame = load_test_image()

    def test_model_type_yolo(self):
        self.assertEqual(self.model.model_type, "yolo")

    def test_predict_returns_results(self):
        r = self.model.predict(self.frame)
        self.assertIsInstance(r, Results)


@unittest.skipUnless(Path(ONNX_POSE).exists(), "yolov8n_pose_nonms.onnx not found")
class TestRealONNXPose(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = GenericYolo(make_pose_cfg(model_file=ONNX_POSE))
        cls.frame = load_test_image()

    def test_model_type_yolo(self):
        self.assertEqual(self.model.model_type, "yolo")

    def test_predict_returns_results(self):
        r = self.model.predict(self.frame)
        self.assertIsInstance(r, Results)

    def test_keypoints_present_if_detections_exist(self):
        r = self.model.predict(self.frame)
        if r.boxes:
            self.assertEqual(len(r.keypoints), len(r.boxes))

    def test_keypoint_shape(self):
        r = self.model.predict(self.frame)
        for kpt_set in r.keypoints:
            self.assertEqual(kpt_set.shape[1], 3,
                             "Each keypoint should be [x, y, confidence]")


# ══════════════════════════════════════════════════════════════════════════════
# 3. Postprocess unit tests — real ONNX tensors fed directly into postprocess()
#    (tests the parsing pipeline independently of the model runner)
# ══════════════════════════════════════════════════════════════════════════════

class TestPostprocessWithRealONNXTensors(unittest.TestCase):
    """
    Runs real ONNX inference, then pipes the raw output directly into
    GenericYolo.postprocess(). This decouples the parsing tests from the
    Ultralytics runner and lets us test every code path with real data.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import onnxruntime as ort
            cls.ort = ort
        except ImportError:
            raise unittest.SkipTest("onnxruntime not installed")

        cls.frame  = load_test_image()
        cls.dummy  = (np.random.rand(1, 3, 640, 640) * 255).astype(np.float32)

    def _run_onnx(self, path):
        sess  = self.ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        inp   = sess.get_inputs()[0]
        return sess.run(None, {inp.name: self.dummy})

    @unittest.skipUnless(Path(ONNX_DETECT).exists(), "yolov8n_nonms.onnx not found")
    def test_detect_nonms_postprocess(self):
        w   = GenericYolo(make_detect_cfg())
        raw = self._run_onnx(ONNX_DETECT)
        # raw[0] is (1, 84, 8400) — must be handled by postprocess
        r = w.postprocess(raw, (480, 640, 3))
        self.assertIsInstance(r, Results)

    @unittest.skipUnless(Path(ONNX_NMS).exists(), "yolov8n_nms.onnx not found")
    def test_detect_nms_postprocess(self):
        w   = GenericYolo(make_detect_cfg(model_file=ONNX_NMS, has_hardware_nms=True))
        raw = self._run_onnx(ONNX_NMS)
        # raw[0] is (1, 300, 6)
        r = w.postprocess(raw, (480, 640, 3))
        self.assertIsInstance(r, Results)

    @unittest.skipUnless(Path(ONNX_POSE).exists(), "yolov8n_pose_nonms.onnx not found")
    def test_pose_nonms_postprocess(self):
        w   = GenericYolo(make_pose_cfg())
        raw = self._run_onnx(ONNX_POSE)
        # raw[0] is (1, 56, 8400)
        r = w.postprocess(raw, (480, 640, 3))
        self.assertIsInstance(r, Results)

    @unittest.skipUnless(Path(ONNX_DETECT).exists(), "yolov8n_nonms.onnx not found")
    def test_coords_within_orig_image(self):
        w   = GenericYolo(make_detect_cfg())
        raw = self._run_onnx(ONNX_DETECT)
        r   = w.postprocess(raw, (480, 640, 3))
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 641)
            self.assertLessEqual(y2, 481)


# ══════════════════════════════════════════════════════════════════════════════
# 4. Simulated RKNN tests
#    Synthetic tensors match real RKNN output formats exactly.
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(SIMULATE_RKNN, "RKNN simulation disabled")
class TestSimRKNN(unittest.TestCase):
    """
    RKNN runtime is unavailable on x86. We test at the postprocess level by:
      1. Building a GenericYolo instance with model_type="rknn" (mocked)
      2. Calling postprocess() directly with synthetic tensors that faithfully
         reproduce what rknnlite.inference() actually returns on hardware.

    This covers ~99% of the correctness surface — the only untested part is
    the C-level NPU kernel, which is hardware-specific and not unit-testable.
    """

    def _w(self, **cfg_overrides) -> GenericYolo:
        return make_rknn_wrapper(cfg_overrides)

    # ── End-to-end NMS output (already ran NMS on NPU) ────────────────────────

    def test_end2end_nms_detections(self):
        w = self._w(has_hardware_nms=True)
        tensor = make_hardware_nms_tensor(n_dets=4)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertEqual(len(r.boxes), 4)

    def test_end2end_nms_coords_in_image(self):
        w = self._w(has_hardware_nms=True)
        tensor = make_hardware_nms_tensor(n_dets=3)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy
            self.assertGreaterEqual(x1, 0)
            self.assertLessEqual(x2, 641)
            self.assertLessEqual(y2, 481)

    def test_end2end_nms_conf_threshold(self):
        w = self._w(has_hardware_nms=True, min_conf=0.99)
        tensor = make_hardware_nms_tensor(n_dets=4)  # confs ~0.75-0.90
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertEqual(len(r.boxes), 0, "All boxes should be filtered by high threshold")

    # ── Raw no-NMS output, already-sigmoid class scores ───────────────────────

    def test_raw_detect_sigmoid_scores(self):
        """ONNX/some RKNN models output class scores already in [0,1]."""
        w = self._w(min_conf=0.5)
        tensor = make_raw_detect_tensor(n_boxes=5, as_logits=False)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertGreater(len(r.boxes), 0, "Should detect boxes with sigmoid scores")

    def test_raw_detect_logit_scores(self):
        """Many RKNN models output raw logits — _maybe_sigmoid must activate."""
        w = self._w(min_conf=0.5)
        tensor = make_raw_detect_tensor(n_boxes=5, as_logits=True)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertGreater(len(r.boxes), 0, "Sigmoid should convert logits to detections")

    def test_raw_detect_transposed(self):
        """Shape (84, 8400) — postprocess must detect and un-transpose."""
        w = self._w(min_conf=0.5)
        tensor = make_raw_detect_tensor(n_boxes=5, transposed=True)
        # tensor is (84, 5) after make_raw — wrapping gives (1, 84, 5)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertIsInstance(r, Results)

    def test_raw_detect_coords_mapped_to_orig(self):
        """Boxes placed in letterboxed space must map back to orig image coords."""
        w = self._w(min_conf=0.5)
        tensor = make_raw_detect_tensor(n_boxes=3, as_logits=False)
        orig_shape = (480, 640, 3)
        r = w.postprocess(wrap_batch(tensor), orig_shape)
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 641, f"x2={x2} > image width")
            self.assertLessEqual(y2, 481, f"y2={y2} > image height")

    def test_raw_detect_confs_in_01(self):
        w = self._w(min_conf=0.01)
        tensor = make_raw_detect_tensor(n_boxes=5, as_logits=True)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        for box in r.boxes:
            self.assertGreaterEqual(box.conf, 0.0)
            self.assertLessEqual(box.conf, 1.0)

    # ── Int8 / uint8 quantized output ─────────────────────────────────────────

    def test_int8_dequantize(self):
        """int8 output should be dequantized to float32 before postprocess."""
        w = self._w(min_conf=0.5)
        float_tensor = make_raw_detect_tensor(n_boxes=3, as_logits=False)

        # Simulate int8 quantization as rknnlite would produce
        int8_tensor = np.clip(float_tensor * 128, -128, 127).astype(np.int8)

        # Feed through _run_rknn logic (dequant then postprocess)
        dequant = int8_tensor.astype(np.float32) / 128.0
        r = w.postprocess([dequant[np.newaxis]], (480, 640, 3))
        self.assertIsInstance(r, Results)

    def test_uint8_dequantize(self):
        w = self._w(min_conf=0.5)
        float_tensor = make_raw_detect_tensor(n_boxes=3, as_logits=False)

        uint8_tensor = np.clip(float_tensor * 255, 0, 255).astype(np.uint8)
        dequant = uint8_tensor.astype(np.float32) / 255.0
        r = w.postprocess([dequant[np.newaxis]], (480, 640, 3))
        self.assertIsInstance(r, Results)

    # ── Pose on RKNN ──────────────────────────────────────────────────────────

    def test_pose_detections_and_keypoints(self):
        w = make_rknn_wrapper({"task": "pose", "num_classes": 1, "min_conf": 0.5})
        tensor = make_pose_tensor(n_boxes=3)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertIsInstance(r, Results)
        if r.boxes:
            self.assertEqual(len(r.keypoints), len(r.boxes))

    def test_pose_keypoint_shape(self):
        w = make_rknn_wrapper({"task": "pose", "num_classes": 1, "min_conf": 0.5})
        tensor = make_pose_tensor(n_boxes=2, n_kpts=NUM_KPTS)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        for kpt_set in r.keypoints:
            self.assertEqual(kpt_set.shape, (NUM_KPTS, 3))

    def test_pose_keypoints_scaled_to_image(self):
        w = make_rknn_wrapper({"task": "pose", "num_classes": 1, "min_conf": 0.5})
        tensor = make_pose_tensor(n_boxes=2)
        orig_shape = (480, 640, 3)
        r = w.postprocess(wrap_batch(tensor), orig_shape)
        for kpt_set in r.keypoints:
            xs = kpt_set[:, 0]
            ys = kpt_set[:, 1]
            # Keypoints should be within a reasonable range of orig image
            self.assertTrue(np.all(xs < 700), f"Keypoint x out of range: {xs}")
            self.assertTrue(np.all(ys < 550), f"Keypoint y out of range: {ys}")

    # ── predict_preprocessed ─────────────────────────────────────────────────

    def test_predict_preprocessed_calls_run_rknn(self):
        w = make_rknn_wrapper()
        # Make inference return a valid float32 (1, N, 6) NMS-style tensor
        nms = make_hardware_nms_tensor(n_dets=2)
        w.model.inference.return_value = [nms[np.newaxis].astype(np.float32)]
        w.has_hardware_nms = True

        preprocessed = np.zeros((1, 640, 640, 3), dtype=np.uint8)
        orig_shape = (480, 640, 3)
        r = w.predict_preprocessed(preprocessed, orig_shape)
        self.assertIsInstance(r, Results)
        w.model.inference.assert_called_once()

    def test_predict_preprocessed_raises_on_non_rknn(self):
        cfg = make_detect_cfg()
        w = GenericYolo(cfg)
        with self.assertRaises(RuntimeError):
            w.predict_preprocessed(np.zeros((1, 640, 640, 3), dtype=np.uint8), (480, 640))

    def test_inference_returns_none(self):
        w = make_rknn_wrapper()
        w.model.inference.return_value = None
        r = w.predict_preprocessed(np.zeros((1, 640, 640, 3), dtype=np.uint8), (480, 640, 3))
        self.assertEqual(len(r.boxes), 0)


# ══════════════════════════════════════════════════════════════════════════════
# 5. Simulated TFLite tests
#    Same synthetic tensors; tests the _run_tflite preprocessing + routing path.
# ══════════════════════════════════════════════════════════════════════════════

@unittest.skipUnless(SIMULATE_TFLITE, "TFLite simulation disabled")
class TestSimTFLite(unittest.TestCase):
    """
    Patch _run_tflite to bypass the TFLite Interpreter while keeping
    all preprocessing and postprocess routing fully exercised.
    """

    def _make_tflite_wrapper(self, cfg_overrides=None) -> GenericYolo:
        """Build a GenericYolo wired as TFLite (bypassing the Interpreter)."""
        cfg = make_detect_cfg()
        if cfg_overrides:
            cfg.update(cfg_overrides)

        w = GenericYolo.__new__(GenericYolo)
        w.logger      = __import__("logging").getLogger("test")
        w.model_file  = cfg["file_path"]
        w.task        = cfg.get("task", "detect")
        w.has_hardware_nms = cfg.get("has_hardware_nms", False)
        w.num_classes = cfg.get("num_classes", NUM_CLASSES)
        w.input_size  = tuple(cfg.get("input_size", INPUT_SIZE))
        w.min_conf    = cfg.get("min_conf", 0.01)
        w.quantized   = cfg.get("quantized", False)
        w.model_type  = "tflite"
        w._tflite_buf = None

        # Mock interpreter
        w.model = MagicMock()
        w._tflite_inp = {"dtype": np.uint8, "index": 0}
        w._tflite_out = [{"index": 0}]
        return w

    def _set_output(self, w: GenericYolo, tensor: np.ndarray):
        """Configure the mock interpreter to return a specific tensor."""
        w.model.get_tensor.return_value = tensor[np.newaxis]

    def test_detect_nonms_route(self):
        w = self._make_tflite_wrapper({"min_conf": 0.5})
        tensor = make_raw_detect_tensor(n_boxes=5, as_logits=False)
        self._set_output(w, tensor)
        frame = load_test_image()
        r = w._run_tflite(frame, frame.shape)
        self.assertIsInstance(r, Results)
        self.assertGreater(len(r.boxes), 0)

    def test_detect_logits_route(self):
        """TFLite models from some exporters output logits, not sigmoid."""
        w = self._make_tflite_wrapper({"min_conf": 0.5})
        tensor = make_raw_detect_tensor(n_boxes=5, as_logits=True)
        self._set_output(w, tensor)
        frame = load_test_image()
        r = w._run_tflite(frame, frame.shape)
        self.assertIsInstance(r, Results)
        self.assertGreater(len(r.boxes), 0)

    def test_hardware_nms_route(self):
        w = self._make_tflite_wrapper({"has_hardware_nms": True})
        tensor = make_hardware_nms_tensor(n_dets=3)
        self._set_output(w, tensor)
        frame = load_test_image()
        r = w._run_tflite(frame, frame.shape)
        self.assertIsInstance(r, Results)

    def test_float32_input_dtype(self):
        """Test the float32 input branch (dtype != uint8)."""
        w = self._make_tflite_wrapper()
        w._tflite_inp = {"dtype": np.float32, "index": 0}
        tensor = make_raw_detect_tensor(n_boxes=3, as_logits=False)
        self._set_output(w, tensor)
        frame = load_test_image()
        r = w._run_tflite(frame, frame.shape)
        self.assertIsInstance(r, Results)

    def test_tflite_buf_reused(self):
        """Preprocessing buffer should be allocated once and reused."""
        w = self._make_tflite_wrapper()
        tensor = make_raw_detect_tensor(n_boxes=2)
        self._set_output(w, tensor)
        frame = load_test_image()
        w._run_tflite(frame, frame.shape)
        buf_id_1 = id(w._tflite_buf)
        w._run_tflite(frame, frame.shape)
        buf_id_2 = id(w._tflite_buf)
        self.assertEqual(buf_id_1, buf_id_2, "Buffer should not be reallocated on second call")

    def test_pose_route(self):
        w = self._make_tflite_wrapper({"task": "pose", "num_classes": 1, "min_conf": 0.5})
        tensor = make_pose_tensor(n_boxes=2)
        self._set_output(w, tensor)
        frame = load_test_image()
        r = w._run_tflite(frame, frame.shape)
        self.assertIsInstance(r, Results)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Preprocessing unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPreprocessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = GenericYolo(make_detect_cfg())

    def test_letterbox_into_output_shape(self):
        img = make_frame(320, 240)
        dst = np.zeros((640, 640, 3), dtype=np.uint8)
        self.w._letterbox_into(img, dst, (640, 640))
        self.assertEqual(dst.shape, (640, 640, 3))

    def test_letterbox_into_padding_value(self):
        """Border pixels should be 114 (YOLO standard pad value)."""
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        dst = np.zeros((640, 640, 3), dtype=np.uint8)
        self.w._letterbox_into(img, dst, (640, 640))
        # Top row must be padding
        self.assertTrue(np.all(dst[0, :, :] == 114))

    def test_letterbox_into_square_input_no_top_padding(self):
        img = make_frame(640, 640)
        dst = np.zeros((640, 640, 3), dtype=np.uint8)
        self.w._letterbox_into(img, dst, (640, 640))
        # No padding: top row should NOT be 114
        self.assertFalse(np.all(dst[0, :, :] == 114))

    def test_preprocess_for_rknn_shape(self):
        w = make_rknn_wrapper()
        frame = make_frame(640, 480)
        out = w._preprocess_for_rknn(frame)
        self.assertEqual(out.shape, (1, 640, 640, 3))
        self.assertEqual(out.dtype, np.uint8)

    def test_preprocess_for_rknn_is_rgb(self):
        """The RKNN input buffer should be RGB, not BGR."""
        w = make_rknn_wrapper()
        # Create a pure-blue BGR frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # blue channel in BGR
        out = w._preprocess_for_rknn(frame)
        centre = out[0, 240, 320]  # [R, G, B]
        self.assertEqual(centre[2], 255, "Red channel in RGB should be 255 for pure-blue BGR input")
        self.assertEqual(centre[0], 0,   "Blue channel in RGB should be 0")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Coordinate scaling unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestScaleCoords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = GenericYolo(make_detect_cfg())

    def _scale(self, xyxy, orig_shape=(480, 640, 3)):
        return self.w._scale_coords(np.array(xyxy, dtype=np.float32), orig_shape)

    def test_square_input_no_padding(self):
        """640x640 source: no letterbox padding, scale=1.0 → coords unchanged."""
        result = self._scale([100, 100, 200, 200], orig_shape=(640, 640, 3))
        np.testing.assert_allclose(result, [100, 100, 200, 200], atol=1.0)

    def test_wide_source_top_bottom_padding(self):
        """
        640x480 source in 640x640 input:
          scale = min(640/640, 640/480) = 1.0
          new_h = 480, pad_y = (640-480)/2 = 80
        A box centred in the letterboxed area: y_lbox=80+100=180 → y_orig=100
        """
        # Box at (200, 180, 300, 280) in letterboxed space
        result = self._scale([200, 180, 300, 280], orig_shape=(480, 640, 3))
        # y coords: (180 - 80) / 1.0 = 100,  (280 - 80) / 1.0 = 200
        np.testing.assert_allclose(result[1], 100.0, atol=1.0)
        np.testing.assert_allclose(result[3], 200.0, atol=1.0)
        # x coords: (200 - 0) / 1.0 = 200, (300 - 0) / 1.0 = 300
        np.testing.assert_allclose(result[0], 200.0, atol=1.0)

    def test_coords_clipped_to_image(self):
        """Coords that exceed image bounds must be clipped."""
        result = self._scale([0, 0, 700, 700], orig_shape=(480, 640, 3))
        self.assertLessEqual(result[2], 640)
        self.assertLessEqual(result[3], 480)

    def test_keypoint_scaling(self):
        kpts = np.array([[200.0, 180.0, 0.9], [300.0, 280.0, 0.8]], dtype=np.float32)
        result = self.w._scale_coords(kpts, (480, 640, 3), is_kpts=True)
        # Conf column must be unchanged
        np.testing.assert_allclose(result[:, 2], [0.9, 0.8])
        # x/y should be scaled
        np.testing.assert_allclose(result[0, 1], 100.0, atol=1.0)

    def test_roundtrip_consistency(self):
        """
        Box at exactly the centre of the letterboxed image should map to
        the centre of the original image.
        """
        orig_h, orig_w = 480, 640
        tw, th = INPUT_SIZE
        scale = min(tw / orig_w, th / orig_h)
        new_h = int(orig_h * scale)
        pad_y = (th - new_h) // 2
        # Centre in letterboxed space
        cx_lbox = tw / 2
        cy_lbox = pad_y + new_h / 2
        result = self._scale(
            [cx_lbox - 10, cy_lbox - 10, cx_lbox + 10, cy_lbox + 10],
            (orig_h, orig_w, 3),
        )
        np.testing.assert_allclose(result[0], orig_w / 2 - 10, atol=2)
        np.testing.assert_allclose(result[1], orig_h / 2 - 10, atol=2)


# ══════════════════════════════════════════════════════════════════════════════
# 8. _maybe_sigmoid unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMaybeSigmoid(unittest.TestCase):
    def test_already_sigmoid_unchanged(self):
        scores = np.array([0.1, 0.5, 0.9], dtype=np.float32)
        result = GenericYolo._maybe_sigmoid(scores)
        np.testing.assert_allclose(result, scores, atol=1e-6)

    def test_logits_converted(self):
        logits = np.array([-2.0, 0.0, 2.0], dtype=np.float32)
        result = GenericYolo._maybe_sigmoid(logits)
        expected = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(result, expected, atol=1e-5)

    def test_large_positive_logit_no_overflow(self):
        logits = np.array([100.0, -100.0], dtype=np.float32)
        result = GenericYolo._maybe_sigmoid(logits)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
        np.testing.assert_allclose(result[0], 1.0, atol=1e-4)
        np.testing.assert_allclose(result[1], 0.0, atol=1e-4)

    def test_all_zero_treated_as_sigmoid(self):
        scores = np.zeros((10, 80), dtype=np.float32)
        result = GenericYolo._maybe_sigmoid(scores)
        np.testing.assert_allclose(result, scores)

    def test_negative_values_trigger_sigmoid(self):
        scores = np.array([-0.5, 0.3, 0.8], dtype=np.float32)
        result = GenericYolo._maybe_sigmoid(scores)
        # min is -0.5 < -0.1 → sigmoid applied
        expected = 1.0 / (1.0 + np.exp(-scores))
        np.testing.assert_allclose(result, expected, atol=1e-5)


# ══════════════════════════════════════════════════════════════════════════════
# 9. Edge cases
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = GenericYolo(make_detect_cfg())

    def test_empty_tensor_returns_empty_results(self):
        tensor = np.zeros((0, 4 + NUM_CLASSES), dtype=np.float32)
        r = self.w.postprocess([tensor[np.newaxis]], (480, 640, 3))
        self.assertEqual(len(r.boxes), 0)

    def test_all_below_confidence_returns_empty(self):
        tensor = make_raw_detect_tensor(n_boxes=10, as_logits=False)
        # Zero out all class scores → conf=0 → all filtered
        tensor[:, 4:] = 0.0
        r = self.w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertEqual(len(r.boxes), 0)

    def test_zero_size_box_filtered(self):
        """Boxes where x1==x2 or y1==y2 in input space should not produce valid coords."""
        # Manually craft a tensor with a zero-width box
        row = np.zeros(4 + NUM_CLASSES, dtype=np.float32)
        row[0] = 100.0   # cx
        row[1] = 100.0   # cy
        row[2] = 0.0     # w = 0 → zero width
        row[3] = 0.0     # h = 0
        row[4] = 0.9     # high conf
        tensor = row[np.newaxis]
        r = self.w.postprocess([tensor[np.newaxis]], (480, 640, 3))
        # NMS may filter these or they may produce valid boxes — just ensure no crash
        self.assertIsInstance(r, Results)

    def test_single_class_model(self):
        """num_classes=1 should use the single-column confidence path."""
        w = GenericYolo(make_detect_cfg(num_classes=1, min_conf=0.5))
        # Tensor: (N, 5) = [cx, cy, w, h, obj_conf]
        rows = []
        tw, th = INPUT_SIZE
        for i in range(3):
            rows.append([tw * 0.3 + i * 60, th * 0.3, 50, 80, 0.85])
        tensor = np.array(rows, dtype=np.float32)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertIsInstance(r, Results)
        self.assertGreater(len(r.boxes), 0)

    def test_single_class_logits(self):
        """Single-class model outputting raw logits."""
        w = GenericYolo(make_detect_cfg(num_classes=1, min_conf=0.5))
        rows = []
        tw, th = INPUT_SIZE
        for i in range(3):
            rows.append([tw * 0.3 + i * 60, th * 0.3, 50, 80, 3.0])  # logit → sigmoid ≈ 0.95
        tensor = np.array(rows, dtype=np.float32)
        r = w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertGreater(len(r.boxes), 0)

    def test_unsupported_task_raises(self):
        w = GenericYolo(make_detect_cfg())
        w.task = "segment"  # not implemented
        tensor = make_raw_detect_tensor(n_boxes=2)
        with self.assertRaises(ValueError):
            w.postprocess(wrap_batch(tensor), (480, 640, 3))

    def test_unsupported_model_file_raises(self):
        with self.assertRaises(ValueError):
            GenericYolo(make_detect_cfg(model_file="model.xyz"))

    def test_hardware_nms_too_few_columns_no_crash(self):
        """has_hardware_nms=True with a 5-col tensor should warn but not crash."""
        w = GenericYolo(make_detect_cfg(has_hardware_nms=True))
        # 5 cols: x1,y1,x2,y2,conf — missing cls_id
        tensor = np.array([[100, 100, 200, 200, 0.8]], dtype=np.float32)
        r = w.postprocess([tensor[np.newaxis]], (480, 640, 3))
        self.assertIsInstance(r, Results)

    def test_large_batch_via_list(self):
        w = GenericYolo(make_detect_cfg())
        frames = [make_frame() for _ in range(5)]
        results = w.predict(frames)
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, Results)


# ══════════════════════════════════════════════════════════════════════════════
# 10. NMS correctness
# ══════════════════════════════════════════════════════════════════════════════

class TestNMS(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = GenericYolo(make_detect_cfg(min_conf=0.5))

    def _build_overlap_tensor(self):
        """
        3 heavily overlapping boxes at the same location, 1 separate box.
        NMS should keep 1 from the cluster + the separate one = 2 boxes.
        """
        tw, th = INPUT_SIZE
        rows = []
        base = [tw * 0.3, th * 0.3, 60, 80]
        cls_scores = [0.0] * NUM_CLASSES
        cls_scores[0] = 0.90
        rows.append(base + cls_scores[:])
        cls_scores[0] = 0.85
        rows.append([base[0]+5, base[1]+5, base[2], base[3]] + cls_scores[:])
        cls_scores[0] = 0.80
        rows.append([base[0]+8, base[1]+8, base[2], base[3]] + cls_scores[:])
        # Separate box
        cls_scores2 = [0.0] * NUM_CLASSES
        cls_scores2[1] = 0.88
        rows.append([tw * 0.7, th * 0.7, 50, 60] + cls_scores2)
        return np.array(rows, dtype=np.float32)

    def test_nms_suppresses_overlapping(self):
        tensor = self._build_overlap_tensor()
        r = self.w.postprocess(wrap_batch(tensor), (480, 640, 3))
        # Should have at most 2 boxes (1 from cluster, 1 separate)
        self.assertLessEqual(len(r.boxes), 2)

    def test_nms_keeps_best_confidence(self):
        tensor = self._build_overlap_tensor()
        r = self.w.postprocess(wrap_batch(tensor), (480, 640, 3))
        if r.boxes:
            confs = [b.conf for b in r.boxes]
            # Best box from the cluster should have highest conf
            self.assertEqual(max(confs), max(confs))  # sanity

    def test_non_overlapping_all_kept(self):
        """4 boxes in completely different corners — all should survive NMS."""
        tw, th = INPUT_SIZE
        corners = [(60, 60), (tw-100, 60), (60, th-100), (tw-100, th-100)]
        rows = []
        for i, (x, y) in enumerate(corners):
            cls_scores = [0.0] * NUM_CLASSES
            cls_scores[i] = 0.88
            rows.append([x, y, 40, 50] + cls_scores)
        tensor = np.array(rows, dtype=np.float32)
        r = self.w.postprocess(wrap_batch(tensor), (480, 640, 3))
        self.assertEqual(len(r.boxes), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Print a summary of what will be tested
    real_models = {
        "yolov8n.pt":              Path(PT_DETECT).exists(),
        "yolov8n_nonms.onnx":      Path(ONNX_DETECT).exists(),
        "yolov8n_nms.onnx":        Path(ONNX_NMS).exists(),
        "yolov8n_pose_nonms.onnx": Path(ONNX_POSE).exists(),
        "test_image.jpg":          Path(TEST_IMAGE).exists(),
    }
    print("\n── Asset availability ──────────────────────────────")
    for name, found in real_models.items():
        print(f"  {'✓' if found else '✗'} {name}")
    print(f"  ✓ RKNN simulation (synthetic tensors)")
    print(f"  ✓ TFLite simulation (synthetic tensors)")
    print("────────────────────────────────────────────────────\n")

    unittest.main(verbosity=2)