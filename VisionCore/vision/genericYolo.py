import logging
import cv2
import numpy as np
from ultralytics import YOLO

try:
    from rknnlite.api import RKNNLite

    RKNN_FOUND = True
except ImportError:
    RKNN_FOUND = False


class Box:
    def __init__(self, xyxy, conf, cls_id=0):
        self.xyxy = xyxy
        self.conf = conf
        self.cls_id = cls_id


class Results:
    def __init__(
        self, boxes: list[Box], orig_shape, keypoints: list[np.ndarray] = None
    ):
        self.boxes = boxes
        self.orig_shape = orig_shape
        self.keypoints = keypoints if keypoints is not None else []

    def plot(self, frame):
        for box in self.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for kpt_set in self.keypoints:
            for kpt in kpt_set:
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

        return frame

    def __str__(self):
        return f"Results(boxes={len(self.boxes)}, keypoints={len(self.keypoints)})"


class GenericYolo:
    def __init__(self, model_config: dict, core_mask=None):
        self.logger = logging.getLogger(__name__)

        self.model_file = model_config["file_path"]
        self.task = model_config.get("task", "detect")
        self.has_hardware_nms = model_config.get("has_hardware_nms", False)
        self.num_classes = model_config.get("num_classes", 80)
        self.input_size = tuple(model_config.get("input_size", (640, 640)))
        self.min_conf = model_config.get("min_conf", 0.25)
        self.quantized = model_config.get("quantized", False)

        self.model_type = None

        if self.model_file.endswith(".rknn"):
            if not RKNN_FOUND:
                raise ImportError(
                    "rknnlite not installed but .rknn model was specified."
                )

            self.model_type = "rknn"
            self.model = RKNNLite()

            if self.model.load_rknn(self.model_file) != 0:
                raise ValueError(f"Failed to load RKNN model: {self.model_file}")

            if self.model.init_runtime(core_mask=core_mask) != 0:
                raise ValueError(f"Failed to init RKNN runtime: {self.model_file}")

            h, w = self.input_size[1], self.input_size[0]
            self._input_buf = np.empty((1, h, w, 3), dtype=np.uint8)

        elif (
            self.model_file.endswith(".pt")
            or self.model_file.endswith(".onnx")
            or "openvino_model" in self.model_file
            or self.model_file.endswith(".mlpackage")
        ):
            if self.model_file.endswith(".pt"):
                self.logger.info(".pt model — using Ultralytics backend directly.")

            self.model_type = "yolo"
            # Let Ultralytics auto-detect task from the model metadata
            self.model = YOLO(self.model_file, verbose=False)

        elif self.model_file.endswith(".tflite"):
            self.model_type = "tflite"
            self._tflite_buf = None  # allocated lazily on first frame
            self._load_tflite(self.model_file)

        else:
            raise ValueError(f"Unsupported model file type: {self.model_file}")

        self.logger.info(
            "GenericYolo loaded: %s  type=%s  task=%s  nms=%s",
            self.model_file,
            self.model_type,
            self.task,
            self.has_hardware_nms,
        )

    def _load_tflite(self, model_file: str):
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate

            delegates = []
            try:
                delegates = [load_delegate("libedgetpu.so.1")]
                self.logger.info("Coral Edge TPU delegate loaded.")
            except Exception:
                self.logger.info("No Edge TPU delegate — running TFLite on CPU.")
            self.model = Interpreter(
                model_path=model_file, experimental_delegates=delegates
            )
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter

            self.model = Interpreter(model_path=model_file)

        self.model.allocate_tensors()
        self._tflite_inp = self.model.get_input_details()[0]
        self._tflite_out = self.model.get_output_details()

    def _letterbox_into(
        self, img: np.ndarray, dst: np.ndarray, target_size: tuple
    ) -> None:
        h, w = img.shape[:2]
        target_w, target_h = target_size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        top = (target_h - new_h) // 2
        left = (target_w - new_w) // 2
        dst[:] = 114
        dst[top : top + new_h, left : left + new_w] = cv2.resize(img, (new_w, new_h))

    def _preprocess_for_rknn(self, frame: np.ndarray) -> np.ndarray:
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._letterbox_into(img_rgb, self._input_buf[0], self.input_size)
        return self._input_buf

    def _preprocess_for_tflite(self, frame: np.ndarray) -> np.ndarray:
        h, w = self.input_size[1], self.input_size[0]
        if self._tflite_buf is None:
            self._tflite_buf = np.empty((1, h, w, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._letterbox_into(img_rgb, self._tflite_buf[0], self.input_size)

        dtype = self._tflite_inp["dtype"]
        if dtype == np.uint8:
            return self._tflite_buf
        else:
            return self._tflite_buf.astype(np.float32) / 255.0

    def predict_preprocessed(self, preprocessed: np.ndarray, orig_shape) -> Results:
        if self.model_type != "rknn":
            raise RuntimeError("predict_preprocessed is only valid for RKNN models.")
        return self._run_rknn(preprocessed, orig_shape)

    def predict(self, frame_or_frames, orig_shape=None) -> "Results | list[Results]":
        is_list = isinstance(frame_or_frames, list)
        frames = frame_or_frames if is_list else [frame_or_frames]
        results_list = []

        for frame in frames:
            target_shape = orig_shape if orig_shape is not None else frame.shape

            if self.model_type == "rknn":
                results_list.append(
                    self._run_rknn(self._preprocess_for_rknn(frame), target_shape)
                )

            elif self.model_type == "tflite":
                results_list.append(self._run_tflite(frame, target_shape))

            else:
                result = self.model(
                    frame.copy(),
                    verbose=False,
                    show=False,
                    imgsz=(self.input_size[1], self.input_size[0]),
                    conf=self.min_conf,
                )
                result[0].orig_img = None
                results_list.append(self._convert_ultralytics_to_results(result[0]))

        return results_list if is_list else results_list[0]

    def _run_rknn(self, preprocessed: np.ndarray, orig_shape) -> Results:
        raw_outputs = self.model.inference(inputs=[preprocessed])
        if raw_outputs is None:
            return Results([], orig_shape)

        # Dequantize if needed — must happen before postprocess
        tensor = raw_outputs[0]
        if tensor.dtype == np.int8:
            tensor = tensor.astype(np.float32) / 128.0
        elif tensor.dtype == np.uint8:
            tensor = tensor.astype(np.float32) / 255.0
        raw_outputs[0] = tensor

        return self.postprocess(raw_outputs, orig_shape)

    def _run_tflite(self, frame: np.ndarray, orig_shape) -> Results:
        inp = self._preprocess_for_tflite(frame)
        self.model.set_tensor(self._tflite_inp["index"], inp)
        self.model.invoke()
        raw = [self.model.get_tensor(d["index"]) for d in self._tflite_out]
        return self.postprocess(raw, orig_shape)

    def postprocess(self, raw_outputs, orig_shape) -> Results:
        tensor = raw_outputs[0]

        # Remove batch dim if present
        if tensor.ndim == 3:
            tensor = tensor[0]

        # Transpose detection: shape is (D, N) when rows << cols
        # Use the expected feature width to decide, not a magic threshold.
        if tensor.ndim == 2:
            rows, cols = tensor.shape
            expected_feat = self._expected_feature_width()
            if rows == expected_feat and cols != expected_feat:
                # Clearly transposed: feature dim is on axis-0
                tensor = tensor.T
            elif rows < cols and cols == expected_feat:
                pass  # already (N, D)
            elif rows < cols and rows < cols // 2:
                # Heuristic fallback: very few rows relative to columns → transposed
                tensor = tensor.T

        if self.has_hardware_nms:
            return self._parse_hardware_nms(tensor, orig_shape)
        elif self.task == "detect":
            return self._parse_raw_detect(tensor, orig_shape)
        elif self.task == "pose":
            return self._parse_raw_pose(tensor, orig_shape)
        else:
            raise ValueError(
                f"Unsupported task: '{self.task}'. Use 'detect' or 'pose'."
            )

    def _expected_feature_width(self) -> int:
        if self.task == "pose":
            # 4 box + num_classes objectness + keypoints (num_kpts * 3)
            # We don't know num_kpts at init, so return the minimum (detect width)
            return 4 + self.num_classes
        # detect: 4 box coords + num_classes scores  (or 5 for single-class objectness)
        return 4 + self.num_classes

    def _scale_coords(
        self, coords: np.ndarray, orig_shape, is_kpts=False
    ) -> np.ndarray:
        orig_h, orig_w = orig_shape[:2]
        target_w, target_h = self.input_size
        scale = min(target_w / orig_w, target_h / orig_h)
        pad_x = (target_w - int(orig_w * scale)) / 2
        pad_y = (target_h - int(orig_h * scale)) / 2

        scaled = coords.copy().astype(np.float32)

        if not is_kpts:
            scaled[[0, 2]] = np.clip((scaled[[0, 2]] - pad_x) / scale, 0, orig_w)
            scaled[[1, 3]] = np.clip((scaled[[1, 3]] - pad_y) / scale, 0, orig_h)
        else:
            # keypoints: only x/y columns, leave confidence (col 2) untouched
            scaled[:, 0] = (scaled[:, 0] - pad_x) / scale
            scaled[:, 1] = (scaled[:, 1] - pad_y) / scale

        return scaled

    @staticmethod
    def _maybe_sigmoid(scores):
        # FIX: Return immediately if the array is empty
        if scores.size == 0:
            return scores
        if scores.min() < -0.1 or scores.max() > 1.1:
            return 1.0 / (1.0 + np.exp(-np.clip(scores, -88.0, 88.0)))
        return scores

    def _apply_software_nms(
        self,
        boxes_xyxy: np.ndarray,
        confs: np.ndarray,
        class_ids: np.ndarray,
        orig_shape,
        kpts_raw: np.ndarray = None,
    ) -> Results:
        mask = confs >= self.min_conf
        boxes_xyxy = boxes_xyxy[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]
        if kpts_raw is not None:
            kpts_raw = kpts_raw[mask]

        if len(boxes_xyxy) == 0:
            return Results([], orig_shape)

        # cv2.dnn.NMSBoxes expects [x, y, w, h]
        nms_input = [
            [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
            for b in boxes_xyxy
        ]
        indices = cv2.dnn.NMSBoxes(nms_input, confs.tolist(), self.min_conf, 0.45)
        indices = indices.flatten() if len(indices) > 0 else []

        final_boxes = []
        final_kpts = []

        for i in indices:
            xyxy = self._scale_coords(boxes_xyxy[i], orig_shape, is_kpts=False)
            final_boxes.append(Box(xyxy.tolist(), float(confs[i]), int(class_ids[i])))

            if kpts_raw is not None:
                # reshape to (num_kpts, 3) → scale x/y → keep conf column
                kpt_set = kpts_raw[i].reshape(-1, 3)
                kpt_scaled = self._scale_coords(kpt_set, orig_shape, is_kpts=True)
                final_kpts.append(kpt_scaled)

        return Results(
            final_boxes,
            orig_shape,
            keypoints=final_kpts if kpts_raw is not None else None,
        )

    def _parse_hardware_nms(self, tensor, orig_shape):
        # Safe list/tuple unboxing
        while isinstance(tensor, (list, tuple)) and len(tensor) > 0:
            tensor = tensor[0]
            
        # Safely peel off ONLY the first dimension if it represents a single batch axis
        if hasattr(tensor, 'ndim') and tensor.ndim == 3 and tensor.shape[0] == 1:
            tensor = tensor[0]

        # Handle empty outputs safely
        if tensor is None or (hasattr(tensor, 'size') and tensor.size == 0):
            return Results(orig_shape=orig_shape, boxes=[])

        # If it's a 2D tensor but has too few columns for a hardware NMS layout, early return
        if hasattr(tensor, 'shape') and len(tensor.shape) > 1 and tensor.shape[1] < 6:
            return Results(orig_shape=orig_shape, boxes=[])

        confs = tensor[:, 4]
        # ... rest of your original method code ...
        mask = confs >= self.min_conf
        valid = tensor[mask]

        boxes = []
        for det in valid:
            xyxy = self._scale_coords(det[:4], orig_shape)
            boxes.append(
                Box(
                    xyxy.tolist(), float(det[4]), int(det[5]) if det.shape[0] > 5 else 0
                )
            )

        return Results(orig_shape=orig_shape, boxes=[])
    
    def _parse_raw_detect(self, tensor: np.ndarray, orig_shape) -> Results:
        cx, cy, w, h = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
        boxes_xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        num_cols = tensor.shape[1]
        score_cols = num_cols - 4  # everything after the 4 box coords

        if score_cols == 1 or (self.num_classes == 1 and score_cols == 1):
            # Single objectness score — apply sigmoid if raw logits
            raw_confs = tensor[:, 4]
            confs = self._maybe_sigmoid(raw_confs)
            class_ids = np.zeros(len(confs), dtype=np.int32)
        else:
            # Multi-class scores
            class_scores = tensor[:, 4 : 4 + self.num_classes]
            class_scores = self._maybe_sigmoid(class_scores)
            confs = np.max(class_scores, axis=1)
            class_ids = np.argmax(class_scores, axis=1)

        return self._apply_software_nms(boxes_xyxy, confs, class_ids, orig_shape)

    def _parse_raw_pose(self, tensor, orig_shape):
        # Loop to unpack nested lists or tuples until we hit a raw array
        while isinstance(tensor, (list, tuple)) and len(tensor) > 0:
            tensor = tensor[0]
            
        # Squeeze out all single-dimensional outer batch axes
        if hasattr(tensor, 'ndim') and tensor.ndim > 2:
            tensor = np.squeeze(tensor)
            if tensor.ndim == 1:
                tensor = np.expand_dims(tensor, axis=0)

        cx, cy, w, h = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
        # ... rest of your code ...
        boxes_xyxy = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

        class_scores = tensor[:, 4 : 4 + self.num_classes]
        class_scores = self._maybe_sigmoid(class_scores)
        confs = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        kpts_start = 4 + self.num_classes
        kpts_raw = tensor[:, kpts_start:]  # (N, num_kpts*3)

        if kpts_raw.shape[1] == 0:
            self.logger.warning(
                "task='pose' but no keypoint columns found after class scores. "
                "Check num_classes=%d vs actual model output width %d.",
                self.num_classes,
                tensor.shape[1],
            )
            kpts_raw = None

        return self._apply_software_nms(
            boxes_xyxy, confs, class_ids, orig_shape, kpts_raw=kpts_raw
        )

    # ── Ultralytics converter ─────────────────────────────────────────────────

    def _convert_ultralytics_to_results(self, ultralytics_result) -> Results:
        boxes = []
        for b in ultralytics_result.boxes:
            xyxy = np.asarray(b.xyxy)
            xyxy = xyxy[0] if xyxy.ndim > 1 else xyxy
            conf = float(np.asarray(b.conf).item())
            cls_id = int(np.asarray(b.cls).item()) if hasattr(b, "cls") else 0
            boxes.append(Box(xyxy.tolist(), conf, cls_id))

        keypoints_list = []
        if (
            hasattr(ultralytics_result, "keypoints")
            and ultralytics_result.keypoints is not None
        ):
            kpt_data = ultralytics_result.keypoints.data
            # .data may be a torch.Tensor — convert safely
            if hasattr(kpt_data, "cpu"):
                kpt_data = kpt_data.cpu().numpy()
            for kpt_set in kpt_data:
                keypoints_list.append(np.asarray(kpt_set))

        return Results(boxes, ultralytics_result.orig_shape, keypoints_list or None)

    def release(self):
        if self.model_type == "rknn":
            self.model.release()
