import time
from typing import List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
import supervision as sv
from PIL import Image

from rfdetr.util.coco_classes import COCO_CLASSES


class RFDETROnnx:
    """ONNX Runtime inference wrapper for RF-DETR models."""

    # ImageNet normalization values
    means = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    stds = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_path: str = "output/inference_model.onnx",
        providers: Optional[List[str]] = None,
        num_select: int = 300,
    ):
        """
        Initialize ONNX Runtime inference session.

        Args:
            model_path: Path to the ONNX model file.
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).
                      If None, will try CUDA first, then fall back to CPU.
            num_select: Number of top detections to select (default 300, same as PostProcess).
        """
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.num_select = num_select

        # Get input information
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        input_shape = input_info.shape
        # Expected shape: [batch_size, 3, height, width]
        self.resolution = input_shape[2] if isinstance(input_shape[2], int) else 576

        # Get output names
        self.output_names = [output.name for output in self.session.get_outputs()]

        print(f"Loaded ONNX model from {model_path}")
        print(f"Input name: {self.input_name}, Resolution: {self.resolution}")
        print(f"Output names: {self.output_names}")
        print(f"Providers: {self.session.get_providers()}")

    def preprocess(
        self, image: Union[str, Image.Image, np.ndarray]
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for inference.

        Args:
            image: Input image (file path, PIL Image, or numpy array in RGB format).

        Returns:
            Tuple of (preprocessed tensor, original size (h, w)).
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if isinstance(image, Image.Image):
            orig_size = (image.height, image.width)
            image_array = np.array(image)
        else:
            orig_size = (image.shape[0], image.shape[1])
            image_array = image

        # Ensure RGB format and float32
        if image_array.dtype == np.uint8:
            image_array = image_array.astype(np.float32) / 255.0

        # Resize to model resolution (square resize)
        pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
        pil_image = pil_image.resize(
            (self.resolution, self.resolution), Image.Resampling.BILINEAR
        )
        image_array = np.array(pil_image).astype(np.float32) / 255.0

        # Normalize with ImageNet mean/std
        image_array = (image_array - self.means) / self.stds

        # Convert from HWC to CHW format
        image_array = image_array.transpose(2, 0, 1)

        return image_array, orig_size

    def postprocess(
        self,
        pred_boxes: np.ndarray,
        pred_logits: np.ndarray,
        orig_sizes: List[Tuple[int, int]],
        threshold: float = 0.5,
    ) -> List[sv.Detections]:
        """
        Postprocess model outputs to get detections.

        This implements the same logic as PostProcess in lwdetr.py.

        Args:
            pred_boxes: Predicted boxes in cxcywh format, shape [batch, num_queries, 4].
            pred_logits: Predicted logits, shape [batch, num_queries, num_classes].
            orig_sizes: List of original image sizes (h, w).
            threshold: Confidence threshold for filtering detections.

        Returns:
            List of sv.Detections objects.
        """
        batch_size = pred_boxes.shape[0]
        num_classes = pred_logits.shape[2]

        # Apply sigmoid to get probabilities
        prob = 1 / (1 + np.exp(-pred_logits))  # sigmoid

        # Get top-k predictions across all queries and classes
        prob_flat = prob.reshape(batch_size, -1)  # [batch, num_queries * num_classes]

        # Get top-k indices
        topk_indices = np.argsort(-prob_flat, axis=1)[:, : self.num_select]
        topk_values = np.take_along_axis(prob_flat, topk_indices, axis=1)

        # Decode indices to get box and class indices
        topk_boxes_idx = topk_indices // num_classes
        labels = topk_indices % num_classes

        # Convert boxes from cxcywh to xyxy
        boxes_xyxy = self._box_cxcywh_to_xyxy(pred_boxes)

        detections_list = []
        for i in range(batch_size):
            # Gather boxes for this batch
            box_indices = topk_boxes_idx[i]
            boxes_i = boxes_xyxy[i, box_indices]  # [num_select, 4]

            # Scale boxes to original image size
            h, w = orig_sizes[i]
            scale = np.array([w, h, w, h], dtype=np.float32)
            boxes_i = boxes_i * scale

            scores_i = topk_values[i]
            labels_i = labels[i]

            # Apply threshold
            keep = scores_i > threshold
            boxes_i = boxes_i[keep]
            scores_i = scores_i[keep]
            labels_i = labels_i[keep]

            detections = sv.Detections(
                xyxy=boxes_i.astype(np.float32),
                confidence=scores_i.astype(np.float32),
                class_id=labels_i.astype(np.int32),
            )
            detections_list.append(detections)

        return detections_list

    def _box_cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """
        Convert boxes from center format (cx, cy, w, h) to corner format (x1, y1, x2, y2).

        Args:
            boxes: Boxes in cxcywh format, shape [..., 4].

        Returns:
            Boxes in xyxy format, shape [..., 4].
        """
        cx, cy, w, h = np.split(boxes, 4, axis=-1)
        w = np.clip(w, 0, None)
        h = np.clip(h, 0, None)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return np.concatenate([x1, y1, x2, y2], axis=-1)

    def predict(
        self,
        images: Union[
            str,
            Image.Image,
            np.ndarray,
            List[Union[str, Image.Image, np.ndarray]],
        ],
        threshold: float = 0.5,
    ) -> Union[sv.Detections, List[sv.Detections]]:
        """
        Run inference on one or more images.

        Args:
            images: Single image or list of images.
            threshold: Confidence threshold for detections.

        Returns:
            Single Detections object or list of Detections objects.
        """
        single_image = not isinstance(images, list)
        if single_image:
            images = [images]

        # Preprocess all images
        processed_images = []
        orig_sizes = []
        for img in images:
            processed, orig_size = self.preprocess(img)
            processed_images.append(processed)
            orig_sizes.append(orig_size)

        # Stack into batch
        batch = np.stack(processed_images, axis=0).astype(np.float32)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: batch})

        # Parse outputs - the model outputs 'dets' (boxes) and 'labels' (logits)
        # Based on export.py: output_names = ['dets', 'labels']
        pred_boxes = np.asarray(outputs[0])  # dets: [batch, num_queries, 4]
        pred_logits = np.asarray(
            outputs[1]
        )  # labels: [batch, num_queries, num_classes]

        # Postprocess
        detections_list = self.postprocess(
            pred_boxes, pred_logits, orig_sizes, threshold
        )

        return detections_list[0] if single_image else detections_list


if __name__ == "__main__":
    # Load ONNX model
    start_time = time.time()
    model = RFDETROnnx(model_path="output/inference_model.onnx")
    model_load_time = time.time() - start_time
    print(f"Model loading time: {model_load_time:.4f} seconds")

    # Load test image
    image = Image.open("sample.jpg")

    # Warmup and benchmark
    warmup = 10
    iters = 10
    inference_start_time: Optional[float] = None
    detections: Optional[sv.Detections] = None

    for i in range(warmup + iters):
        if i >= warmup and inference_start_time is None:
            inference_start_time = time.time()
        result = model.predict(image, threshold=0.5)
        if isinstance(result, sv.Detections):
            detections = result

    if inference_start_time is not None:
        predict_time = (time.time() - inference_start_time) / iters
        print(f"Avg model prediction time: {predict_time:.4f} seconds")

    if (
        detections is not None
        and detections.class_id is not None
        and detections.confidence is not None
    ):
        # Create labels for visualization
        labels = [
            f"{COCO_CLASSES[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate image
        annotated_image = image.copy()
        annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
        annotated_image = sv.LabelAnnotator().annotate(
            annotated_image, detections, labels
        )

        # Save output
        if isinstance(annotated_image, Image.Image):
            annotated_image.save("output_onnx.jpg")
        else:
            Image.fromarray(annotated_image).save("output_onnx.jpg")

        print("Saved annotated image to output_onnx.jpg")
        print(f"Detected {len(detections)} objects")
    else:
        print("No detections were made")
