"""Lazy SAM3 text detection and single-object interactive video tracking."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

from image_review import Candidate
from video_data import Mask

if TYPE_CHECKING:
    from transformers import (
        Sam3Model,
        Sam3Processor,
        Sam3TrackerVideoModel,
        Sam3TrackerVideoProcessor,
    )
    from transformers.models.sam3_tracker_video.modeling_sam3_tracker_video import (
        Sam3TrackerVideoInferenceSession,
        Sam3TrackerVideoSegmentationOutput,
    )


class Sam3VideoBackend:
    def __init__(self, model_name: str, device: torch.device, threshold: float = 0.5):
        self.model_name = model_name
        self.device = device
        self.threshold = threshold
        self.detector: Sam3Model | None = None
        self.image_processor: Sam3Processor | None = None
        self.tracker: Sam3TrackerVideoModel | None = None
        self.video_processor: Sam3TrackerVideoProcessor | None = None

    @torch.inference_mode()
    def text(self, image: Image.Image, prompt: str) -> list[Candidate]:
        from transformers import Sam3Model, Sam3Processor

        if self.detector is None:
            self.detector = (
                Sam3Model.from_pretrained(self.model_name).to(self.device).eval()
            )
            self.image_processor = Sam3Processor.from_pretrained(self.model_name)
        assert self.image_processor is not None
        inputs = self.image_processor(
            images=[image], text=[prompt], return_tensors="pt"
        ).to(self.device)
        output = self.detector(**inputs)
        result = self.image_processor.post_process_instance_segmentation(
            output,
            threshold=self.threshold,
            mask_threshold=0.5,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        masks = result["masks"].detach().cpu().numpy()
        if masks.ndim == 4:
            masks = masks[:, 0]
        scores = result["scores"].detach().cpu().numpy().reshape(-1)
        return [
            Candidate(mask.astype(np.bool_), float(score), prompt, f"candidate-{index}")
            for index, (mask, score) in enumerate(zip(masks, scores, strict=True))
            if mask.any()
        ]

    def _session(
        self, images: Sequence[Image.Image]
    ) -> Sam3TrackerVideoInferenceSession:
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        if self.tracker is None:
            self.tracker = (
                Sam3TrackerVideoModel.from_pretrained(self.model_name)
                .to(self.device)
                .eval()
            )
            self.video_processor = Sam3TrackerVideoProcessor.from_pretrained(
                self.model_name
            )
        assert self.video_processor is not None
        return self.video_processor.init_video_session(
            video=list(images),
            inference_device=self.device,
            processing_device=self.device,
            video_storage_device="cpu",
            inference_state_device="cpu",
        )

    def _mask(
        self, output: Sam3TrackerVideoSegmentationOutput, size: tuple[int, int]
    ) -> Mask:
        assert self.video_processor is not None
        masks = self.video_processor.post_process_masks(
            [output.pred_masks],
            original_sizes=[list(size)],
            binarize=True,
        )[0]
        return masks.detach().cpu().numpy().reshape(-1, *size)[0].astype(np.bool_)

    @torch.inference_mode()
    def clicks(
        self,
        image: Image.Image,
        points: list[tuple[float, float, int]],
        box: tuple[int, int, int, int] | None = None,
    ) -> Mask:
        if not points and box is None:
            raise ValueError("Provide positive/negative clicks or a box.")
        session = self._session([image])
        assert self.video_processor is not None and self.tracker is not None
        self.video_processor.add_inputs_to_inference_session(
            session,
            frame_idx=0,
            obj_ids=1,
            input_points=[[[[x, y] for x, y, _ in points]]] if points else None,
            input_labels=[[[label for _, _, label in points]]] if points else None,
            input_boxes=[[list(box)]] if box is not None else None,
        )
        return self._mask(
            self.tracker(inference_session=session, frame_idx=0),
            (image.height, image.width),
        )

    @torch.inference_mode()
    def track(
        self,
        images: Sequence[Image.Image],
        anchors: dict[int, Mask],
        start: int,
        stop: int,
        reverse: bool,
    ) -> Iterator[tuple[int, Mask]]:
        session = self._session(images)
        assert self.video_processor is not None and self.tracker is not None
        for index, mask in sorted(anchors.items()):
            self.video_processor.add_inputs_to_inference_session(
                session,
                frame_idx=index,
                obj_ids=1,
                input_masks=mask,
            )
            self.tracker(inference_session=session, frame_idx=index)
        for output in self.tracker.propagate_in_video_iterator(
            inference_session=session,
            start_frame_idx=start,
            max_frame_num_to_track=abs(stop - start),
            reverse=reverse,
        ):
            index = int(output.frame_idx)
            mask = (
                anchors[index]
                if index in anchors
                else self._mask(output, (images[index].height, images[index].width))
            )
            yield index, mask
