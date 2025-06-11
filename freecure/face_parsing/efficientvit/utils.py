try:
    # import pdb;pdb.set_trace()
    from inference.models import YOLOWorld
    from freecure.face_parsing.efficientvit.models.efficientvit.sam import EfficientViTSamPredictor
    from freecure.face_parsing.efficientvit.sam_model_zoo import create_sam_model
    import supervision as sv
except:
    print("YoloWorld can not be load")

def build_yolo_segment_model(sam_path, device):
    yolo_world = YOLOWorld(model_id="yolo_world/l")
    sam = EfficientViTSamPredictor(
        create_sam_model(name="xl1", weight_url=sam_path).to(device).eval()
    )
    return yolo_world, sam

# segment anything inference
def sam_run_prediction(segmentmodel, sam, image, TEXT_PROMPT, segmentType, confidence = 0.2, threshold = 0.5):
    if segmentType=='GroundingDINO':
        image_source, image = load_image_dino(image)
        boxes, logits, phrases = predict(
            model=segmentmodel,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=0.3,
            text_threshold=0.25
        )
        sam.set_image(image_source)
        H, W, _ = image_source.shape
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

        transformed_boxes = sam.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).cuda()
        masks, _, _ = sam.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        masks=masks[0].squeeze(0)
    else:
        image_source = load_image_yoloworld(image)
        segmentmodel.set_classes([TEXT_PROMPT])
        results = segmentmodel.infer(image_source, confidence=confidence)
        detections = sv.Detections.from_inference(results).with_nms(
            class_agnostic=True, threshold=threshold
        )
        masks = None
        if len(detections) != 0:
            sam.set_image(image_source, image_format="RGB")
            masks, _, _ = sam.predict(box=detections.xyxy[0], multimask_output=False)
            masks = torch.from_numpy(masks.squeeze())

    return masks