from ultralytics import YOLOv10
import glob
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import transformers
import torch
import os
from tqdm import tqdm


from .dataset import get_dataset_loader
from .logger_config import logger  

def get_detection_model(path=None):
    ''' load the last trained model using modification date if path is not set'''
    
    if path is None:
        path = max(glob.iglob('models/*.pt'), key=os.path.getctime)
        logger.info(f"Loading detection model from {path}")
    return YOLOv10(path, verbose=False)

def get_ocr(path=None):
    # Download and load the processor and model with progress bar enabled
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
    model_ocr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
    return processor, model_ocr


def predict(img_paths, detection_model, ocr_model, processor, device='cuda', max_batch_size=32, return_cropped_images=False):
    ''' inference pipeline for detection and OCR '''
    # Detection
    results = detection_model(img_paths)
    all_detections = [result.boxes.xyxy for result in results]

    _, dataloader = get_dataset_loader(img_paths, all_detections, batch_size=max_batch_size)
    ocr_results = []

    cropped_images = []

    for batch in tqdm(dataloader):
        # OCR
        batch = batch.to(device)
        if return_cropped_images:
            cropped_images.extend(batch)
        # logger.info(f"Batch shape: {batch.shape}")
        # print(ocr_model.device, batch.device)
        with torch.no_grad():
            pixel_values = processor(images=batch, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device)
            generated_ids = ocr_model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
            ocr_results.extend(generated_text)
    
    if return_cropped_images:
        return results, ocr_results, cropped_images
    
    return results, ocr_results

def inference_pipe(img_paths, device='cuda', return_cropped_images=False):

    # Load models
    logger.info("Loading detection model")
    detection_model = get_detection_model()
    logger.info("Detection model loaded and moved to device")

    logger.info("Loading OCR model and processor")
    processor, ocr_model = get_ocr()
    ocr_model = ocr_model.to(device)
    logger.info("OCR model and processor loaded and moved to device")

    # Run inference
    logger.info("Running inference")
    output = predict(img_paths, detection_model, ocr_model, processor, return_cropped_images=return_cropped_images)
    logger.info("Inference completed")

    return output