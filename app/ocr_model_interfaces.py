import pytesseract
from PIL import Image
import numpy as np
import logging
from abc import ABC, abstractmethod
import easyocr
import cv2
from paddleocr import PaddleOCR



logger = logging.getLogger(__name__)

class ocr_interface(ABC):
    @abstractmethod
    def ocr(self, *args, **kwargs):
        pass

class ocr_tesseract_interface(ocr_interface):
    def __init__(self, *args, **kwargs):
        pass  # pytesseract does not require initialization like easyocr

    def ocr(self, temp_path, *args, **kwargs):
        image = Image.open(temp_path)
        outs = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        if len(outs['text']) == 0:
            return [""]
        logger.debug(outs)
        confs = [int(conf) for conf in outs['conf'] if conf != '-1']
        if not confs:
            return [""]
        arg_max_conf = np.argmax(confs)
        return [outs['text'][arg_max_conf]]

class ocr_easy_interface(ocr_interface):
    def __init__(self, *args, **kwargs):
        self.reader = easyocr.Reader(['en'])

    def ocr(self, temp_path, *args, **kwargs):
        outs = self.reader.readtext(temp_path)
        if len(outs) == 0:
            return [""]
        logger.debug(outs)
        confs = [out[-1] for out in outs]
        arg_max_conf = np.argmax(confs)
        return [outs[arg_max_conf][-2]]

class ocr_easyR_interface(ocr_interface):
    def __init__(self, *args, **kwargs):
        self.reader = easyocr.Reader(['en'], recog_network='standard')  # Only using the recognition part

    def ocr(self, temp_path, *args, **kwargs):
        outs = self.reader.recognize(temp_path)  # Directly using the recognize method
        if len(outs) == 0:
            return [""]
        logger.debug(outs)
        confs = [out[-1] for out in outs]
        arg_max_conf = np.argmax(confs)
        return [outs[arg_max_conf]]
        
class ocr_llm_interface(ocr_interface):
    def __init__(self, processor, ocr_model, device):
        self.processor = processor
        self.ocr_model = ocr_model
        self.device = device

    def ocr(self, cropped_image, *args, **kwargs):
        # returns = []
        with torch.no_grad():
            pixel_values = self.processor(images=cropped_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            generated_ids = self.ocr_model.generate(pixel_values)
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            # print("OCR results: ", generated_text)
            return generated_text


class ocr_easyRA_interface(ocr_interface):
    def __init__(self, *args, **kwargs):
        self.reader = easyocr.Reader(['en'], recog_network='standard')  # Only using the recognition part

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        return image

    def ocr(self, temp_path, *args, **kwargs):
        preprocessed_image = self.preprocess_image(temp_path)
        outs = self.reader.recognize(preprocessed_image, detail=0)  # Directly using the recognize method
        if len(outs) == 0:
            return [""]
        return [outs[0]]


class ocr_paddle_interface(ocr_interface):
    def __init__(self, *args, **kwargs):
        # Initialize PaddleOCR with English language
        self.paddle = PaddleOCR(use_angle_cls=True, lang='en')

    def ocr(self, temp_path, *args, **kwargs):
        # Read the image
        image = Image.open(temp_path)
        # Perform OCR using PaddleOCR
        results = self.paddle.ocr(np.array(image), cls=True)
        print("resutls", results)
        if results[0] == None:
            return [""]
        
        logger.debug(results)
        
        # Extracting the text and confidence scores
        texts = [line[1][0] for result in results for line in result]
        confs = [line[1][1] for result in results for line in result]
        
        if not confs:
            return [""]
        
        # Find the index of the maximum confidence
        arg_max_conf = np.argmax(confs)
        
        # Return the text with the highest confidence
        return [texts[arg_max_conf]]