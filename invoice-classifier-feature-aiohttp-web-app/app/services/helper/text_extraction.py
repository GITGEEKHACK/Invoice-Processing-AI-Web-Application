import numpy as np
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import cv2
from app.services.helper.text_postprocessing import TextProcessor
from app.constants import Path


class TextExtractor:
    """
    The TextExtractor class provides a method to crop the image based on the detected coordinates and label ids.
    It uses the OCRService and TextExtractor classes to extract text from the cropped images.

    Methods:

    expand(cords, margin): Returns a list of coordinates by expanding the given coordinates by the given margin value.
    extract_text(results, img): Takes the result of object detection and the image as input and returns a dictionary
    containing extracted text and its label name and confidence score.
    Attributes:

    ocr_service: An instance of the OCRService class.
    handle_text: An instance of the TextProcessor class.
    """
    def __init__(self):
        self.ocr_service = OCRService()
        self.text_processor = TextProcessor()

    async def expand(self, cords, margin):
        # suppose cords is x1, y1, x2, y2
        return [
            cords[0] - margin,
            cords[1] - margin,
            cords[2] + margin,
            cords[3] + margin
        ]

    async def extract_text(self, results, img):
        dic = {}
        df = results.pandas().xyxy[0]
        df = df.loc[df.groupby('name')['confidence'].idxmax()]

        for idx in range(len(df)):
            label_name = df.iloc[idx][6]
            label_confidence = float(round(df.iloc[idx][4], 2))
            cords = df.iloc[idx][0:4]
            label_id = df.iloc[idx][5]

            if label_id == 2:
                cords_exp = await self.expand(cords, margin=4)
                img_crop = img.crop((tuple(cords_exp)))
                text, _ = await self.ocr_service.tr_ocr(img_crop=img_crop, label_id=label_id)
                merchant_text = await self.text_processor.handle_text(text, label_id)
                dic[label_name] = [merchant_text, label_confidence]

            elif label_id == 0:
                cords_exp = await self.expand(cords, margin=5)
                img_crop = img.crop((tuple(cords_exp)))
                text, _ = await self.ocr_service.tr_ocr(img_crop=img_crop, label_id=label_id)
                invoice_text = await self.text_processor.handle_text(text, label_id)
                dic[label_name] = [invoice_text, label_confidence]

            elif label_id == 1:
                cords_exp = await self.expand(cords, margin=6)
                img_crop = img.crop((tuple(cords_exp)))
                text, _ = await self.ocr_service.tr_ocr(img_crop=img_crop, label_id=label_id)
                amount_text = await self.text_processor.handle_text(text, label_id)
                dic[label_name] = [amount_text, label_confidence]

        return dic


class OCRService:

    """
    The OCRService class is responsible for performing OCR on an image using the microsoft/trocr-base-printed model from
    the Hugging Face Transformers library.

    Attributes:
    processor: TrOCRProcessor object, used for preprocessing images before feeding them to the OCR model.
    model_ocr: VisionEncoderDecoderModel object, the OCR model that generates text from an image.

    Methods:
    tr_ocr(img_crop, label_id): An asynchronous method that takes an image crop and a label ID as input.
    The img_crop argument should be a NumPy array representing an image crop, and label_id is an integer representing
    the label ID of the corresponding OCR label. This method performs OCR on the image crop by first preprocessing it using
    the processor, and then feeding it to the OCR model. The generated text is returned along with the label_id.
    """
    def __init__(self):
        self.processor = TrOCRProcessor.from_pretrained(Path.TR_OCR_PATH)
        self.model_ocr = VisionEncoderDecoderModel.from_pretrained(Path.TR_OCR_PATH)

    async def tr_ocr(self, img_crop, label_id):
        image = np.asarray(img_crop)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if len(image.shape) > 2 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model_ocr.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text, label_id
