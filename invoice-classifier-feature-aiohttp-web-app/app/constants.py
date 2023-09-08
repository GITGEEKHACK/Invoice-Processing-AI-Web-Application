"""
This class contains static paths used by the invoice classifier web application.
"""


class Path:
    UPLOAD_PATH = './app/static/upload'
    PREDICT_PATH = './app/static/predict'
    MODEL_PATH = './app/model/invoice_classifier.pt'
    DISTILBERT_PATH = './app/model/distilbert-model'
    TR_OCR_PATH = './app/model/tr-ocr-model'
    XGB_PATH = './app/model/embed-XGB-DOC-CLF.pkl'
    PPOCR_RECN = './app/model/.paddleocr/whl/rec/en/en_PP-OCRv3_rec_slim_infer'
    PPOCR_DETN = './app/model/.paddleocr/whl/det/en/en_PP-OCRv3_det_slim_infer'


class Dimensions:
    IMAGE_WIDTH = 500
    IMAGE_HEIGHT = 700


class Regex:
    DATE_PATTERN = [
        "[0-9]{2}/{1}[0-9]{2}/{1}[0-9]{4}",
        "\\d{1,2}-(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)-\\d{4}",
        "\\d{4}-\\d{1,2}-\\d{1,2}",
        "[0-9]{1,2}\\s(January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sept|October|Oct|November|Nov|December|Dec)\\s\\d{4}",
        "\\d{1,2}-\\d{1,2}-\\d{2,4}"]
    CLEANED_TEXT = r'[a-z]:\s'
    CAPTURES_PATTERN = r'\d+'
    DATE_FORMAT_1 = "%Y-%m-%d"
    DATE_FORMAT_2 = "%Y-%d-%m"


class Labels:
    DATE_LABEL = 'DATE'
    AMOUNT_LABEL = ['MONEY', 'CARDINAL']
    INVOICE = 0
    OTHER = 1


class Threshold:
    THRESHOLD_VALUE = 0.96


class Paddleocr:
    REC_BATCH_NUM = 2
    REC_ALGORITHM = 'CRNN'


class Configuration:
    MAX_IMAGE_SIZE = 1024 * 1024 * 10


class Message:
    MESSAGE = "This is not an Invoice!!"
