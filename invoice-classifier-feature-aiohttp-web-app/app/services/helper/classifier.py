from paddleocr import PaddleOCR
import pickle
from app.services.helper.embedding_generator import EmbeddingsGenerator
from app.constants import Path, Threshold, Labels, Paddleocr


class Classifier:
    def __init__(self):
        """
        Initialize the Classifier object.

        Attributes:
        embeddings (EmbeddingsGenerator): Object for generating text embeddings.
        paddle_ocr (PaddleOCR): PaddleOCR object for optical character recognition.
        xgb_model (XGBoost model): Pre-trained XGBoost model for document classification.
        """
        self.embeddings = EmbeddingsGenerator()
        self.paddle_ocr = PaddleOCR(lang='en', rec_batch_num=Paddleocr.REC_BATCH_NUM,
                             rec_model_dir=Path.PPOCR_RECN,
                             det_model_dir=Path.PPOCR_DETN,
                             rec_algorithm=Paddleocr.REC_ALGORITHM)
        self.xgb_model = pickle.load(open(Path.XGB_PATH, 'rb'))

    def doc_classifier(self, uploaded_image_path):
        """
        Perform document classification based on OCR results and a pre-trained XGBoost model.

        Args:
            uploaded_image_path (str): Path to the uploaded image file.

        Returns:
            str: Label indicating the document type (Labels.INVOICE or Labels.OTHER).
        """
        result = self.paddle_ocr.ocr(uploaded_image_path)
        result = result[0]
        text_list = [line[1][0] for line in result]
        text = " ".join(text_list)

        text_embd = self.embeddings.generate_text_embeddings([text])
        invoice_prob = (self.xgb_model.predict_proba(text_embd).tolist()[0][0])
        if invoice_prob >= Threshold.THRESHOLD_VALUE:
            return Labels.INVOICE
        else:
            return Labels.OTHER


