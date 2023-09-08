"""
The ImagePredictor class contains methods for predicting the content of an uploaded image file. It uses a YOLOv5 model
to detect various regions of interest (ROIs) within the image, and then uses an ImageCropMaker object to extract the
relevant text from each ROI.

Methods:
predict_image(file): Given a file, this method saves it to the Path.UPLOAD_PATH directory, and uses the YOLOv5 model to
predict the ROIs within the image. It then saves the resulting image with the predicted ROIs to the Path.PREDICT_PATH
directory, and extracts the relevant text from each ROI using the TextExtractor object.

`handle_request(request)`: This method handles the image prediction request and returns a tuple of image names and prediction results.
- First, it retrieves the uploaded files from the request object and raises an error if no image files are uploaded.
- Then, it calls `predict_image()` method on each file and stores the results in a list.
- Finally, it returns a tuple of image names and prediction results.
"""
import io
import os
from PIL import Image
import torch
from app.services.helper.text_extraction import TextExtractor
from app.constants import Path, Dimensions, Message, Labels
from app.services.helper.classifier import Classifier


class ImagePredictor:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5:v6.0', 'custom', path=Path.MODEL_PATH)
        self.text_extractor = TextExtractor()
        self.classifier = Classifier()

    async def predict_image(self, file):
        img_path = os.path.join(Path.UPLOAD_PATH, file.filename)
        img = Image.open(img_path)
        results = self.model(img)
        results.save(Path.PREDICT_PATH)

        image_name = os.path.splitext(file.filename)[0]
        pred_img_path = os.path.join(Path.PREDICT_PATH, image_name + '.jpg')
        pred_image = Image.open(pred_img_path).resize((Dimensions.IMAGE_WIDTH, Dimensions.IMAGE_HEIGHT))
        pred_image.save(pred_img_path)

        extracted_invoice_info = await self.text_extractor.extract_text(results=results, img=img)

        img = img.resize((Dimensions.IMAGE_WIDTH, Dimensions.IMAGE_HEIGHT))
        img.save(img_path)
        return image_name, extracted_invoice_info

    async def handle_request(self, request):
        data = await request.post()
        files = [file for file in data.getall('file') if file.content_type.startswith('image/')]

        image_names, prediction_results, results_invoice, results_other = [], [], [], []

        for file in files:
            img_bytes = file.file.read()
            img = Image.open(io.BytesIO(img_bytes))
            img_path = os.path.join(Path.UPLOAD_PATH, file.filename)
            img.save(img_path)

            is_invoice = self.classifier.doc_classifier(img_path)

            if is_invoice == Labels.INVOICE:
                results_invoice.append(await self.predict_image(file))
                image_names.append(results_invoice[-1][0])
                prediction_results.append(results_invoice[-1][1])
            else:
                img = img.resize((Dimensions.IMAGE_WIDTH, Dimensions.IMAGE_HEIGHT))
                img.save(img_path)
                image_name, msg = file.filename, Message.MESSAGE
                results_other.append((image_name, msg))

        return image_names, prediction_results, results_other
