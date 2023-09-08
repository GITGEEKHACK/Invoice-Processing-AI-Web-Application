"""
This module contains two functions, 'home' and 'predict', used to handle HTTP requests related to the invoice app.

Functions:

home(request): Renders the index.html template with an empty context when a GET request is made to the home route.
predict(request): Handles the image prediction request made to the predict route using the ImagePredictor object from the invoice_app module.
Renders the predict.html template with the predicted results when the prediction is successful.
Renders the index.html template with an error message when the prediction fails.
"""

import aiohttp_jinja2
from app.services.invoice_app import ImagePredictor


async def home(request):
    response = aiohttp_jinja2.render_template('index.html', request, context={})
    return response


async def predict(request):
    image_predictor = ImagePredictor()  # get the ImagePredictor object
    try:
        base_name_list, my_dict_list, other_image_name = await image_predictor.handle_request(request)
        context = {'base_name': base_name_list, 'my_dict': my_dict_list, 'other_image_name': other_image_name}
        response = aiohttp_jinja2.render_template('predict.html', request, context=context)
        return response
    except Exception as error:
        response = aiohttp_jinja2.render_template('index.html', request, context={'error': type(error).__name__})
        return response


