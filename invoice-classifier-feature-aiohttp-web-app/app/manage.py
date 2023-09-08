"""
This module provides a function to create and configure an aiohttp web application.

The create_app() function initializes and returns a web.Application instance with the following configurations:

Sets a client max size of 10MB to limit the size of uploaded files.
Sets up Jinja2 templates for rendering HTML pages using the aiohttp_jinja2 library.
Configures the static root URL to be '/static'.

"""

import aiohttp_jinja2
import jinja2
from aiohttp import web
from app.constants import Configuration


def create_app():
    app = web.Application(client_max_size=Configuration.MAX_IMAGE_SIZE)  #This means that the server will only accept requests with a payload (e.g., uploaded files) of up to 10MB in size.
    jinja2_env = aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('app/templates/'))
    app['static_root_url'] = '/static'
    return app
