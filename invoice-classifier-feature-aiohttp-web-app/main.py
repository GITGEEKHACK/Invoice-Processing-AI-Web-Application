"""
This module starts the web application using aiohttp.
"""

from aiohttp import web
from app import app

if __name__ == '__main__':
    web.run_app(app, host='0.0.0.0')
