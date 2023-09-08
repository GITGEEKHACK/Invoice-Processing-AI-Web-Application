"""
This script is used to create an instance of the web application.

Functions:
- create_app: This function is used to create and configure an instance of the web application.
- home: This function renders the home page of the web application.
- predict: This function renders the predict page of the web application.

Routes:
- GET /: It maps the home function to the home route of the web application.
- POST /predict: It maps the predict function to the predict route of the web application.

Static Files:
- /static: It maps the 'app/static' directory as the static directory of the web application.

"""


from app.manage import create_app
from app.resources.router import home,predict
from aiohttp import web

app = create_app()
app.router.add_static('/static', 'app/static')
app.add_routes([web.get('/', home)])
app.add_routes([web.post('/predict', predict)])

