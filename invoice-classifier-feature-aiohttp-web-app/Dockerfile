FROM python:3.8
WORKDIR /home/kelvin/Desktop/temp_invoice
COPY . .
RUN apt-get update
RUN apt install -y libgl1-mesa-glx
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
CMD ["python3.8", "main.py"]
