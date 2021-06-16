FROM python:3.7-alpine

MAINTAINER Manikant Kumar "mani2474695@gmail.com"

COPY . /app

WORKDIR /app

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]

