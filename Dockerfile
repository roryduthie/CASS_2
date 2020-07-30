FROM python:3.7.4

RUN mkdir -p /home/cass
WORKDIR /home/cass

RUN pip install --upgrade pip
RUN apt-get install -y git

ADD requirements.txt .
RUN pip install -r requirements.txt
RUN git clone https://github.com/Jacobe2169/GMatch4py
WORKDIR /home/cass/GMatch4py
RUN pip install .
WORKDIR /home/cass
RUN pip install gunicorn

ADD app app
ADD boot.sh ./
RUN chmod +x boot.sh

ENV FLASK_APP cass.py

EXPOSE 8200
ENTRYPOINT ["./boot.sh"]