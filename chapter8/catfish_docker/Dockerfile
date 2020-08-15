ARG port=8080

FROM continuumio/miniconda3:latest

#ARG port
#ARG host

#Neu
#WORKDIR /app

ENV CATFISH_PORT=$port
#ENV CATFISH_HOST=$host

#### requests
RUN conda install -y flask && conda install -c pytorch torchvision && conda install -y requests pillow waitress
RUN mkdir -p /app

COPY ./catfish_model.py /app
# ./
COPY ./catfish_server.py /app
# ./
#COPY ./run-model-service.sh /
#RUN chmod 644 ./catfish_server.py

#EXPOSE 5001
EXPOSE $port

#ENTRYPOINT [ "python" ]
# Neu
CMD ["python", "./catfish_server.py"]
# /home/marcus/Uebersezungen/beginners-pytorch-deep-learning/chapter8/catfish_docker/app/

#CMD ["./app/catfish_server.py"]
# ENTRYPOINT ["/run-model-service.sh"]


