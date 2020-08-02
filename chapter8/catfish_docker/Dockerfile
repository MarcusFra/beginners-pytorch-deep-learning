#ARG port=8080

FROM continuumio/miniconda3:latest

ARG port

ENV CATFISH_PORT=$port
#### requests
RUN conda install -y flask && conda install -c pytorch torchvision && conda install -y requests pillow waitress
RUN mkdir -p /app

COPY ./catfish_model.py /app
COPY ./catfish_server.py /app
#COPY ./run-model-service.sh /

EXPOSE $port

ENTRYPOINT [ "python" ]
# Neu
CMD ["./app/catfish_server.py"]
# ENTRYPOINT ["/run-model-service.sh"]


