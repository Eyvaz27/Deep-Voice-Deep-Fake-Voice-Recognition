FROM python:3.12.7-slim

RUN python -m pip install --no-cache-dir pipenv
RUN apt-get update  
RUN apt-get install zip --assume-yes
RUN apt-get install wget --assume-yes

WORKDIR /app
COPY Pipfile /app/Pipfile
COPY Pipfile.lock /app/Pipfile.lock
RUN pipenv install --system --deploy

RUN cd /app && mkdir deepvoice_project
RUN cd /app && mkdir dataset
ADD /src/ /app/deepvoice_project/
ADD /dataset/ /app/dataset/

EXPOSE 9696

WORKDIR /app/deepvoice_project/
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "server:app"]