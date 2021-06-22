FROM python:3.7
COPY . /app
WORKDIR /app
RUN apt-get update
RUN pip install .
RUN pip install gunicorn
RUN ls -la
EXPOSE 3000
ENTRYPOINT ["./gunicorn.sh"]