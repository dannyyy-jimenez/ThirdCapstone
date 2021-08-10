FROM python:3.7
WORKDIR /backend
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
ENV PATH="/opt/venv/bin:$PATH"
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc libsndfile1
EXPOSE 8080
COPY . .
CMD ["gunicorn", "wsgi:app", "-w 2", "-b 0.0.0.0:8080", "-t 30"]
