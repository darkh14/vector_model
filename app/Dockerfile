FROM tensorflow/tensorflow:2.14.0rc1
LABEL authors="dark_h14"

WORKDIR /app

EXPOSE 8060:8060

COPY . .

RUN python3 -m pip install --root-user-action=ignore --upgrade pip && python3 -m  pip install --root-user-action=ignore -r requirements.txt

ENTRYPOINT ["uvicorn", "main_app:app", "--host", "0.0.0.0", "--port", "8060"]