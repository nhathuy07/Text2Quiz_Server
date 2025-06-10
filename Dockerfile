FROM python:3.10

RUN apt-get update && apt-get install -y libgl1-mesa-glx libxext6 lsb-release curl gpg pandoc tesseract-ocr tesseract-ocr-vie tesseract-ocr-script-viet

RUN curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list

RUN apt-get update && apt-get install -y redis-server

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH


WORKDIR $HOME/app

COPY --chown=user  ./requirements.txt $HOME/app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r $HOME/app/requirements.txt

COPY --chown=user . $HOME/app

RUN mkdir nltk_data
CMD ["sh", "./entrypoint.sh"]
