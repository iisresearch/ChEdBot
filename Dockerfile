FROM python:3.11

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app
 
WORKDIR $HOME/app

COPY --chown=user . $HOME/app

CMD ["chainlit", "run", "app.py", "--port", "7860", "--no-cache"]