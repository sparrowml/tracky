FROM python:3.8

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" && echo $SNIPPET >> "/root/.bashrc"

RUN pip install poetry==1.1.6

WORKDIR /code
RUN mkdir tracky
RUN touch tracky/__init__.py

COPY poetry.* pyproject.toml /code/
RUN poetry install
