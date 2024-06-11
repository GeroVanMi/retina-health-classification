FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ADD data/Eyes /app/data/Eyes

RUN apt update && \
  apt install git python3.10 python3.10-venv python3-pip apt-transport-https ca-certificates gnupg curl  -y 

ENV VIRTUAL_ENV=/app/venv
RUN python3 -m venv ${VIRTUAL_ENV} --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && apt update -y && apt install google-cloud-sdk -y

WORKDIR /app

ADD pyproject.toml /app/
ADD src/__init__.py /app/src/
RUN pip install --upgrade pip && \
  pip install -e . &&\
  pip cache purge

ADD src /app/src

CMD ["bash"]
