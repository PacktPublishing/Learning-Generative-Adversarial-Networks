FROM ubuntu:latest

MAINTAINER kuntalganguly.it@gmail.com

# install python3, nginx, supervisor
RUN apt-get update --fix-missing && apt-get install -y --allow-unauthenticated \
  build-essential \
  git \
  python3 \
  python3-pip \
  nginx \
  supervisor \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/* /tmp/*

# install virtualenv
RUN pip3 install -U pip && pip install virtualenv && rm -rf /tmp/* /root/.cache/pip

# create virtual env and install dependencies
# due to a bug in h5, we need to install Cython first
RUN virtualenv /opt/venv
ADD ./requirements.txt /opt/venv/requirements.txt
RUN /opt/venv/bin/pip install Cython && /opt/venv/bin/pip install -r /opt/venv/requirements.txt && rm -rf /tmp/* /root/.cache/pip
RUN mkdir -p /opt/deep/model && chmod  777 /opt/deep/model/
VOLUME /deep/model

# expose port
EXPOSE 80
EXPOSE 5000

# add config files
ADD ./supervisord.conf /etc/supervisord.conf
ADD ./nginx.conf /etc/nginx/nginx.conf

# copy the service code
ADD ./service /opt/app

# start supervisor to run our wsgi server
CMD supervisord -c /etc/supervisord.conf -n
