From ubuntu:16.04

RUN apt-get clean
RUN apt-get update -y
RUN apt-get install build-essential -y 
RUN apt-get install vim wget curl git zip -y
RUN apt-get install gcc make openssl libssl-dev libbz2-dev libreadline-dev libsqlite3-dev -y
RUN git clone https://github.com/yyuu/pyenv.git /root/.pyenv
ENV HOME  /root/
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv --version
RUN pyenv install 3.6.5
RUN pyenv global 3.6.5
RUN pyenv rehash
RUN pip install --upgrade pip
RUN pip install tensorflow keras pandas opencv-python \
    sklearn matplotlib jupyter Cython pillow
RUN echo 'alias jupyter-notebook="jupyter-notebook --allow-root --ip 0.0.0.0"' >> ~/.bashrc
