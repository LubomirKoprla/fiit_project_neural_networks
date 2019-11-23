# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2019 Slovak University of Technology. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG UBUNTU_VERSION=18.04
FROM 88178d65d12c

RUN apt-get update && apt-get install -y curl

ARG _PY_SUFFIX=3
ARG PYTHON=python${_PY_SUFFIX}
ARG PIP=pip${_PY_SUFFIX}

RUN apt-get update && apt-get install -y \
    ${PYTHON} \
    ${PYTHON}-pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ${PIP} install jupyter matplotlib
RUN ${PIP} install jupyter_http_over_ws
RUN ${PIP} install ipython==7.9.0
RUN jupyter serverextension enable --py jupyter_http_over_ws

RUN ${PIP} install numpy
RUN ${PIP} install pandas
RUN ${PIP} install keras
RUN ${PIP} install tensorflow

