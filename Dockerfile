# syntax=docker/dockerfile:1
FROM python:3.13-slim

RUN apt update && apt install -y git && \
    apt-get install -y locales && \
    sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    dpkg-reconfigure --frontend=noninteractive locales
WORKDIR /nightjar
COPY pyproject.toml /nightjar/pyproject.toml
# Install dependencies without installing the package itself
RUN pip install $(python -c "import tomllib; data=tomllib.load(open('pyproject.toml', 'rb')); deps=data['project']['dependencies'] + data['project']['optional-dependencies']['research']; print(' '.join(deps))")
COPY benchmarks/words /usr/share/dict/words
COPY LICENSE /nightjar/LICENSE
COPY README.md /nightjar/README.md
COPY src /nightjar/src
RUN pip install .
COPY scripts /nightjar/scripts
COPY benchmarks /nightjar/benchmarks
COPY benchmarks_cpython /nightjar/benchmarks_cpython

CMD ["/bin/bash"]