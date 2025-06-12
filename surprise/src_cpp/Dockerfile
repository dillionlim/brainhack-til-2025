FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

COPY Crow-1.2.1-Linux.deb /tmp/

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      libopencv-dev \
      libssl-dev \
      zlib1g-dev \
      libboost-dev \
      libasio-dev \
 && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      dpkg \
      apt-transport-https \
      ca-certificates \
 && dpkg -i /tmp/Crow-1.2.1-Linux.deb \
 && apt-get install -y -f \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN mkdir build \
 && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=Release \
 && make -j$(nproc)

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      libopencv-dev \
      libasio-dev \
      libboost-dev \
      libssl3 \
      zlib1g \
      curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/build/surprise .

EXPOSE 5005

ENTRYPOINT ["./surprise"]
