FROM debian:bullseye-slim AS build-cpp

# install dependencies
RUN apt update && apt install -y \
  # common build dependencies
  bash git curl ccache cmake \
  # c++ build dependencies
  build-essential libc6-dev \
  && apt clean && rm -rf /var/lib/apt/lists/*

# build inference repo
COPY . /src
RUN cd /src && \
    bash ./build.sh

FROM debian:bullseye-slim AS runtime

# install dependencies
RUN apt update && apt install -y \
  bash \
  && apt clean && rm -rf /var/lib/apt/lists/*

# copy built inference binaries
COPY --from=build-cpp /src/build/starcoder-server /usr/local/bin/starcoder-server

# set up main to run starcoder server with provided arguments
ENTRYPOINT ["/usr/local/bin/starcoder-server"]