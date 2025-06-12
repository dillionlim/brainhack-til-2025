# C++ Implementation of Python Algorithm

## Local Testing

This was tested on an Ubuntu 24.04, it is not guaranteed to work with other distros / operating systems.

### Installation

1. Install required libraries:

```bash
sudo apt install libopencv-dev libssl-dev zlib1g-dev libboost-dev libasio-dev
```

These include the necessary dependencies for both openCV2 and Crow, which will be our HTTP server of choice.

2. Install Crow. There are a few ways to achieve this, as highlighted in [the crow installation documentation](https://crowcpp.org/master/getting_started/setup/linux/). In testing, it was installed through downloading the `.deb` file directly, using:

```bash
sudo dpkg -i Crow-1.2.1-Linux.deb
```

### Compilation

```bash
mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running

```bash
./surprise
```

## Dockerfile

### Building Dockerfile

The Dockerfile installs it using the `.deb` file, which is assumed to be in the root directory.

```bash
docker build -t <team_name>-surprise .
```

### Running Dockerfile

```bash
docker run -p 5005:5005 <team_name>-surprise
```