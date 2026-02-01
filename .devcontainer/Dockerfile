# MineAgent dev container: Pixi manages all project dependencies
# (Python, Java 21, Node, Gradle, etc. come from pixi.lock)
FROM mcr.microsoft.com/devcontainers/base:ubuntu-22.04

# Install Pixi system-wide so it's available for the vscode user
RUN curl -fsSL https://pixi.sh/install.sh | bash -s - \
    && mv /root/.pixi/bin/pixi /usr/local/bin/pixi
