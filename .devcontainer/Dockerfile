# Use the specified NVIDIA CUDA image as a parent image
FROM nvcr.io/nvidia/tensorflow:24.08-tf2-py3

# Set environment variable for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Install software-properties-common to use add-apt-repository
RUN apt-get update && apt-get install -y software-properties-common

# Install additional tools
RUN apt-get update && apt-get install -y wget curl gnupg apt-transport-https

# Install LaTeX packages
RUN apt-get install -y texlive-full texlive-xetex texlive-latex-extra

# Add the Microsoft GPG key and install VSCode CLI
RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add - \
    && add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" \
    && apt-get update \
    && apt-get install -y code
    
# Add deadsnakes PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    python3-apt

# Set timezone to Brisbane non-interactively
RUN ln -fs /usr/share/zoneinfo/Australia/Sydney /etc/localtime && \
    echo "Australia/Sydney" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

# Install Miniconda
RUN mkdir -p /opt/miniconda3 && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda3/miniconda.sh && \
bash /opt/miniconda3/miniconda.sh -b -u -p /opt/miniconda3 && \
rm /opt/miniconda3/miniconda.sh
ENV PATH="/opt/miniconda3/bin:$PATH"

# Copy environment.yml and create conda environment
COPY environment.yml environment.yml
RUN conda env create -f environment.yml

# # Activate conda base environment and install requirements
# COPY requirements.txt /app/requirements.txt
# RUN /bin/bash -c "source /opt/miniconda3/bin/activate && conda init bash && pip3 install -r /app/requirements.txt"


# Clean up
RUN rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

RUN pip install jupyter ipython  ipywidgets matplotlib plotly bokeh nbconvert

RUN pip install --user ipywidgets ipympl nbconvert ipykernel nbformat

# Set up a virtual environment with Python 3.11
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# # Install additional tools
# RUN apt-get update && apt-get install -y wget curl gnupg apt-transport-https

# # Add the Microsoft GPG key and install VSCode CLI
# RUN wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | apt-key add - \
#     && add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main" \
#     && apt-get update \
#     && apt-get install -y code

# Clean up apt cache to reduce image size
RUN rm -rf /var/lib/apt/lists/*