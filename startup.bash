wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.10.0-1-Linux-x86_64.sh
bash Miniconda3-py310_23.10.0-1-Linux-x86_64.sh
source ~/.bashrc
conda create -n seldo python=3.10 py
conda activate seldo
pip install -r requirements.txt
echo "Installed seldo conda environment successfully"
