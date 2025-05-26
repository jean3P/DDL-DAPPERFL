# setup.sh
#!/usr/bin/env bash
conda create -n ddl-env python=3.9 -y
conda install -n ddl-env -c conda-forge openmpi compilers -y
conda run -n ddl-env pip install -r requirements.txt
echo "Activating environment 'ddl-env'..."
conda init
conda activate ddl-env
