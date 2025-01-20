Run the following commands in your terminal:
```bash
# Create and activate Conda environment
conda create -n safety_rules_following-dev python=3.10 -y
conda activate safety_rules_following-dev

# Install PyTorch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt