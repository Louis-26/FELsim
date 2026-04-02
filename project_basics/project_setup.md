# Docker
## step 1
download Docker
## step 2
```bash
docker compose pull
docker compose up --build
```

# run locally
## step 1: configure environment
```bash
cd $(git rev-parse --show-toplevel)/backend/
if ! conda info --envs | grep -q "^felsim "; then
    conda create -n felsim python=3.11 -y
    conda activate felsim
    # pip install "huggingface-hub<1.0,>=0.34.0"
    # pip install filelock pyyaml regex requests safetensors tokenizers
    pip install -r requirements.txt
else 
    conda activate felsim
fi

uvicorn felAPI:app --host=127.0.0.1 --port=8000 --reload
```
make sure you installed Node.js, verify:  

open a new terminal
```bash
conda activate felsim
cd $(git rev-parse --show-toplevel)/fel-app/
npm install
npm run dev
```