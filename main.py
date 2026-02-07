import sys
import logging
import os
from pathlib import Path
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Determine data directory - use /data if writable, otherwise use local data/
def get_data_dir():
    data_dir = Path("/data")
    if not os.access(data_dir, os.W_OK):
        data_dir = Path("data")
    return data_dir


def run_command(cmd, cwd=None, description=None, env=None):
    if description:
        logger.info(f"Starting: {description}")
    
    try:
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        subprocess.run(cmd, cwd=cwd, check=True, capture_output=False, env=run_env)
        if description:
            logger.info(f"Completed: {description}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run command: {' '.join(cmd)}")
        logger.error(f"Return code: {e.returncode}")
        return False


def download_data():
    data_dir = get_data_dir()
    corpus_path = data_dir / "corpus.txt.gz"
    
    if corpus_path.exists():
        logger.info(f"Data already exists at {corpus_path}")
        return True
    
    if not run_command(
        [sys.executable, "data/gatherData.py"],
        cwd=None,
        description="Download Wikipedia dataset",
        env={"DATA_DIR": "/data"}
    ):
        return False
    
    if not corpus_path.exists():
        logger.error(f"Data download failed - {corpus_path} not found")
        return False
    
    return True


def preprocess_data():
    data_dir = get_data_dir()
    corpus_clean_path = data_dir / "corpus_clean.txt.gz"
    
    if corpus_clean_path.exists():
        logger.info(f"Preprocessed data already exists at {corpus_clean_path}")
        return True
    
    if not run_command(
        [sys.executable, "data/preprocess.py"],
        cwd=None,
        description="Preprocess and clean data",
        env={"DATA_DIR": "/data"}
    ):
        return False
    
    if not corpus_clean_path.exists():
        logger.error(f"Data preprocessing failed - {corpus_clean_path} not found")
        return False
    
    return True


def train_tokenizer():
    model_dir = Path("model")
    tokenizer_path = model_dir / "tokenizer.json"
    
    if tokenizer_path.exists():
        logger.info(f"Tokenizer already exists at {tokenizer_path}")
        return True
    
    if not run_command(
        [sys.executable, "model/tokenizer.py"],
        cwd=None,
        description="Train BPE tokenizer"
    ):
        return False
    
    if not tokenizer_path.exists():
        logger.error(f"Tokenizer training failed - {tokenizer_path} not found")
        return False
    
    return True


def train_model():
    if not run_command(
        [sys.executable, "model/train.py"],
        cwd=None,
        description="Train MLM model"
    ):
        return False
    
    return True


if not Path("data").exists() or not Path("model").exists():
    logger.error("Error: data/ and model/ directories not found")
    logger.error("Please run this script from the CDiffusion root directory")

steps = [
    ("Download Data", download_data),
    ("Preprocess Data", preprocess_data),
    ("Train Tokenizer", train_tokenizer),
    ("Train Model", train_model),
]

for step_name, step_func in steps:
    try:
        if not step_func():
            logger.error(f"Pipeline failed at: {step_name}")
            exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during {step_name}: {e}")
        exit(1)