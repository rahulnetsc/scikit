from pathlib import Path
import tarfile
import pandas as pd
import urllib.request as urlreq
import argparse 

from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_configs(path):
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def load_data(url,csv_path)-> pd.DataFrame:
    tgz_path = Path("data/raw/raw.tgz")
    
    if not csv_path.is_file():
        Path("data/raw").mkdir(parents=True, exist_ok= True)
        try:
            urlreq.urlretrieve(url,tgz_path)
        except Exception as e:
            logger.error(f"Error downloading from {url}: {e}")
            raise RuntimeError(f"Error downloading from {url}: {e}")
        try:
            with tarfile.open(tgz_path) as housing_tarball:
                housing_tarball.extractall(path="data/raw")
                tgz_path.unlink()  # deletes raw.tgz
        except Exception as e:
            raise RuntimeError(f"Error extracting tarfile {tgz_path}: {e}")
        
    logger.info(f"Loading dataset from {csv_path}")
    return pd.read_csv(csv_path)

    
