import os
import sys
import argparse
from pathlib import Path
import urllib.request
from tqdm import tqdm

def setup_kaggle_api():
    """Instructions for setting up Kaggle API key"""
    print("--- Kaggle API Setup Instructions ---")
    print("1. Create an account on Kaggle (https://www.kaggle.com)")
    print("2. Go to 'Account' and click 'Create New API Token'")
    print("3. A 'kaggle.json' file will be downloaded")
    if os.name == 'nt':
        print("4. Move it to: C:\\Users\\<Username>\\.kaggle\\kaggle.json")
    else:
        print("4. Move it to: ~/.kaggle/kaggle.json")
    print("-------------------------------------")

def download_ucf_crime(target_dir):
    """Downloads UCF-Crime dataset from Kaggle"""
    try:
        import kaggle
    except OSError as e:
        print("Kaggle API key not found. Please set it up first.")
        setup_kaggle_api()
        return False
    except ImportError:
        print("Kaggle library not installed. Install with: pip install kaggle")
        return False
        
    print("Downloading UCF-Crime dataset...")
    kaggle.api.dataset_download_cli("odins0n/ucf-crime-dataset", path=target_dir, unzip=True)
    print(f"Dataset downloaded to {target_dir}")
    return True

def download_mot17(target_dir):
    """Downloads MOT17 dataset (placeholder implementation)"""
    print("MOT17 dataset download is manual. Please visit https://motchallenge.net/")
    return False

def download_sample_videos(target_dir):
    """Fallback: Downloads sample surveillance videos from public URLs"""
    print("Downloading sample videos...")
    
    samples = {
        "sample_normal.mp4": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4",  # Placeholder
        "sample_pedestrian.mp4": "https://test-videos.co.uk/vids/bigbuckbunny/mp4/h264/720/Big_Buck_Bunny_720_10s_1MB.mp4" # Placeholder
    }
    
    os.makedirs(target_dir, exist_ok=True)
    
    for filename, url in samples.items():
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                # Custom hook for tqdm progress bar
                class DownloadProgressBar(tqdm):
                    def update_to(self, b=1, bsize=1, tsize=None):
                        if tsize is not None:
                            self.total = tsize
                        self.update(b * bsize - self.n)
                        
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urllib.request.urlretrieve(url, filename=filepath, reporthook=t.update_to)
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"{filename} already exists.")
            
    print("Sample downloads complete.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download datasets for Sentinel AI")
    parser.add_argument("--dataset", type=str, choices=["ucf_crime", "mot17", "samples"], default="samples", help="Dataset to download")
    args = parser.parse_args()
    
    # Create directory structure
    base_dir = Path(__file__).parent.parent
    raw_dir = base_dir / "data" / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    if args.dataset == "ucf_crime":
        ucf_dir = raw_dir / "ucf_crime"
        download_ucf_crime(str(ucf_dir))
    elif args.dataset == "mot17":
        mot_dir = raw_dir / "mot17"
        download_mot17(str(mot_dir))
    else:
        sample_dir = raw_dir / "samples"
        download_sample_videos(str(sample_dir))
