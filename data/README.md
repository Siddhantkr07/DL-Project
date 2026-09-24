# Dataset Documentation

This directory contains scripts for downloading and loading the datasets used for training and evaluating Sentinel AI.

## Supported Datasets

1. **UCF-Crime Dataset**
   - Contains 1900 long surveillance videos covering 13 real-world anomalies.
   - Categories include: Abuse, Arson, Assault, Burglary, Explosion, Fighting, RoadAccidents, Robbery, Shooting, Shoplifting, Stealing, Vandalism, and Normal.

2. **MOT17 (Multiple Object Tracking)**
   - Used for evaluating the tracking module (DeepSORT).

## How to Download UCF-Crime

1. Setup Kaggle API:
   - Go to [Kaggle](https://www.kaggle.com/) -> Account -> Create New API Token.
   - Place the downloaded `kaggle.json` in your `.kaggle` directory.
     - Windows: `C:\Users\<YourUsername>\.kaggle\kaggle.json`
     - Linux/Mac: `~/.kaggle/kaggle.json`

2. Run the download script:
   ```bash
   python data/download_dataset.py --dataset ucf_crime
   ```

## Dataset Structure

After downloading, the data should be organized as follows:

```text
data/
├── raw/
│   ├── ucf_crime/
│   │   ├── Abuse/
│   │   │   ├── Abuse001_x264.mp4
│   │   │   └── ...
│   │   ├── Arson/
│   │   └── ... (other categories)
│   ├── mot17/
│   └── samples/
└── processed/
```
