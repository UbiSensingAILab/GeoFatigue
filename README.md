# GeoFatigue

Python tools for working with the **GeoFatigue** dataset — a multimodal physiological and contextual sensing dataset for fatigue monitoring in construction workers. This repository provides:

- Loaders for the raw data formats (Empatica AVRO physiological files, smartphone contextual-sensing CSVs, GeoJSON spatial layouts, session/demographics metadata)
- Code to regenerate **Figure 6** (elevation–physiology composite) **Figure 7** (spatial physiology heatmap) from the paper
- A getting-started notebook demonstrating how to load and explore the dataset

## Dataset Access

The complete GeoFatigue dataset is deposited in the University of Calgary's PRISM Data repository:

**DOI:** [https://doi.org/10.5683/SP3/I0KSNR](https://doi.org/10.5683/SP3/I0KSNR)

The dataset is distributed under the **Creative Commons Attribution–NonCommercial–ShareAlike 4.0 International (CC BY-NC-SA 4.0)** license and is freely accessible without registration. If you use the dataset in your research, please cite both the associated publication and the dataset DOI (see [Citation](#citation)).

### Ethics Statement

The study protocol was reviewed and approved by the University of Calgary Conjoint Faculties Research Ethics Board (CFREB; Ethics ID: REB24-1215). All methods were performed in accordance with relevant guidelines and regulations. All experiments were conducted outdoors on the University of Calgary main campus between June and September 2025, over 27 data collection days.

All participants provided written informed consent include permission for participation and the use of de-identified data in scientific publications and future research. 

## About the Dataset

The GeoFatigue dataset contains multimodal data collected from 40 participants performing manual material handling tasks outdoors:

- **Physiological Data**: High-frequency wearable sensor data from Empatica EmbracePlus devices (AVRO, and CSV)
- **Contextual Sensing Data**: Smartphone IMU, GPS, barometric pressure, magnetometer, and sound data (CSV)
- **Spatial Information (Experiment layout)**: GeoJSON boundaries defining the experiment site layout, and a high-resolution 3D point cloud generated using a terrestrial laser scanner (Trimble X9)
- **Weather Data**: Hourly meteorological observations obtained from the Visual Crossing Weather service ([Visual Crossing Corporation](https://www.visualcrossing.com))
- **Metadata**: Session information, self-reported fatigue level, and participant demographics

Participants performed manual material handling tasks (carrying 20 lbs for 20 minutes) across different terrains — flat trail, stairs, ramp — with a resting area for baseline measurements and recovery.

## Installation

```bash
git clone https://github.com/UbiSensingAILab/GeoFatigue.git
cd GeoFatigue
pip install -r requirements.txt
```

Then copy `.env.template` to `.env` and fill in the local paths to your copy of the dataset:

```bash
cp .env.template .env
```

```
DATA_ROOT=/path/to/empatica/raw/data
METADATA_PATH=/path/to/metadata
PHYS_DATA_PATH=/path/to/physiological/data
CONTEXTUAL_DATA_PATH=/path/to/contextual/sensing/data
SPATIAL_DATA_PATH=/path/to/experiment/layout/data
TIF_DIR=/path/to/qgis/surface/rasters
OUTPUT_DIR=figures
```

## Quick Start

See [`examples/01_getting_started.ipynb`](examples/01_getting_started.ipynb) for a full walkthrough. In short:

```python
from geofatigue.loaders import (
    load_session_metadata,
    load_demographics,
    load_avro_physiological_data,
    load_contextual_sensing_data,
    load_spatial_layout,
)

# Metadata
sessions = load_session_metadata('path/to/session_metadata.json')
demographics = load_demographics('path/to/demographics.csv')

# Physiological data (AVRO)
data = load_avro_physiological_data('path/to/participant.avro')
eda = data['eda']

# Contextual sensing (smartphone CSV)
contextual = load_contextual_sensing_data('path/to/p1-flat_ground.csv')

# Spatial layout (GeoJSON)
layout = load_spatial_layout('path/to/resting area.geojson')
```

## Regenerating Figures 6 and 7

This repository's public figure-plotting API (`geofatigue.figures`) exposes two functions from the paper's figure set:

- `plot_elevation_physiology_composite` — Figure 6 (elevation–physiology composite)
- `plot_spatial_physiology_heatmap` — Figure 7 (spatial physiology heatmap)

Both are driven by `scripts/generate_figures.py`, which reads its input paths from `.env` (see [Installation](#installation)):

```bash
python scripts/generate_figures.py                # both figures
python scripts/generate_figures.py --skip-fig6     # only Figure 7
python scripts/generate_figures.py --skip-fig7     # only Figure 6
python scripts/generate_figures.py --participants p1 p2
```

## Requirements

See [`requirements.txt`](requirements.txt) for pinned dependency versions. Core dependencies:

- Python >= 3.8
- numpy, pandas, scipy
- avro-python3 (AVRO file reading)
- geopandas, shapely, pyproj, rasterio, contextily (spatial data and Figure 6/7 plotting)
- matplotlib
- python-dotenv

## License

The **dataset** is licensed under CC BY-NC-SA 4.0 — see [Dataset Access](#dataset-access).

## Citation

If you use the GeoFatigue dataset in your research, please cite both the publication and the dataset:

```bibtex
@article{geofatigue2026,
  title={GeoFatigue: A Multimodal Dataset for Fatigue Monitoring in Construction Workers},
  author={Jahromi, M.M., Liang, S.},
  journal={Scientific Data},
  year={2026}
}

@dataset{geofatigue_dataset,
  title={GeoFatigue},
  author={Mohammadi Jahromi, Mahnoush and Liang, Steve},
  UNF = {UNF:6:WErXEXUst78l7zxArBewtA==},
  year={2026},
  version = {V2},
  publisher={Borealis},
  doi={10.5683/SP3/I0KSNR},
  url={https://doi.org/10.5683/SP3/I0KSNR}
}
```

## Acknowledgments

This research was conducted at the University of Calgary. We thank all participants who contributed to this dataset.

---

**Keywords:** Fatigue monitoring, Wearable sensors, IoT, Construction safety, Physiological sensing, Contextual awareness, Multimodal data
