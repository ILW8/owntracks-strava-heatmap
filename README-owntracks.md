# OwnTracks Heatmap Tool

Generate interactive heatmaps from OwnTracks location data using the same visualization engine as strava-local-heatmap-tool.

## Prerequisites

1. **OwnTracks .rec files** - Your location tracking data from OwnTracks
2. **Python dependencies** - Install from the project root:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Place your OwnTracks .rec files in a directory (e.g., `activities/`):
```
activities/
├── 2024-01.rec
├── 2024-02.rec
└── 2024-03.rec
```

All `.rec` files in the directory will be automatically discovered and metadata (activity date, name, etc.) will be generated from the GPS data.

## Usage

### Command Line Interface

Generate a heatmap with default settings:
```bash
python owntracks_heatmap.py --activities-directory activities
```

Customize output and styling:
```bash
python owntracks_heatmap.py \
  --activities-directory path/to/rec/files \
  --output my_heatmap.html \
  --color "#FF6600" \
  --tile dark_all
```

### CLI Options

- `--activities-directory` - Directory containing .rec files (default: `activities`)
- `--output` - Output HTML filename (default: `owntracks_heatmap.html`)
- `--color` - Color for tracks in hex format (default: `#FF6600`)
- `--tile` - Map tile style (default: `dark_all`)
- `--no-segmentation` - Disable activity segmentation based on GPS gaps
- `--max-distance` - Maximum distance between consecutive points in meters (default: `300`)
- `--max-time` - Maximum time between consecutive points in seconds (default: `60`)
- `--min-points` - Minimum number of points required for a segment to be kept (default: `5`)

### Python Library Usage

```python
from owntracks_heatmap import generate_owntracks_heatmap

success = generate_owntracks_heatmap(
    activities_directory='activities',
    output_file='my_heatmap.html',
    activity_colors={'OwnTracks': '#FF6600'},
    map_tile='dark_all'
)
```

## Output

The tool generates an interactive HTML file that you can open in any web browser. The heatmap shows your location data as colored tracks with popup information for each activity.

## Dependencies

This tool reuses functions from [strava-local-heatmap-tool](https://github.com/remisalmon/strava-local-heatmap-tool) via the local `strava_local_heatmap_tool.py` module. All required packages are listed in `requirements.txt`.