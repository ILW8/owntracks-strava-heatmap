# OwnTracks Heatmap Tool

Generate interactive heatmaps from OwnTracks location data using the same visualization engine as strava-local-heatmap-tool.

## Prerequisites

1. **OwnTracks .rec files** - Your location tracking data from OwnTracks
2. **Python dependencies** - Install from strava-local-heatmap-tool:
   ```bash
   cd strava-local-heatmap-tool
   pip install -r requirements.txt
   ```

## Data Preparation

### 1. Organize .rec files
Place your OwnTracks .rec files in a directory (e.g., `activities/`):
```
activities/
├── 2024-01.rec
├── 2024-02.rec
└── 2024-03.rec
```

### 2. Create activities.csv
Create a CSV file with metadata for your .rec files:

```csv
Activity Date,Activity Type,Activity ID,Activity Name,Filename,Elapsed Time,Moving Time,Distance,Max Speed,Average Speed,Elevation Gain
2024-01-01,OwnTracks,1,January 2024,2024-01.rec,0,0,0,0,0,0
2024-02-01,OwnTracks,2,February 2024,2024-02.rec,0,0,0,0,0,0
2024-03-01,OwnTracks,3,March 2024,2024-03.rec,0,0,0,0,0,0
```

**Required columns:**
- `Activity Date` - Date in YYYY-MM-DD format
- `Activity Type` - Activity type (e.g., "OwnTracks")
- `Activity ID` - Unique identifier
- `Activity Name` - Descriptive name
- `Filename` - Must match your .rec filenames
- Other columns can be set to 0 for OwnTracks data

## Usage

### Command Line Interface

Generate a heatmap with default settings:
```bash
python owntracks_heatmap.py --activities-directory activities --activities-file activities.csv
```

Customize output and styling:
```bash
python owntracks_heatmap.py \
  --activities-directory path/to/rec/files \
  --activities-file path/to/activities.csv \
  --output my_heatmap.html \
  --color "#FF6600" \
  --tile dark_all
```

### CLI Options

- `--activities-directory` - Directory containing .rec files (default: `activities`)
- `--activities-file` - CSV file with activity metadata (default: `activities.csv`)
- `--output` - Output HTML filename (default: `owntracks_heatmap.html`)
- `--color` - Color for tracks in hex format (default: `#FF6600`)
- `--tile` - Map tile style (default: `dark_all`)

### Python Library Usage

```python
from owntracks_heatmap import generate_owntracks_heatmap

success = generate_owntracks_heatmap(
    activities_directory='activities',
    activities_file='activities.csv',
    output_file='my_heatmap.html',
    activity_colors={'OwnTracks': '#FF6600'},
    map_tile='dark_all'
)
```

## Output

The tool generates an interactive HTML file that you can open in any web browser. The heatmap shows your location data as colored tracks with popup information for each activity.

## Dependencies

This tool reuses functions from [strava-local-heatmap-tool](https://github.com/remisalmon/strava-local-heatmap-tool) and requires:
- The `strava-local-heatmap-tool/` directory in the same location
- The `recparse.py` OwnTracks parser module