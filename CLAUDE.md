# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an OwnTracks Heatmap Tool that generates interactive heatmaps from OwnTracks location data using the visualization engine from strava-local-heatmap-tool. The project combines location tracking data from OwnTracks (.rec files) with the existing Strava visualization library to create web-based heatmaps.

**Key Feature**: Activity segmentation automatically splits GPS tracks at gaps (distance > 300m or time > 60s) to avoid unrealistic lines across long distances due to GPS signal loss or tracking interruptions.

## Architecture

The project consists of two main Python modules:

- `owntracks_heatmap.py` - Main tool that imports and reuses functions from strava-local-heatmap-tool to generate heatmaps from OwnTracks data
- `recparse.py` - OwnTracks .rec file parser that handles the tab-separated format: `timestamp\t*\t{json_payload}`

The tool depends on the `strava-local-heatmap-tool/` subdirectory which contains the original Strava visualization engine with its own requirements and functions.

## Dependencies

The project uses the strava-local-heatmap-tool dependencies:
```bash
cd strava-local-heatmap-tool
pip install -r requirements.txt
```

Key dependencies include: fitparse, folium, geopy, gpxpy, pandas, pyjanitor, python-dateutil, tcxreader

## Common Commands

### Run the main tool:
```bash
python owntracks_heatmap.py --activities-directory activities --activities-file activities.csv
```

### Custom output and styling:
```bash
python owntracks_heatmap.py \
  --activities-directory path/to/rec/files \
  --activities-file path/to/activities.csv \
  --output my_heatmap.html \
  --color "#FF6600" \
  --tile dark_all
```

### Activity segmentation (improved GPS visualization):
```bash
# Default segmentation (300m distance, 60s time gaps)
python owntracks_heatmap.py --activities-directory activities --activities-file activities.csv

# Custom thresholds for segmentation
python owntracks_heatmap.py \
  --activities-directory activities \
  --activities-file activities.csv \
  --max-distance 500 \
  --max-time 120

# Disable segmentation (original behavior)
python owntracks_heatmap.py --activities-directory activities --activities-file activities.csv --no-segmentation
```

### Parse individual .rec files:
```bash
python recparse.py path/to/file.rec
```

### Test the recparse module:
```bash
python -c "from recparse import RecParser; parser = RecParser('data/sample.rec'); print(parser.get_stats())"
```

## Data Format

- **Input**: OwnTracks .rec files in `data/` directory with format: `timestamp\t*\t{json_payload}`
- **Metadata**: CSV file with columns: Activity Date, Activity Type, Activity ID, Activity Name, Filename, plus optional metric columns
- **Output**: Interactive HTML files with folium-based heatmaps

## Key Functions

- `generate_owntracks_heatmap()` - Main function to create heatmaps from OwnTracks data with segmentation options
- `owntracks_file_parse()` - Parse individual .rec files into DataFrames with optional segmentation
- `owntracks_coordinates_import()` - Import multiple .rec files with segmentation support
- `segment_gps_points()` - Split GPS points into segments based on distance/time thresholds
- `haversine_distance()` - Calculate distance between GPS coordinates
- `create_segment_activities_metadata()` - Generate metadata for segmented activities
- `RecParser.parse_iter()` - Memory-efficient parsing of large .rec files