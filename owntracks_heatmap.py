#!/usr/bin/env python3
"""
OwnTracks Heatmap Tool

Generate interactive heatmaps from OwnTracks .rec files using the strava-local-heatmap-tool
as a base. This tool imports and reuses functions from the original strava tool while adding
OwnTracks .rec file support.
"""

import os
import sys
import glob
import pandas as pd
from pathlib import Path

# Import the recparse module
from recparse import RecParser

# Import reusable functions from the original strava tool
sys.path.append('strava-local-heatmap-tool')
exec(open('strava-local-heatmap-tool/strava-local-heatmap-tool.py').read())


def owntracks_file_parse(*, file_path):
    """Parse OwnTracks .rec files into a DataFrame with datetime, latitude, longitude."""
    parsed_data = []
    
    if file_path.endswith('.rec'):
        parser = RecParser(file_path)
        for record in parser.parse_iter():
            if record.latitude is not None and record.longitude is not None:
                parsed_data.append({
                    'datetime': record.timestamp,
                    'latitude': record.latitude,
                    'longitude': record.longitude,
                })
    
    # Create DataFrame
    df = pd.DataFrame(data=parsed_data, index=None, dtype=None)
    
    # Remove timezone information (owntracks timestamps are timezone-aware)
    if not df.empty:
        df = df.assign(datetime=df['datetime'].dt.tz_localize(tz=None))
    
    return df


def owntracks_coordinates_import(*, activities_directory):
    """Import OwnTracks .rec files into a DataFrame."""
    # List of .rec files to be imported
    activities_files = glob.glob(pathname=os.path.join(activities_directory, '*.rec'), recursive=False)
    
    # Create empty DataFrame
    activities_coordinates_df = pd.DataFrame(data=None, index=None, dtype='str')
    
    # Import activities
    for activities_file in activities_files:
        try:
            # Parse the file
            df = owntracks_file_parse(file_path=activities_file)
            
            # Create 'filename' column
            df['filename'] = activities_file
            df['filename'] = df['filename'].replace(to_replace=r'.*activities', value=r'', regex=True)
            df['filename'] = df['filename'].replace(to_replace=r'^/[/]?|\\[\\]?', value=r'', regex=True)
            
            # Concatenate DataFrame
            activities_coordinates_df = pd.concat(objs=[activities_coordinates_df, df], axis=0, ignore_index=False, sort=False)
            
        except Exception as e:
            print(f"Error parsing {activities_file}: {e}")
            pass
    
    activities_coordinates_df = activities_coordinates_df.filter(items=['datetime', 'filename', 'latitude', 'longitude'])
    
    # Remove rows without latitude/longitude
    if not activities_coordinates_df.empty and 'latitude' in activities_coordinates_df.columns:
        activities_coordinates_df = activities_coordinates_df[activities_coordinates_df['latitude'].notna()]
    else:
        print('No activities with GPS data (latitude/longitude) found.')
    
    return activities_coordinates_df


def owntracks_import(*, activities_directory, activities_file, skip_geolocation=True):
    """
    Import OwnTracks activities and generate heatmap data.
    
    Similar to the strava activities_import function but designed for OwnTracks data.
    """
    # Import .rec activity files into a DataFrame
    activities_coordinates_df = owntracks_coordinates_import(activities_directory=activities_directory)
    
    # Get geolocation (reuse from strava tool)
    activities_geolocation_df = activities_geolocator(activities_coordinates_df=activities_coordinates_df, skip_geolocation=skip_geolocation)
    
    # Import activities CSV
    activities_df = pd.read_csv(filepath_or_buffer=activities_file, sep=',', header=0, index_col=None, skiprows=0, skipfooter=0, dtype=None, engine='python', encoding='utf-8', keep_default_na=True)
    
    # Clean column names
    activities_df = clean_names(activities_df)
    
    activities_df = (
        activities_df
        # Clean 'filename' column
        .assign(filename=lambda row: row['filename'].replace(to_replace=r'^activities/|\.gz$', value='', regex=True))
        # Left join geolocation data
        .merge(right=activities_geolocation_df, how='left', on=['filename'], indicator=False)
        # Remove problematic columns
        .drop(columns=['distance', 'commute'], axis=1, errors='ignore')
        # Select essential columns
        .filter(items=[
            'activity_date', 'activity_type', 'activity_id', 'activity_name',
            'filename', 'elapsed_time', 'moving_time', 'max_speed', 'average_speed',
            'elevation_gain', 'activity_location_country_code', 'activity_location_country',
            'activity_location_state', 'activity_location_city', 'activity_location_postal_code',
            'activity_location_latitude', 'activity_location_longitude'
        ])
        # Change dtypes
        .astype(dtype={'activity_id': 'str'})
        .assign(activity_date=lambda row: row['activity_date'].apply(parser.parse))
        # Transform columns (handle potential missing columns gracefully)
        .assign(
            elapsed_time=lambda row: row['elapsed_time'] / 60 if 'elapsed_time' in row.columns else 0,
            moving_time=lambda row: row['moving_time'] / 60 if 'moving_time' in row.columns else 0,
            max_speed=lambda row: row['max_speed'] * 3.6 if 'max_speed' in row.columns else 0,
            average_speed=lambda row: row['average_speed'] * 3.6 if 'average_speed' in row.columns else 0
        )
        # Sort by date
        .sort_values(by=['activity_date', 'activity_type'], ignore_index=True)
    )
    
    return activities_df, activities_coordinates_df


def generate_owntracks_heatmap(*, activities_directory, activities_file, output_file='owntracks_heatmap.html', activity_colors=None, map_tile='dark_all'):
    """
    Generate a heatmap from OwnTracks data.
    
    Parameters:
    - activities_directory: Directory containing .rec files
    - activities_file: CSV file with activity metadata
    - output_file: Output HTML file name
    - activity_colors: Dictionary mapping activity types to colors
    - map_tile: Map tile style
    """
    if activity_colors is None:
        activity_colors = {'OwnTracks': '#FF6600'}
    
    print("Loading OwnTracks data...")
    
    # Import activities and coordinates
    activities_df, activities_coordinates_df = owntracks_import(
        activities_directory=activities_directory,
        activities_file=activities_file,
        skip_geolocation=True
    )
    
    print(f"Loaded {len(activities_df)} activities")
    print(f"Loaded {len(activities_coordinates_df)} coordinate points")
    
    if len(activities_coordinates_df) > 0:
        print("Generating heatmap...")
        
        # Create heatmap using the original strava function
        strava_activities_heatmap(
            activities_df=activities_df,
            activities_coordinates_df=activities_coordinates_df,
            strava_activities_heatmap_output_path=output_file,
            activity_colors=activity_colors,
            map_tile=map_tile
        )
        
        print(f"Heatmap generated: {output_file}")
        return True
    else:
        print("No coordinate data found")
        return False


if __name__ == "__main__":
    # Simple command line interface
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate heatmaps from OwnTracks .rec files')
    parser.add_argument('--activities-directory', default='activities', help='Directory containing .rec files')
    parser.add_argument('--activities-file', default='activities.csv', help='CSV file with activity metadata')
    parser.add_argument('--output', default='owntracks_heatmap.html', help='Output HTML file')
    parser.add_argument('--color', default='#FF6600', help='Color for OwnTracks activities')
    parser.add_argument('--tile', default='dark_all', help='Map tile style')
    
    args = parser.parse_args()
    
    activity_colors = {'OwnTracks': args.color}
    
    success = generate_owntracks_heatmap(
        activities_directory=args.activities_directory,
        activities_file=args.activities_file,
        output_file=args.output,
        activity_colors=activity_colors,
        map_tile=args.tile
    )
    
    if success:
        print(f"\n✅ Success! Open {args.output} in your browser to view the heatmap.")
    else:
        print("\n❌ Failed to generate heatmap.")