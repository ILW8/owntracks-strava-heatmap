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
import math

from line_profiler_pycharm import profile

# Import the recparse module
from recparse import RecParser

# Import reusable functions from the original strava tool
sys.path.append('strava-local-heatmap-tool')
import importlib.util
spec = importlib.util.spec_from_file_location("strava_tool", "strava-local-heatmap-tool/strava-local-heatmap-tool.py")
strava_tool = importlib.util.module_from_spec(spec)
sys.modules['strava_tool'] = strava_tool
spec.loader.exec_module(strava_tool)

# Import specific functions we need
activities_geolocator = strava_tool.activities_geolocator
strava_activities_heatmap = strava_tool.strava_activities_heatmap
clean_names = strava_tool.clean_names
from dateutil import parser as date_parser


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees).
    Returns distance in meters.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    
    return c * r


def segment_gps_points(points, max_distance_m=300, max_time_s=60, min_points=5):
    """
    Split GPS points into segments based on distance and time thresholds.
    
    Args:
        points: List of LocationRecord objects with latitude, longitude, timestamp
        max_distance_m: Maximum distance between consecutive points (meters)
        max_time_s: Maximum time between consecutive points (seconds)
        min_points: Minimum number of points required for a segment to be kept (default: 5)
    
    Returns:
        List of segments, where each segment is a list of LocationRecord objects
    """
    if not points:
        return []
    
    segments = []
    current_segment = [points[0]]
    
    for i in range(1, len(points)):
        prev_point = points[i-1]
        curr_point = points[i]
        
        # Calculate distance and time difference
        distance = haversine_distance(
            prev_point.latitude, prev_point.longitude,
            curr_point.latitude, curr_point.longitude
        )
        time_diff = (curr_point.timestamp - prev_point.timestamp).total_seconds()
        
        # Check if we should start a new segment
        if distance > max_distance_m or time_diff > max_time_s:
            # Finish current segment if it has points
            if current_segment:
                segments.append(current_segment)
            # Start new segment
            current_segment = [curr_point]
        else:
            # Add to current segment
            current_segment.append(curr_point)
    
    # Don't forget the last segment
    if current_segment:
        segments.append(current_segment)
    
    # Filter out segments with fewer than min_points
    filtered_segments = [segment for segment in segments if len(segment) >= min_points]
    
    # Report filtering if any segments were removed
    if len(segments) > len(filtered_segments):
        removed_count = len(segments) - len(filtered_segments)
        print(f"Filtered out {removed_count} segment(s) with fewer than {min_points} points")
    
    return filtered_segments


def owntracks_file_parse(*, file_path, enable_segmentation=True, max_distance_m=300, max_time_s=60, min_points=5):
    """Parse OwnTracks .rec files into a DataFrame with datetime, latitude, longitude, segment_id."""
    parsed_data = []
    
    if file_path.endswith('.rec'):
        parser = RecParser(file_path)
        
        if enable_segmentation:
            # Collect all valid GPS points first
            valid_points = []
            for record in parser.parse_iter():
                if record.latitude is not None and record.longitude is not None:
                    valid_points.append(record)
            
            # Segment the points
            segments = segment_gps_points(valid_points, max_distance_m, max_time_s, min_points)
            
            # Convert segments to data rows
            for segment_idx, segment_points in enumerate(segments):
                for record in segment_points:
                    parsed_data.append({
                        'datetime': record.timestamp,
                        'latitude': record.latitude,
                        'longitude': record.longitude,
                        'segment_id': segment_idx,
                    })
        else:
            # Original behavior - no segmentation
            for record in parser.parse_iter():
                if record.latitude is not None and record.longitude is not None:
                    parsed_data.append({
                        'datetime': record.timestamp,
                        'latitude': record.latitude,
                        'longitude': record.longitude,
                        'segment_id': 0,  # Single segment
                    })
    
    # Create DataFrame
    df = pd.DataFrame(data=parsed_data, index=None, dtype=None)
    
    # Remove timezone information (owntracks timestamps are timezone-aware)
    if not df.empty:
        df = df.assign(datetime=df['datetime'].dt.tz_localize(tz=None))
    
    return df


def owntracks_coordinates_import(*, activities_directory, activities_file_list=None, enable_segmentation=True, max_distance_m=300, max_time_s=60, min_points=5):
    """Import OwnTracks .rec files into a DataFrame, with optional activity segmentation."""
    # List of .rec files to be imported
    if activities_file_list is not None:
        # Only load files listed in activities.csv
        activities_files = [os.path.join(activities_directory, filename) for filename in activities_file_list if filename.endswith('.rec')]
    else:
        # Fallback to original behavior (load all .rec files)
        activities_files = glob.glob(pathname=os.path.join(activities_directory, '*.rec'), recursive=False)
    
    # Collect all DataFrames for efficient concatenation
    dataframes = []
    
    # Import activities
    for activities_file in activities_files:
        try:
            # Parse the file with segmentation
            df = owntracks_file_parse(
                file_path=activities_file, 
                enable_segmentation=enable_segmentation,
                max_distance_m=max_distance_m,
                max_time_s=max_time_s,
                min_points=min_points
            )
            
            if df.empty:
                continue
            
            # Create base filename
            base_filename = activities_file
            base_filename = base_filename.replace(activities_directory, '').lstrip('/\\')
            
            # Create unique filename for each segment (optimized)
            if 'segment_id' in df.columns:
                # Calculate once per file instead of once per row
                num_segments = df['segment_id'].nunique()
                if enable_segmentation and num_segments > 1:
                    # Use vectorized string operations instead of apply
                    df['filename'] = base_filename + '#' + df['segment_id'].astype(str)
                else:
                    df['filename'] = base_filename
            else:
                df['filename'] = base_filename
            
            # Add DataFrame to list for efficient concatenation
            dataframes.append(df)
            
            # Print segmentation info
            if enable_segmentation and 'segment_id' in df.columns:
                if num_segments > 1:
                    print(f"Segmented {base_filename} into {num_segments} activities")
            
        except Exception as e:
            print(f"Error parsing {activities_file}: {e}")
            pass
    
    # Efficient concatenation of all DataFrames at once
    if dataframes:
        activities_coordinates_df = pd.concat(dataframes, axis=0, ignore_index=True, sort=False)

        # Filter columns (keep segment_id for debugging if present)
        available_columns = ['datetime', 'filename', 'latitude', 'longitude']
        if 'segment_id' in activities_coordinates_df.columns:
            available_columns.append('segment_id')
        
        activities_coordinates_df = activities_coordinates_df.filter(items=available_columns)
        
        # Remove rows without latitude/longitude
        if 'latitude' in activities_coordinates_df.columns:
            activities_coordinates_df = activities_coordinates_df[activities_coordinates_df['latitude'].notna()]
        else:
            print('No activities with GPS data (latitude/longitude) found.')
    else:
        print('No activities with GPS data (latitude/longitude) found.')
        activities_coordinates_df = pd.DataFrame()
    
    return activities_coordinates_df


@profile
def create_segment_activities_metadata(activities_coordinates_df, original_activities_df):
    """
    Create activities metadata for segmented data (optimized).
    
    Args:
        activities_coordinates_df: DataFrame with coordinates and segmented filenames
        original_activities_df: DataFrame with original activities metadata
        
    Returns:
        DataFrame with activities metadata for each segment
    """
    # Get unique filenames from coordinates (these include segment info like "file.rec#0")
    unique_filenames = activities_coordinates_df['filename'].unique()
    
    # Pre-index original activities for faster lookup
    original_lookup = {row['filename']: row for _, row in original_activities_df.iterrows()}
    
    # Pre-compute datetime minimums for all filenames using groupby (MASSIVE speedup)
    datetime_mins = activities_coordinates_df.groupby('filename')['datetime'].min()
    
    segment_activities = []
    
    for segment_filename in unique_filenames:
        # Extract base filename (remove segment suffix)
        if '#' in segment_filename:
            base_filename, segment_id = segment_filename.split('#', 1)
            segment_id = int(segment_id)
        else:
            base_filename = segment_filename
            segment_id = 0
        
        # Fast lookup using pre-built index
        original_row = original_lookup.get(base_filename)
        
        if original_row is not None:
            # Use original metadata as template
            row_data = original_row.to_dict() if hasattr(original_row, 'to_dict') else dict(original_row)
            
            # Modify for segment
            row_data['filename'] = segment_filename
            row_data['activity_id'] = f"{row_data['activity_id']}_{segment_id}" if segment_id > 0 else row_data['activity_id']
            if segment_id > 0:
                row_data['activity_name'] = f"{row_data['activity_name']} (Segment {segment_id + 1})"
            
            segment_activities.append(row_data)
        else:
            # Create default metadata for segments without original data
            # Use pre-computed datetime minimum (eliminates 30ms per lookup!)
            first_timestamp = datetime_mins.get(segment_filename)
            if first_timestamp is not None:
                activity_date = first_timestamp.strftime('%Y-%m-%d') if pd.notna(first_timestamp) else '1970-01-01'
                
                segment_activities.append({
                    'activity_date': activity_date,
                    'activity_type': 'OwnTracks',
                    'activity_id': f"owntracks_{segment_filename.replace('.', '_').replace('#', '_')}",
                    'activity_name': f"OwnTracks {base_filename}" + (f" (Segment {segment_id + 1})" if segment_id > 0 else ""),
                    'filename': segment_filename,
                    'elapsed_time': 0,
                    'moving_time': 0,
                    'max_speed': 0,
                    'average_speed': 0,
                    'elevation_gain': 0,
                })
    
    return pd.DataFrame(segment_activities)


def owntracks_import(*, activities_directory, activities_file, skip_geolocation=True, 
                     enable_segmentation=True, max_distance_m=300, max_time_s=60, min_points=5):
    """
    Import OwnTracks activities and generate heatmap data.
    
    Similar to the strava activities_import function but designed for OwnTracks data.
    """
    # Import original activities CSV to get list of files to process
    original_activities_df = pd.read_csv(filepath_or_buffer=activities_file, sep=',', header=0, index_col=None, skiprows=0, skipfooter=0, dtype=None, engine='python', encoding='utf-8', keep_default_na=True)
    
    # Clean column names
    original_activities_df = clean_names(original_activities_df)
    
    # Clean filename column in original data
    original_activities_df = original_activities_df.assign(
        filename=lambda row: row['filename'].replace(to_replace=r'^activities/|\.gz$', value='', regex=True)
    )
    
    # Get list of .rec files from activities.csv
    activities_file_list = original_activities_df['filename'].tolist()
    
    # Import .rec activity files into a DataFrame with segmentation
    activities_coordinates_df = owntracks_coordinates_import(
        activities_directory=activities_directory,
        activities_file_list=activities_file_list,
        enable_segmentation=enable_segmentation,
        max_distance_m=max_distance_m,
        max_time_s=max_time_s,
        min_points=min_points
    )
    
    # Get geolocation (reuse from strava tool)
    activities_geolocation_df = activities_geolocator(activities_coordinates_df=activities_coordinates_df, skip_geolocation=skip_geolocation)
    
    # Create segment activities metadata
    activities_df = create_segment_activities_metadata(activities_coordinates_df, original_activities_df)
    
    # Merge with geolocation data
    activities_df = (
        activities_df
        .merge(right=activities_geolocation_df, how='left', on=['filename'], indicator=False)
        # Remove problematic columns
        .drop(columns=['distance', 'commute'], axis=1, errors='ignore')
        # Select essential columns (handle missing columns gracefully)
    )
    
    # Ensure required columns exist
    required_columns = [
        'activity_date', 'activity_type', 'activity_id', 'activity_name',
        'filename', 'elapsed_time', 'moving_time', 'max_speed', 'average_speed',
        'elevation_gain'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in activities_df.columns:
            activities_df[col] = 0 if col in ['elapsed_time', 'moving_time', 'max_speed', 'average_speed', 'elevation_gain'] else ''
    
    # Add geolocation columns if they exist
    geolocation_columns = [
        'activity_location_country_code', 'activity_location_country',
        'activity_location_state', 'activity_location_city', 'activity_location_postal_code',
        'activity_location_latitude', 'activity_location_longitude'
    ]
    
    available_columns = required_columns + [col for col in geolocation_columns if col in activities_df.columns]
    activities_df = activities_df.filter(items=available_columns)
    
    # Clean up data types and format
    activities_df = (
        activities_df
        .astype(dtype={'activity_id': 'str'})
        .assign(activity_date=lambda row: row['activity_date'].apply(date_parser.parse))
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


def generate_owntracks_heatmap(*, activities_directory, activities_file, output_file='owntracks_heatmap.html',
                               activity_colors=None, map_tile='dark_all', enable_segmentation=True, 
                               max_distance_m=300, max_time_s=60, min_points=5):
    """
    Generate a heatmap from OwnTracks data.
    
    Parameters:
    - activities_directory: Directory containing .rec files
    - activities_file: CSV file with activity metadata
    - output_file: Output HTML file name
    - activity_colors: Dictionary mapping activity types to colors
    - map_tile: Map tile style
    - enable_segmentation: Whether to segment activities based on gaps
    - max_distance_m: Maximum distance between consecutive points (meters)
    - max_time_s: Maximum time between consecutive points (seconds)
    - min_points: Minimum number of points required for a segment to be kept (default: 5)
    """
    if activity_colors is None:
        activity_colors = {'OwnTracks': '#FF6600'}
    
    print("Loading OwnTracks data...")
    if enable_segmentation:
        print(f"Activity segmentation enabled: max distance {max_distance_m}m, max time {max_time_s}s, min points {min_points}")
    
    # Import activities and coordinates
    activities_df, activities_coordinates_df = owntracks_import(
        activities_directory=activities_directory,
        activities_file=activities_file,
        skip_geolocation=True,
        enable_segmentation=enable_segmentation,
        max_distance_m=max_distance_m,
        max_time_s=max_time_s,
        min_points=min_points
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
    parser.add_argument('--no-segmentation', action='store_true', help='Disable activity segmentation based on gaps')
    parser.add_argument('--max-distance', type=int, default=300, help='Maximum distance between consecutive points in meters (default: 300)')
    parser.add_argument('--max-time', type=int, default=60, help='Maximum time between consecutive points in seconds (default: 60)')
    parser.add_argument('--min-points', type=int, default=5, help='Minimum number of points required for a segment to be kept (default: 5)')
    
    args = parser.parse_args()
    
    activity_colors = {'OwnTracks': args.color}
    
    success = generate_owntracks_heatmap(
        activities_directory=args.activities_directory,
        activities_file=args.activities_file,
        output_file=args.output,
        activity_colors=activity_colors,
        map_tile=args.tile,
        enable_segmentation=not args.no_segmentation,
        max_distance_m=args.max_distance,
        max_time_s=args.max_time,
        min_points=args.min_points
    )
    
    if success:
        print(f"\n✅ Success! Open {args.output} in your browser to view the heatmap.")
    else:
        print("\n❌ Failed to generate heatmap.")