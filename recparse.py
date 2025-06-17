"""
OwnTracks .rec file parser

Parses OwnTracks .rec files which contain location tracking data in the format:
timestamp\t*\t{json_payload}

Each line contains:
- ISO timestamp
- Tab separator
- Asterisk (*)
- Tab separator
- JSON payload with location data
"""

import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LocationRecord:
    """Represents a single location record from an OwnTracks .rec file"""
    timestamp: datetime
    data: Dict[str, Any]
    
    @property
    def latitude(self) -> Optional[float]:
        """Get latitude from the location data"""
        return self.data.get('lat')
    
    @property
    def longitude(self) -> Optional[float]:
        """Get longitude from the location data"""
        return self.data.get('lon')
    
    @property
    def accuracy(self) -> Optional[int]:
        """Get GPS accuracy in meters"""
        return self.data.get('acc')
    
    @property
    def battery(self) -> Optional[int]:
        """Get battery level percentage"""
        return self.data.get('batt')
    
    @property
    def altitude(self) -> Optional[float]:
        """Get altitude in meters"""
        return self.data.get('alt')
    
    @property
    def velocity(self) -> Optional[float]:
        """Get velocity in km/h"""
        return self.data.get('vel')
    
    @property
    def connection_type(self) -> Optional[str]:
        """Get connection type (w=wifi, m=mobile)"""
        return self.data.get('conn')
    
    @property
    def tracker_id(self) -> Optional[str]:
        """Get tracker ID"""
        return self.data.get('tid')


class RecParser:
    """Parser for OwnTracks .rec files"""
    
    def __init__(self, file_path: str):
        """Initialize parser with file path"""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

    @staticmethod
    def parse_line(line: str) -> Optional[LocationRecord]:
        """Parse a single line from the .rec file"""
        line = line.strip()
        if not line:
            return None

        try:
            # Split on tabs - expecting: timestamp\t*\tjson_data
            parts = line.split('\t', 2)
            if len(parts) != 3:
                return None

            timestamp_str, asterisk, json_str = parts

            # Validate format
            if asterisk.strip() != '*':
                return None

            # Parse timestamp
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

            # Parse JSON data
            data = json.loads(json_str)

            return LocationRecord(timestamp=timestamp, data=data)

        except (ValueError, json.JSONDecodeError) as e:
            print(f"Skipping malformed line: {e}")
            return None

    def parse_all(self) -> List[LocationRecord]:
        """Parse all records from the file"""
        records = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                record = self.parse_line(line)
                if record:
                    records.append(record)
        return records
    
    def parse_iter(self) -> Iterator[LocationRecord]:
        """Parse records as an iterator (memory efficient for large files)"""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = self.parse_line(line)
                if record:
                    yield record
    
    def get_location_points(self) -> List[Dict[str, float]]:
        """Extract just the lat/lon coordinates as a list of dictionaries"""
        points = []
        for record in self.parse_iter():
            if record.latitude is not None and record.longitude is not None:
                points.append({
                    'lat': record.latitude,
                    'lon': record.longitude,
                    'timestamp': record.timestamp.isoformat()
                })
        return points
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the file"""
        records = list(self.parse_iter())
        if not records:
            return {'total_records': 0}
        
        timestamps = [r.timestamp for r in records]
        valid_coords = [(r.latitude, r.longitude) for r in records 
                       if r.latitude is not None and r.longitude is not None]
        
        stats = {
            'total_records': len(records),
            'valid_coordinates': len(valid_coords),
            'date_range': {
                'start': min(timestamps).isoformat(),
                'end': max(timestamps).isoformat()
            }
        }
        
        if valid_coords:
            lats, lons = zip(*valid_coords)
            stats['coordinate_bounds'] = {
                'lat_min': min(lats),
                'lat_max': max(lats),
                'lon_min': min(lons),
                'lon_max': max(lons)
            }
        
        return stats


def parse_rec_file(file_path: str) -> List[LocationRecord]:
    """Convenience function to parse a .rec file"""
    parser = RecParser(file_path)
    return parser.parse_all()


def cli_main():
    # Example usage
    import sys

    if len(sys.argv) != 2:
        print("Usage: python recparse.py <path_to_rec_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    try:
        parser = RecParser(file_path)
        stats = parser.get_stats()

        print(f"File: {file_path}")
        print(f"Total records: {stats['total_records']}")
        print(f"Valid coordinates: {stats['valid_coordinates']}")

        if 'date_range' in stats:
            print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")

        if 'coordinate_bounds' in stats:
            bounds = stats['coordinate_bounds']
            print(f"Coordinate bounds:")
            print(f"  Latitude: {bounds['lat_min']:.6f} to {bounds['lat_max']:.6f}")
            print(f"  Longitude: {bounds['lon_min']:.6f} to {bounds['lon_max']:.6f}")

        # Show first few records
        print(f"\nFirst 3 records:")
        for i, record in enumerate(parser.parse_iter()):
            if i >= 3:
                break
            print(f"  {record.timestamp}: lat={record.latitude}, lon={record.longitude}, batt={record.battery}%")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    cli_main()
