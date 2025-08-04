"""
Data Loader Module for Fantasy Football Analyzer

Handles all data loading, validation, and preprocessing operations.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime

from config.constants import (
    POSITIONS, SUPPORTED_YEARS, FILE_PATTERNS,
    KEY_METRICS, DEFAULT_SCORE
)

warnings.filterwarnings('ignore')


class DataLoader:
    """
    Handles loading and validation of all fantasy football data files.
    
    Expected file naming convention:
    - draft_YYYY.csv (e.g., draft_2024.csv, draft_2023.csv)
    - adp_YYYY.csv (e.g., adp_2024.csv, adp_2023.csv)
    - results_YYYY.csv (e.g., results_2024.csv, results_2023.csv)
    - advanced_[pos]_YYYY.csv (e.g., advanced_qb_2024.csv, advanced_wr_2023.csv)
    """
    
    def __init__(self, data_directory: str):
        """
        Initialize the data loader with a directory containing CSV files.
        
        Args:
            data_directory: Path to directory containing CSV files
        """
        self.data_dir = Path(data_directory)
        self.draft_data = {}
        self.adp_data = {}
        self.results_data = {}
        self.advanced_data = {}  # Stores advanced metrics by position and year
        self.injury_data = {}    # Stores injury predictor data by position
        self.years = []
        
        # Debug logging
        self.debug_log = []
        self.missing_data_log = []
        
    def diagnose_csv_files(self) -> None:
        """Diagnose CSV files in the data directory for potential issues."""
        print("\n" + "="*60)
        print("CSV FILE DIAGNOSIS")
        print("="*60)
        
        if not self.data_dir.exists():
            print(f"ERROR: Data directory '{self.data_dir}' does not exist!")
            return
        
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            print(f"ERROR: No CSV files found in '{self.data_dir}'!")
            return
        
        print(f"Found {len(csv_files)} CSV files:")
        for file in csv_files:
            print(f"  • {file.name}")
        
        # Check for expected file patterns
        expected_patterns = {
            'draft': 'draft_YYYY.csv',
            'adp': 'adp_YYYY.csv', 
            'results': 'results_YYYY.csv',
            'advanced': 'advanced_[pos]_YYYY.csv',
            'injury': 'injury_predictor_[pos].csv'
        }
        
        print(f"\nChecking for expected file patterns:")
        for pattern_name, pattern in expected_patterns.items():
            matching_files = []
            for file in csv_files:
                if self._matches_pattern(file.name, pattern):
                    matching_files.append(file.name)
            
            if matching_files:
                print(f"  ✓ {pattern_name}: {len(matching_files)} files found")
                for file in matching_files[:3]:  # Show first 3
                    print(f"    - {file}")
                if len(matching_files) > 3:
                    print(f"    ... and {len(matching_files) - 3} more")
            else:
                print(f"  ✗ {pattern_name}: No files found")
    
    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches expected pattern."""
        if 'YYYY' in pattern:
            # Replace YYYY with year pattern
            year_pattern = pattern.replace('YYYY', r'\d{4}')
            import re
            return bool(re.match(year_pattern, filename))
        elif '[pos]' in pattern:
            # Replace [pos] with position pattern
            pos_pattern = pattern.replace('[pos]', r'(qb|rb|wr|te)')
            import re
            return bool(re.match(pos_pattern, filename, re.IGNORECASE))
        else:
            return filename == pattern
    
    def safe_read_csv(self, file_path: Path, file_type: str, year: int) -> Optional[pd.DataFrame]:
        """
        Safely read a CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            file_type: Type of file (draft, adp, results, advanced, injury)
            year: Year of the data
            
        Returns:
            DataFrame if successful, None if failed
        """
        try:
            if not file_path.exists():
                print(f"  ✗ {file_type} {year}: File not found - {file_path}")
                return None
            
            df = pd.read_csv(file_path)
            
            if df.empty:
                print(f"  ✗ {file_type} {year}: Empty file - {file_path}")
                return None
            
            print(f"  ✓ {file_type} {year}: Loaded {len(df)} rows - {file_path}")
            return df
            
        except Exception as e:
            print(f"  ✗ {file_type} {year}: Error reading file - {e}")
            return None
    
    def load_data(self) -> None:
        """Load all data files from the data directory."""
        print("\n" + "="*60)
        print("LOADING DATA FILES")
        print("="*60)
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory '{self.data_dir}' does not exist!")
        
        # Find all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in '{self.data_dir}'!")
        
        # Extract years from filenames
        years = set()
        for file in csv_files:
            if 'draft_' in file.name or 'adp_' in file.name or 'results_' in file.name:
                # Extract year from filename
                try:
                    year = int(file.name.split('_')[1].split('.')[0])
                    if 2019 <= year <= 2024:
                        years.add(year)
                except (IndexError, ValueError):
                    continue
        
        self.years = sorted(list(years))
        print(f"Detected years: {self.years}")
        
        # Load draft data
        print(f"\nLoading draft data...")
        for year in self.years:
            file_path = self.data_dir / f"draft_{year}.csv"
            df = self.safe_read_csv(file_path, "draft", year)
            if df is not None:
                self.draft_data[year] = df
        
        # Load ADP data
        print(f"\nLoading ADP data...")
        for year in self.years:
            file_path = self.data_dir / f"adp_{year}.csv"
            df = self.safe_read_csv(file_path, "adp", year)
            if df is not None:
                self.adp_data[year] = df
        
        # Load results data
        print(f"\nLoading results data...")
        for year in self.years:
            file_path = self.data_dir / f"results_{year}.csv"
            df = self.safe_read_csv(file_path, "results", year)
            if df is not None:
                self.results_data[year] = df
        
        # Load advanced metrics data
        print(f"\nLoading advanced metrics data...")
        for year in self.years:
            for position in POSITIONS:
                file_path = self.data_dir / f"advanced_{position.lower()}_{year}.csv"
                df = self.safe_read_csv(file_path, f"advanced_{position}", year)
                if df is not None:
                    if position not in self.advanced_data:
                        self.advanced_data[position] = {}
                    self.advanced_data[position][year] = df
        
        # Load injury data
        self._load_injury_data()
        
        print(f"\nData loading complete!")
        print(f"  • Draft data: {len(self.draft_data)} years")
        print(f"  • ADP data: {len(self.adp_data)} years")
        print(f"  • Results data: {len(self.results_data)} years")
        print(f"  • Advanced data: {len(self.advanced_data)} positions")
        print(f"  • Injury data: {len(self.injury_data)} positions")
    
    def _load_injury_data(self) -> None:
        """Load injury predictor data for each position."""
        print(f"\nLoading injury predictor data...")
        
        for position in POSITIONS:
            file_path = self.data_dir / f"injury_predictor_{position.lower()}.csv"
            df = self.safe_read_csv(file_path, f"injury_{position}", 0)  # No year for injury data
            if df is not None:
                self.injury_data[position] = df
    
    def clean_player_name(self, name: Any) -> str:
        """
        Clean player name for consistent matching.
        
        Args:
            name: Raw player name
            
        Returns:
            Cleaned player name
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        # Remove common suffixes and clean up
        name = str(name).strip()
        name = name.replace(" Jr.", "").replace(" Sr.", "").replace(" III", "").replace(" II", "")
        name = name.replace("'", "").replace("'", "")  # Remove smart quotes
        
        return name
    
    def clean_player_name_for_matching(self, name: Any) -> str:
        """
        Clean player name specifically for matching across datasets.
        
        Args:
            name: Raw player name
            
        Returns:
            Cleaned player name for matching
        """
        if pd.isna(name) or not isinstance(name, str):
            return ""
        
        name = str(name).strip()
        
        # Remove common suffixes
        suffixes = [" Jr.", " Sr.", " III", " II", " IV", " V"]
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        
        # Remove smart quotes and other special characters
        name = name.replace("'", "").replace("'", "").replace('"', "").replace('"', "")
        name = name.replace("-", " ").replace(".", "")
        
        # Convert to lowercase for matching
        name = name.lower()
        
        return name
    
    def safe_column_access(self, df: pd.DataFrame, possible_columns: List[str], 
                          default_value: Any = None) -> Optional[str]:
        """
        Safely find a column in a DataFrame from a list of possible names.
        
        Args:
            df: DataFrame to search
            possible_columns: List of possible column names
            default_value: Default value if no column found
            
        Returns:
            Column name if found, default_value otherwise
        """
        for col in possible_columns:
            if col in df.columns:
                return col
        return default_value
    
    def _find_column_by_patterns(self, df: pd.DataFrame, patterns: List[str], 
                                column_type: str, debug_prefix: str = "") -> Optional[str]:
        """
        Find column by matching patterns.
        
        Args:
            df: DataFrame to search
            patterns: List of patterns to match
            column_type: Type of column being searched
            debug_prefix: Debug prefix for logging
            
        Returns:
            Column name if found, None otherwise
        """
        for pattern in patterns:
            matching_cols = [col for col in df.columns if pattern.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        
        # Log missing column
        self._log_missing_data("unknown", "unknown", f"missing_{column_type}_column", 
                              patterns, f"{debug_prefix}DataFrame")
        return None
    
    def get_player_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Get the player name column from a DataFrame."""
        return self._find_column_by_patterns(df, ['player', 'name', 'playername'], 
                                           'player', debug_prefix)
    
    def get_position_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Get the position column from a DataFrame."""
        return self._find_column_by_patterns(df, ['pos', 'position'], 
                                           'position', debug_prefix)
    
    def get_points_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Get the fantasy points column from a DataFrame."""
        return self._find_column_by_patterns(df, ['pts', 'points', 'fpts', 'ttl'], 
                                           'points', debug_prefix)
    
    def get_adp_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Get the ADP column from a DataFrame."""
        return self._find_column_by_patterns(df, ['adp', 'avg', 'average'], 
                                           'adp', debug_prefix)
    
    def _add_clean_player_column(self, df: pd.DataFrame, player_column: str, 
                                for_matching: bool = False) -> pd.DataFrame:
        """
        Add a cleaned player name column to a DataFrame.
        
        Args:
            df: DataFrame to modify
            player_column: Name of the player column
            for_matching: If True, use matching-specific cleaning
            
        Returns:
            DataFrame with added clean player column
        """
        df_copy = df.copy()
        
        if for_matching:
            df_copy['Player_Clean'] = df_copy[player_column].apply(self.clean_player_name_for_matching)
        else:
            df_copy['Player_Clean'] = df_copy[player_column].apply(self.clean_player_name)
        
        return df_copy
    
    def _log_missing_data(self, player_name: str, position: str, missing_stat: str, 
                         attempted_columns: List[str], data_source: str = "unknown"):
        """Log missing data for debugging."""
        log_entry = {
            'player': player_name,
            'position': position,
            'missing_stat': missing_stat,
            'attempted_columns': attempted_columns,
            'data_source': data_source,
            'timestamp': datetime.now().isoformat()
        }
        self.missing_data_log.append(log_entry)
    
    def _log_fallback_usage(self, player_name: str, position: str, stat_name: str, 
                           fallback_value: Any, reason: str):
        """Log fallback usage for debugging."""
        log_entry = {
            'action': 'FALLBACK_USED',
            'player': player_name,
            'position': position,
            'stat': stat_name,
            'fallback_value': fallback_value,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        self.debug_log.append(log_entry)
    
    def save_debug_logs(self, filename_prefix: str = "data_debug"):
        """Save debug logs to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save missing data log
        if self.missing_data_log:
            missing_file = f"{filename_prefix}_missing_{timestamp}.txt"
            with open(missing_file, 'w') as f:
                f.write("MISSING DATA LOG\n")
                f.write("="*50 + "\n\n")
                for entry in self.missing_data_log:
                    f.write(f"Player: {entry['player']} ({entry['position']})\n")
                    f.write(f"Missing: {entry['missing_stat']}\n")
                    f.write(f"Attempted columns: {entry['attempted_columns']}\n")
                    f.write(f"Data source: {entry['data_source']}\n")
                    f.write(f"Time: {entry['timestamp']}\n")
                    f.write("-"*30 + "\n")
            print(f"Missing data log saved to: {missing_file}")
        
        # Save debug log
        if self.debug_log:
            debug_file = f"{filename_prefix}_debug_{timestamp}.txt"
            with open(debug_file, 'w') as f:
                f.write("DEBUG LOG\n")
                f.write("="*50 + "\n\n")
                for entry in self.debug_log:
                    f.write(f"Action: {entry.get('action', 'unknown')}\n")
                    f.write(f"Player: {entry.get('player', 'unknown')}\n")
                    f.write(f"Position: {entry.get('position', 'unknown')}\n")
                    f.write(f"Details: {entry}\n")
                    f.write("-"*30 + "\n")
            print(f"Debug log saved to: {debug_file}")
    
    def get_real_stat_value(self, player_stats: Dict[str, Any], stat_columns: List[str], 
                           player_name: str = "unknown", position: str = "unknown", 
                           stat_description: str = "unknown") -> Optional[float]:
        """
        Get a real stat value from player stats, trying multiple column names.
        
        Args:
            player_stats: Dictionary of player statistics
            stat_columns: List of possible column names for the stat
            player_name: Player name for logging
            position: Player position for logging
            stat_description: Description of the stat for logging
            
        Returns:
            Stat value if found, None otherwise
        """
        for col in stat_columns:
            if col in player_stats and not pd.isna(player_stats[col]):
                try:
                    value = float(player_stats[col])
                    if not np.isnan(value):
                        return value
                except (ValueError, TypeError):
                    continue
        
        # Log missing data
        self._log_missing_data(player_name, position, stat_description, stat_columns)
        return None
    
    def get_real_dict_value(self, data_dict: Dict[str, Any], keys: List[str], 
                           player_name: str = "unknown", position: str = "unknown", 
                           stat_description: str = "unknown") -> Optional[Any]:
        """
        Get a real value from a dictionary, trying multiple keys.
        
        Args:
            data_dict: Dictionary to search
            keys: List of possible keys
            player_name: Player name for logging
            position: Player position for logging
            stat_description: Description of the stat for logging
            
        Returns:
            Value if found, None otherwise
        """
        for key in keys:
            if key in data_dict and data_dict[key] is not None:
                return data_dict[key]
        
        # Log missing data
        self._log_missing_data(player_name, position, stat_description, keys)
        return None
    
    @staticmethod
    def safe_float_conversion(value: Any) -> float:
        """
        Safely convert a value to float.
        
        Args:
            value: Value to convert
            
        Returns:
            Float value, 0.0 if conversion fails
        """
        if pd.isna(value):
            return 0.0
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0 