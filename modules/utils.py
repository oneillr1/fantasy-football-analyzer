"""
Utilities Module for Fantasy Football Analyzer

Handles utilities, logging, and helper functions.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json


class FantasyUtils:
    """
    Utility functions for fantasy football analysis.
    """
    
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
    
    @staticmethod
    def clean_player_name(name: Any) -> str:
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
    
    @staticmethod
    def clean_player_name_for_matching(name: Any) -> str:
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
    
    @staticmethod
    def get_real_stat_value(player_stats: Dict[str, Any], stat_columns: List[str], 
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
        
        return None
    
    @staticmethod
    def get_real_dict_value(data_dict: Dict[str, Any], keys: List[str], 
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
        
        return None
    
    @staticmethod
    def safe_column_access(df: pd.DataFrame, possible_columns: List[str], 
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
    
    @staticmethod
    def find_column_by_patterns(df: pd.DataFrame, patterns: List[str], 
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
        
        return None
    
    @staticmethod
    def add_clean_player_column(df: pd.DataFrame, player_column: str, 
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
            df_copy['Player_Clean'] = df_copy[player_column].apply(FantasyUtils.clean_player_name_for_matching)
        else:
            df_copy['Player_Clean'] = df_copy[player_column].apply(FantasyUtils.clean_player_name)
        
        return df_copy
    
    @staticmethod
    def log_missing_data(player_name: str, position: str, missing_stat: str, 
                        attempted_columns: List[str], data_source: str = "unknown") -> Dict[str, Any]:
        """
        Create a missing data log entry.
        
        Args:
            player_name: Player name
            position: Player position
            missing_stat: Description of missing stat
            attempted_columns: List of columns that were attempted
            data_source: Source of the data
            
        Returns:
            Log entry dictionary
        """
        return {
            'player': player_name,
            'position': position,
            'missing_stat': missing_stat,
            'attempted_columns': attempted_columns,
            'data_source': data_source,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def log_fallback_usage(player_name: str, position: str, stat_name: str, 
                          fallback_value: Any, reason: str) -> Dict[str, Any]:
        """
        Create a fallback usage log entry.
        
        Args:
            player_name: Player name
            position: Player position
            stat_name: Name of the stat
            fallback_value: Fallback value used
            reason: Reason for using fallback
            
        Returns:
            Log entry dictionary
        """
        return {
            'action': 'FALLBACK_USED',
            'player': player_name,
            'position': position,
            'stat': stat_name,
            'fallback_value': fallback_value,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def save_debug_logs(missing_data_log: List[Dict], debug_log: List[Dict], 
                        filename_prefix: str = "data_debug") -> None:
        """
        Save debug logs to files.
        
        Args:
            missing_data_log: List of missing data log entries
            debug_log: List of debug log entries
            filename_prefix: Prefix for output files
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save missing data log
        if missing_data_log:
            missing_file = f"{filename_prefix}_missing_{timestamp}.txt"
            with open(missing_file, 'w') as f:
                f.write("MISSING DATA LOG\n")
                f.write("="*50 + "\n\n")
                for entry in missing_data_log:
                    f.write(f"Player: {entry['player']} ({entry['position']})\n")
                    f.write(f"Missing: {entry['missing_stat']}\n")
                    f.write(f"Attempted columns: {entry['attempted_columns']}\n")
                    f.write(f"Data source: {entry['data_source']}\n")
                    f.write(f"Time: {entry['timestamp']}\n")
                    f.write("-"*30 + "\n")
            print(f"Missing data log saved to: {missing_file}")
        
        # Save debug log
        if debug_log:
            debug_file = f"{filename_prefix}_debug_{timestamp}.txt"
            with open(debug_file, 'w') as f:
                f.write("DEBUG LOG\n")
                f.write("="*50 + "\n\n")
                for entry in debug_log:
                    f.write(f"Action: {entry.get('action', 'unknown')}\n")
                    f.write(f"Player: {entry.get('player', 'unknown')}\n")
                    f.write(f"Position: {entry.get('position', 'unknown')}\n")
                    f.write(f"Details: {entry}\n")
                    f.write("-"*30 + "\n")
            print(f"Debug log saved to: {debug_file}")
    
    @staticmethod
    def calculate_percentage(value: float, total: float) -> float:
        """
        Calculate percentage safely.
        
        Args:
            value: Value to calculate percentage for
            total: Total value
            
        Returns:
            Percentage (0-100)
        """
        if total == 0:
            return 0.0
        return (value / total) * 100
    
    @staticmethod
    def normalize_score(score: float, min_val: float, max_val: float) -> float:
        """
        Normalize a score to 0-10 range.
        
        Args:
            score: Raw score
            min_val: Minimum value in range
            max_val: Maximum value in range
            
        Returns:
            Normalized score (0-10)
        """
        if max_val == min_val:
            return 5.0  # Neutral score if no range
        
        normalized = ((score - min_val) / (max_val - min_val)) * 10.0
        return max(0.0, min(10.0, normalized))
    
    @staticmethod
    def format_player_name(name: str) -> str:
        """
        Format player name for display.
        
        Args:
            name: Raw player name
            
        Returns:
            Formatted player name
        """
        if not name:
            return "Unknown Player"
        
        # Title case and clean up
        name = name.strip()
        name = name.title()
        
        return name
    
    @staticmethod
    def format_position(position: str) -> str:
        """
        Format position for display.
        
        Args:
            position: Raw position string
            
        Returns:
            Formatted position
        """
        if not position:
            return "Unknown"
        
        return position.upper()
    
    @staticmethod
    def format_score(score: float, decimals: int = 2) -> str:
        """
        Format score for display.
        
        Args:
            score: Raw score
            decimals: Number of decimal places
            
        Returns:
            Formatted score string
        """
        return f"{score:.{decimals}f}"
    
    @staticmethod
    def create_summary_stats(values: List[float]) -> Dict[str, float]:
        """
        Create summary statistics for a list of values.
        
        Args:
            values: List of numeric values
            
        Returns:
            Dictionary with summary statistics
        """
        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'median': np.median(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    @staticmethod
    def validate_data_quality(data_dict: Dict[str, Any], required_keys: List[str]) -> Dict[str, Any]:
        """
        Validate data quality for a dictionary.
        
        Args:
            data_dict: Dictionary to validate
            required_keys: List of required keys
            
        Returns:
            Validation results dictionary
        """
        validation_results = {
            'is_valid': True,
            'missing_keys': [],
            'null_values': [],
            'total_checks': len(required_keys),
            'passed_checks': 0
        }
        
        for key in required_keys:
            if key not in data_dict:
                validation_results['missing_keys'].append(key)
                validation_results['is_valid'] = False
            elif data_dict[key] is None or pd.isna(data_dict[key]):
                validation_results['null_values'].append(key)
                validation_results['is_valid'] = False
            else:
                validation_results['passed_checks'] += 1
        
        return validation_results 