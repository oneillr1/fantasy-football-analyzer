"""
Simple ML System for Fantasy Football Predictions
Trains autoregressive models to predict 2025 fantasy performance based on 2024 data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr
import pickle
import warnings
from typing import Dict, List, Tuple
import os
import random
from datetime import datetime

# Import from existing modular system
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.data_loader import DataLoader
from config.constants import POSITIONS

warnings.filterwarnings('ignore')

# IMPROVEMENT #5: Add deterministic seeds for reproducibility
random.seed(42)
np.random.seed(42)
os.environ["PYTHONHASHSEED"] = "42"

# COMPREHENSIVE FEATURE SCHEMAS - Using ALL Available Metrics
FEATURE_SCHEMAS = {
    'qb': [
        # Base features (2)
        'fantasy_ppg', 'games_played',
        
        # QB-specific basic stats (5)
        'completion_pct', 'yards_per_attempt', 'td_rate', 'int_rate', 'yards_per_game',
        
        # Advanced efficiency metrics (8)
        'y_per_a', 'rtg', 'completion_pct_advanced', 'air_yards', 'air_per_a',
        'pocket_time', 'sack_rate', 'pressure_rate',
        
        # Advanced accuracy metrics (3)
        'poor_throw_rate', 'drop_rate', 'knockdown_rate',
        
        # Explosive play metrics (5)
        'passes_10_plus', 'passes_20_plus', 'passes_30_plus', 'passes_40_plus', 'passes_50_plus',
        
        # Redzone passing metrics (6)
        'rz_pass_att_per_game', 'rz_completion_pct', 'rz_td_rate', 'rz_yards_per_attempt',
        'rz_int_rate', 'rz_sack_rate',
        
        # Redzone rushing metrics (2)
        'rz_rush_att_per_game', 'rz_rush_td_rate',
        
        # Volume metrics (3)
        'completions', 'attempts', 'total_yards',
        
        # Injury and age features (5)
        'age', 'durability', 'injury_risk_score', 'career_injuries', 'projected_games_missed'
    ],
    
    'rb': [
        # Base features (2)
        'fantasy_ppg', 'games_played',
        
        # RB-specific basic stats (4)
        'yards_per_carry', 'catch_rate', 'carries_per_game', 'receptions_per_game',
        
        # Advanced efficiency metrics (6)
        'y_per_a', 'rtg', 'yards_per_attempt_rb', 'yards_before_contact_per_attempt',
        'yards_after_contact_per_attempt', 'brktkl',
        
        # Explosive play metrics (7)
        'passes_10_plus', 'passes_20_plus', 'passes_30_plus', 'passes_40_plus', 'passes_50_plus',
        'longest_touchdown', 'longest_run',
        
        # Power/elusiveness metrics (3)
        'yac_per_r', 'air_per_a', 'tackled_for_loss',
        
        # Redzone rushing metrics (4)
        'rz_carries_per_game', 'rz_rush_td_rate', 'rz_rush_yards_per_attempt', 'rz_rush_att_per_game',
        
        # Redzone receiving metrics (3)
        'rz_receptions_per_game', 'rz_receiving_td_rate', 'rz_target_share',
        
        # Volume metrics (3)
        'receptions_advanced', 'targets_advanced', 'total_yards',
        
        # Injury and age features (5)
        'age', 'durability', 'injury_risk_score', 'career_injuries', 'projected_games_missed'
    ],
    
    'wr': [
        # Base features (2)
        'fantasy_ppg', 'games_played',
        
        # WR-specific basic stats (4)
        'yards_per_reception', 'catch_rate', 'targets_per_game', 'receptions_per_game',
        
        # Advanced efficiency metrics (8)
        'y_per_a', 'rtg', 'yards_per_reception_advanced', 'yards_before_catch_per_reception',
        'yards_after_catch_per_reception', 'yards_after_contact_per_reception',
        'air_yards_wr', 'air_yards_per_reception',
        
        # Explosive play metrics (6)
        'passes_10_plus', 'passes_20_plus', 'passes_30_plus', 'passes_40_plus', 'passes_50_plus',
        'longest_run',
        
        # Playmaking metrics (4)
        'yac_per_r', 'brktkl', 'yards_before_catch', 'yards_after_catch',
        
        # Opportunity metrics (3)
        'team_target_share', 'catchable_targets', 'total_yards',
        
        # Redzone receiving metrics (4)
        'rz_targets_per_game', 'rz_receptions_per_game', 'rz_receiving_td_rate',
        'rz_yards_per_reception', 'rz_target_share',
        
        # Redzone rushing metrics (2)
        'rz_rush_att_per_game', 'rz_rush_td_rate',
        
        # Injury and age features (5)
        'age', 'durability', 'injury_risk_score', 'career_injuries', 'projected_games_missed'
    ],
    
    'te': [
        # Base features (2)
        'fantasy_ppg', 'games_played',
        
        # TE-specific basic stats (4)
        'yards_per_reception', 'catch_rate', 'targets_per_game', 'receptions_per_game',
        
        # Advanced efficiency metrics (8)
        'y_per_a', 'rtg', 'yards_per_reception_advanced', 'yards_before_catch_per_reception',
        'yards_after_catch_per_reception', 'yards_after_contact_per_reception',
        'air_yards_wr', 'air_yards_per_reception',
        
        # Explosive play metrics (6)
        'passes_10_plus', 'passes_20_plus', 'passes_30_plus', 'passes_40_plus', 'passes_50_plus',
        'longest_run',
        
        # Playmaking metrics (4)
        'yac_per_r', 'brktkl', 'yards_before_catch', 'yards_after_catch',
        
        # Opportunity metrics (3)
        'team_target_share', 'catchable_targets', 'total_yards',
        
        # Redzone receiving metrics (4)
        'rz_targets_per_game', 'rz_receptions_per_game', 'rz_receiving_td_rate',
        'rz_yards_per_reception', 'rz_target_share',
        
        # Redzone rushing metrics (2)
        'rz_rush_att_per_game', 'rz_rush_td_rate',
        
        # Injury and age features (5)
        'age', 'durability', 'injury_risk_score', 'career_injuries', 'projected_games_missed'
    ]
}

# Feature dimension validation
FEATURE_DIMENSIONS = {pos: len(features) for pos, features in FEATURE_SCHEMAS.items()}
print(f"Comprehensive feature dimensions: {FEATURE_DIMENSIONS}")
# QB: 47 features, RB: 47 features, WR: 48 features, TE: 48 features

class SimpleMLTrainer:
    def __init__(self, data_directory: str = "fantasy_data"):
        self.data_dir = data_directory
        self.data_loader = DataLoader(data_directory)  # Use existing data loader
        self.models = {}
        self.scalers = {}
        self.feature_names = {}
        self.performance_metrics = {}
        self.validation_results = {}  # Store validation results
        
    def load_data(self) -> Dict:
        """Load all required data files using existing DataLoader."""
        print("Loading data files using modular DataLoader...")
        
        # Use existing data loader
        self.data_loader.load_data()
        
        # Convert to expected format for backward compatibility
        data = {
            'stats': {},
            'results': {},
            'advanced': {},
            'redzone': {},
            'age': None,
            'adp': {}
        }
        
        # Convert results data (fantasy points)
        for year, df in self.data_loader.results_data.items():
            data['results'][year] = df
            print(f"  ✓ Converted {len(df)} results for {year}")
        
        # Convert ADP data
        for year, df in self.data_loader.adp_data.items():
            data['adp'][year] = df
            print(f"  ✓ Converted {len(df)} ADP records for {year}")
        
        # Convert advanced data
        for position, year_data in self.data_loader.advanced_data.items():
            data['advanced'][position.lower()] = year_data
            print(f"  ✓ Converted {len(year_data)} years of advanced data for {position}")
        
        # Load stats data (half PPR format)
        print("\nLoading stats data (half PPR)...")
        for position in ['qb', 'rb', 'wr', 'te']:
            data['stats'][position] = {}
            for year in range(2019, 2025):
                file_path = f"{self.data_dir}/stats_{position}_{year}_half.csv"
                if os.path.exists(file_path):
                    df = self.load_stats_file(file_path, position, year)
                    if not df.empty:
                        data['stats'][position][year] = df
        
        # Load redzone data
        print("\nLoading redzone data...")
        for position in ['qb', 'rb', 'wr', 'te']:
            data['redzone'][position] = {}
            for year in range(2019, 2025):
                file_path = f"{self.data_dir}/redzone_{position}_{year}.csv"
                if os.path.exists(file_path):
                    # Use DataLoader's safe reading method for redzone data
                    from pathlib import Path
                    file_path_obj = Path(file_path)
                    df = self.data_loader.safe_read_csv(file_path_obj, f"redzone_{position}", year)
                    if df is not None and not df.empty:
                        data['redzone'][position][year] = df
                        print(f"  ✓ Loaded {len(df)} {position.upper()} redzone data for {year}")
        
        # Load age data
        age_file = f"{self.data_dir}/player_age.csv"
        if os.path.exists(age_file):
            # Use DataLoader's safe reading method for age data
            from pathlib import Path
            age_file_obj = Path(age_file)
            df = self.data_loader.safe_read_csv(age_file_obj, "age", 0)  # No year for age data
            if df is not None and not df.empty:
                data['age'] = df
                print(f"  ✓ Loaded {len(data['age'])} age records")
        
        # FIXED: Don't calculate position averages here - will be done during training phases
        # to prevent data leakage
        
        return data
    
    def calculate_position_averages(self, data: Dict, max_year: int = 2023) -> Dict:
        """Calculate position-specific averages for missing data imputation."""
        averages = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            averages[position] = {}
            
            # Collect all available data for this position
            all_stats = []
            all_advanced = []
            all_redzone = []
            
            # FIXED: Only use data up to max_year to prevent data leakage
            for year in range(2019, max_year + 1):
                # Stats data
                if position in data['stats'] and year in data['stats'][position]:
                    all_stats.extend(data['stats'][position][year].to_dict('records'))
                
                # Advanced data
                if position in data['advanced'] and year in data['advanced'][position]:
                    all_advanced.extend(data['advanced'][position][year].to_dict('records'))
                
                # Redzone data
                if position in data['redzone'] and year in data['redzone'][position]:
                    all_redzone.extend(data['redzone'][position][year].to_dict('records'))
            
            # Calculate averages for key metrics
            if all_stats:
                stats_df = pd.DataFrame(all_stats)
                averages[position]['stats'] = {
                    'FPTS': stats_df['FPTS'].mean() if 'FPTS' in stats_df.columns else 0,
                    'G': stats_df['G'].mean() if 'G' in stats_df.columns else 16,
                    'ATT': stats_df['ATT'].mean() if 'ATT' in stats_df.columns else 0,
                    'YDS': stats_df['YDS'].mean() if 'YDS' in stats_df.columns else 0,
                    'TD': stats_df['TD'].mean() if 'TD' in stats_df.columns else 0,
                    'REC': stats_df['REC'].mean() if 'REC' in stats_df.columns else 0,
                    'TGT': stats_df['TGT'].mean() if 'TGT' in stats_df.columns else 0,
                }
            
            if all_advanced:
                advanced_df = pd.DataFrame(all_advanced)
                averages[position]['advanced'] = {
                    'Y/A': advanced_df['Y/A'].mean() if 'Y/A' in advanced_df.columns else 0,
                    'RTG': advanced_df['RTG'].mean() if 'RTG' in advanced_df.columns else 0,
                    'BRKTKL': advanced_df['BRKTKL'].mean() if 'BRKTKL' in advanced_df.columns else 0,
                    'YAC/R': advanced_df['YAC/R'].mean() if 'YAC/R' in advanced_df.columns else 0,
                    'AIR/A': advanced_df['AIR/A'].mean() if 'AIR/A' in advanced_df.columns else 0,
                }
            
            if all_redzone:
                redzone_df = pd.DataFrame(all_redzone)
                averages[position]['redzone'] = {
                    'RZ ATT': redzone_df['RZ ATT'].mean() if 'RZ ATT' in redzone_df.columns else 0,
                    'RZ TGT': redzone_df['RZ TGT'].mean() if 'RZ TGT' in redzone_df.columns else 0,
                }
        
        return averages
    
    def clean_player_name(self, name: str) -> str:
        """Clean player name for matching using existing DataLoader method."""
        return self.data_loader.clean_player_name_for_matching(name)
    
    def safe_float_conversion(self, value, default=0.0) -> float:
        """Safely convert value to float with default using existing DataLoader method."""
        return self.data_loader.safe_float_conversion(value)
    
    def validate_feature_consistency(self, X_array, position, stage_name):
        """Validate that feature dimensions are consistent."""
        expected_dim = FEATURE_DIMENSIONS.get(position, 47)  # Updated to comprehensive dimensions
        if X_array.shape[1] != expected_dim:
            print(f"  ⚠️  Feature dimension mismatch in {stage_name}: expected {expected_dim}, got {X_array.shape[1]}")
            # Pad or truncate to expected dimension
            if X_array.shape[1] < expected_dim:
                # Pad with zeros
                padding = np.zeros((X_array.shape[0], expected_dim - X_array.shape[1]))
                X_array = np.hstack([X_array, padding])
                print(f"  ✓ Padded features to {expected_dim} dimensions")
            else:
                # Truncate
                X_array = X_array[:, :expected_dim]
                print(f"  ✓ Truncated features to {expected_dim} dimensions")
        return X_array
    
    def load_stats_file(self, file_path: str, position: str, year: int) -> pd.DataFrame:
        """
        Load and process a stats file with complex multi-level column headers.
        
        Args:
            file_path: Path to the stats CSV file
            position: Position (qb, rb, wr, te)
            year: Year of the data
            
        Returns:
            Processed DataFrame with clean column names
        """
        try:
            # Read the file manually to handle the complex header structure
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                print(f"  ❌ File too short: {file_path}")
                return pd.DataFrame()
            
            # The first line contains multi-level headers, second line has actual column names
            # Skip the first line and use the second line as headers
            header_line = lines[1].strip()
            data_lines = lines[2:]  # Start from third line (index 2)
            
            # Parse the header
            headers = [col.strip() for col in header_line.split(',')]
            print(f"    DEBUG: Headers from file: {headers}")
            
            # Parse data lines
            data_rows = []
            for line in data_lines:
                if line.strip():
                    # Use a more robust CSV parsing approach
                    import csv
                    from io import StringIO
                    
                    try:
                        # Parse the line as CSV to handle quoted values properly
                        csv_reader = csv.reader(StringIO(line))
                        values = next(csv_reader)
                        values = [val.strip() for val in values]
                        
                        if len(values) >= len(headers):
                            data_rows.append(values[:len(headers)])
                    except Exception as e:
                        # Fallback to simple split if CSV parsing fails
                        values = [val.strip() for val in line.split(',')]
                        if len(values) >= len(headers):
                            data_rows.append(values[:len(headers)])
            
            # Create DataFrame
            df = pd.DataFrame(data_rows, columns=headers)
            print(f"    DEBUG: DataFrame shape: {df.shape}")
            print(f"    DEBUG: DataFrame columns: {list(df.columns)}")
            
            # Debug the FPTS column
            if 'FPTS' in df.columns:
                print(f"    DEBUG: FPTS column sample: {df['FPTS'].head(3).tolist()}")
                print(f"    DEBUG: FPTS column type: {df['FPTS'].dtype}")
            
            # Ensure we have the expected columns
            expected_columns = ['Player', 'FPTS', 'G']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            
            if missing_columns:
                print(f"  ⚠️  Missing expected columns in {file_path}: {missing_columns}")
                print(f"  Available columns: {list(df.columns)}")
                return pd.DataFrame()
            
            # Convert numeric columns to proper types
            numeric_columns = ['FPTS', 'G', 'ATT', 'YDS', 'TD', 'INT', 'SACKS', 'FL', 'REC', 'TGT', 'Y/A', 'PCT', 'CMP']
            for col in numeric_columns:
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(float)
                    except (ValueError, TypeError):
                        # If conversion fails, try to clean the data first
                        df[col] = df[col].replace('', 0).replace('nan', 0)
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            # Add cleaned player name column for matching
            if 'Player' in df.columns:
                df['Player_Clean'] = df['Player'].apply(self.clean_player_name)
                print(f"    DEBUG: Sample player names: {df['Player'].head(3).tolist()}")
                print(f"    DEBUG: Sample cleaned names: {df['Player_Clean'].head(3).tolist()}")
            
            print(f"  ✓ Loaded {len(df)} {position.upper()} stats for {year}")
            return df
            
        except Exception as e:
            print(f"  ❌ Error loading stats file {file_path}: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def extract_features_safe(self, player_data: Dict, position: str, year: int, player_name: str = "Unknown") -> np.ndarray:
        """
        Extract features using comprehensive schema-based approach to utilize ALL available metrics.
        
        Args:
            player_data: Dictionary containing player statistics
            position: Player position ('qb', 'rb', 'wr', 'te')
            year: Year of data
            player_name: Player name for logging
            
        Returns:
            numpy array with comprehensive feature dimensions for the position
        """
        if position not in FEATURE_SCHEMAS:
            raise ValueError(f"Unknown position: {position}")
        
        # Initialize feature dict with zeros for all expected features
        feature_dict = {feature: 0.0 for feature in FEATURE_SCHEMAS[position]}
        pos_avg = self.position_averages.get(position, {})
        
        # Basic stats features
        stats = player_data.get('stats', {})
        if stats:
            # Fantasy PPG (most important feature)
            fpts = self.safe_float_conversion(stats.get('FPTS', stats.get('TTL', 0)))
            games = self.safe_float_conversion(stats.get('G', 1))
            
            if games <= 0:
                games = pos_avg.get('stats', {}).get('G', 16)
            
            ppg = fpts / games if games > 0 else 0
            feature_dict['fantasy_ppg'] = ppg
            feature_dict['games_played'] = games
            
            # Position-specific features
            if position == 'qb':
                att = self.safe_float_conversion(stats.get('ATT', 1))
                comp = self.safe_float_conversion(stats.get('CMP', 0))
                yds = self.safe_float_conversion(stats.get('YDS', 0))
                td = self.safe_float_conversion(stats.get('TD', 0))
                ints = self.safe_float_conversion(stats.get('INT', 0))
                
                feature_dict['completion_pct'] = comp / att if att > 0 else 0
                feature_dict['yards_per_attempt'] = yds / att if att > 0 else 0
                feature_dict['td_rate'] = td / att if att > 0 else 0
                feature_dict['int_rate'] = ints / att if att > 0 else 0
                feature_dict['yards_per_game'] = yds / games if games > 0 else 0
                
            elif position == 'rb':
                att = self.safe_float_conversion(stats.get('ATT', 0))
                yds = self.safe_float_conversion(stats.get('YDS', 0))
                rec = self.safe_float_conversion(stats.get('REC', 0))
                tgt = self.safe_float_conversion(stats.get('TGT', 1))
                
                feature_dict['yards_per_carry'] = yds / att if att > 0 else 0
                feature_dict['catch_rate'] = rec / tgt if tgt > 0 else 0
                feature_dict['carries_per_game'] = att / games if games > 0 else 0
                feature_dict['receptions_per_game'] = rec / games if games > 0 else 0
                
            elif position in ['wr', 'te']:
                rec = self.safe_float_conversion(stats.get('REC', 0))
                yds = self.safe_float_conversion(stats.get('YDS', 0))
                tgt = self.safe_float_conversion(stats.get('TGT', 1))
                
                feature_dict['yards_per_reception'] = yds / rec if rec > 0 else 0
                feature_dict['catch_rate'] = rec / tgt if tgt > 0 else 0
                feature_dict['targets_per_game'] = tgt / games if games > 0 else 0
                feature_dict['receptions_per_game'] = rec / games if games > 0 else 0
        else:
            # Use position averages if no stats data
            avg_stats = pos_avg.get('stats', {})
            feature_dict['fantasy_ppg'] = 0
            feature_dict['games_played'] = avg_stats.get('G', 16)
        
        # COMPREHENSIVE Advanced metrics features - using ALL available metrics
        advanced = player_data.get('advanced', {})
        if advanced:
            # Map ALL advanced metrics to feature names (comprehensive mapping)
            advanced_mapping = {
                # Core efficiency metrics
                'Y/A': 'y_per_a',
                'RTG': 'rtg', 
                'BRKTKL': 'brktkl',
                'YAC/R': 'yac_per_r',
                'AIR/A': 'air_per_a',
                
                # QB-specific advanced metrics
                'PCT': 'completion_pct_advanced',
                'AIR': 'air_yards',
                'PKT TIME': 'pocket_time',
                'SACK': 'sack_rate',
                'KNCK': 'knockdown_rate',
                'HRRY': 'pressure_rate',
                'BLITZ': 'blitz_rate',
                'POOR': 'poor_throw_rate',
                'DROP': 'drop_rate',
                
                # Explosive play metrics
                '10+ YDS': 'passes_10_plus',
                '20+ YDS': 'passes_20_plus',
                '30+ YDS': 'passes_30_plus',
                '40+ YDS': 'passes_40_plus',
                '50+ YDS': 'passes_50_plus',
                
                # RB-specific advanced metrics
                'Y/ATT': 'yards_per_attempt_rb',
                'YBCON/ATT': 'yards_before_contact_per_attempt',
                'YACON/ATT': 'yards_after_contact_per_attempt',
                'TK LOSS': 'tackled_for_loss',
                'TK LOSS YDS': 'tackled_for_loss_yards',
                'LNG TD': 'longest_touchdown',
                'LNG': 'longest_run',
                
                # WR/TE-specific advanced metrics
                'Y/R': 'yards_per_reception_advanced',
                'YBC/R': 'yards_before_catch_per_reception',
                'YACON/R': 'yards_after_contact_per_reception',
                'YBC': 'yards_before_catch',
                'YAC': 'yards_after_catch',
                'YACON': 'yards_after_contact',
                'AIR': 'air_yards_wr',
                'AIR/R': 'air_yards_per_reception',
                '% TM': 'team_target_share',
                'CATCHABLE': 'catchable_targets',
                
                # Volume metrics
                'COMP': 'completions',
                'ATT': 'attempts',
                'YDS': 'total_yards',
                'REC': 'receptions_advanced',
                'TGT': 'targets_advanced'
            }
            
            for adv_key, feature_key in advanced_mapping.items():
                if feature_key in feature_dict:
                    value = self.safe_float_conversion(advanced.get(adv_key, 0))
                    feature_dict[feature_key] = value
        else:
            # Use position averages for advanced metrics
            avg_advanced = pos_avg.get('advanced', {})
            # Same mapping as above for fallback values
            advanced_mapping = {
                'Y/A': 'y_per_a',
                'RTG': 'rtg',
                'BRKTKL': 'brktkl', 
                'YAC/R': 'yac_per_r',
                'AIR/A': 'air_per_a',
                # ... (same comprehensive mapping)
            }
            
            for adv_key, feature_key in advanced_mapping.items():
                if feature_key in feature_dict:
                    feature_dict[feature_key] = avg_advanced.get(adv_key, 0)
        
        # COMPREHENSIVE Redzone features - using ALL available redzone metrics
        redzone = player_data.get('redzone', {})
        if redzone:
            if position == 'qb':
                # QB Redzone - ALL available metrics
                rz_att = self.safe_float_conversion(redzone.get('ATT', 0))  # Pass attempts
                rz_comp = self.safe_float_conversion(redzone.get('COMP', 0))
                rz_yds = self.safe_float_conversion(redzone.get('YDS', 0))
                rz_td = self.safe_float_conversion(redzone.get('TD', 0))
                rz_int = self.safe_float_conversion(redzone.get('INT', 0))
                rz_sacks = self.safe_float_conversion(redzone.get('SACKS', 0))
                
                # Rushing redzone metrics
                rz_rush_att = self.safe_float_conversion(redzone.get('ATT.1', 0))  # Second ATT column
                rz_rush_yds = self.safe_float_conversion(redzone.get('YDS.1', 0))  # Second YDS column
                rz_rush_td = self.safe_float_conversion(redzone.get('TD.1', 0))    # Second TD column
                
                feature_dict['rz_pass_att_per_game'] = rz_att / games if games > 0 else 0
                feature_dict['rz_completion_pct'] = rz_comp / rz_att if rz_att > 0 else 0
                feature_dict['rz_td_rate'] = rz_td / rz_att if rz_att > 0 else 0
                feature_dict['rz_yards_per_attempt'] = rz_yds / rz_att if rz_att > 0 else 0
                feature_dict['rz_int_rate'] = rz_int / rz_att if rz_att > 0 else 0
                feature_dict['rz_sack_rate'] = rz_sacks / rz_att if rz_att > 0 else 0
                feature_dict['rz_rush_att_per_game'] = rz_rush_att / games if games > 0 else 0
                feature_dict['rz_rush_td_rate'] = rz_rush_td / rz_rush_att if rz_rush_att > 0 else 0
                
            elif position == 'rb':
                # RB Redzone - ALL available metrics
                rz_att = self.safe_float_conversion(redzone.get('ATT', 0))
                rz_yds = self.safe_float_conversion(redzone.get('YDS', 0))
                rz_td = self.safe_float_conversion(redzone.get('TD', 0))
                rz_pct = self.safe_float_conversion(redzone.get('PCT', 0))
                rz_rec = self.safe_float_conversion(redzone.get('REC', 0))
                rz_tgt = self.safe_float_conversion(redzone.get('TGT', 0))
                rz_rec_pct = self.safe_float_conversion(redzone.get('REC PCT', 0))
                rz_rec_yds = self.safe_float_conversion(redzone.get('YDS.1', 0))  # Receiving yards
                rz_rec_ypr = self.safe_float_conversion(redzone.get('Y/R', 0))
                rz_rec_td = self.safe_float_conversion(redzone.get('TD.1', 0))    # Receiving TD
                rz_tgt_pct = self.safe_float_conversion(redzone.get('TGT PCT', 0))
                
                feature_dict['rz_carries_per_game'] = rz_att / games if games > 0 else 0
                feature_dict['rz_rush_td_rate'] = rz_td / rz_att if rz_att > 0 else 0
                feature_dict['rz_rush_yards_per_attempt'] = rz_yds / rz_att if rz_att > 0 else 0
                feature_dict['rz_receptions_per_game'] = rz_rec / games if games > 0 else 0
                feature_dict['rz_receiving_td_rate'] = rz_rec_td / rz_tgt if rz_tgt > 0 else 0
                feature_dict['rz_target_share'] = rz_tgt_pct / 100.0 if rz_tgt_pct > 0 else 0
                
            elif position in ['wr', 'te']:
                # WR/TE Redzone - ALL available metrics
                rz_rec = self.safe_float_conversion(redzone.get('REC', 0))
                rz_tgt = self.safe_float_conversion(redzone.get('TGT', 0))
                rz_rec_pct = self.safe_float_conversion(redzone.get('REC PCT', 0))
                rz_yds = self.safe_float_conversion(redzone.get('YDS', 0))
                rz_ypr = self.safe_float_conversion(redzone.get('Y/R', 0))
                rz_td = self.safe_float_conversion(redzone.get('TD', 0))
                rz_tgt_pct = self.safe_float_conversion(redzone.get('TGT PCT', 0))
                rz_rush_att = self.safe_float_conversion(redzone.get('ATT', 0))
                rz_rush_yds = self.safe_float_conversion(redzone.get('YDS.1', 0))
                rz_rush_td = self.safe_float_conversion(redzone.get('TD.1', 0))
                
                feature_dict['rz_targets_per_game'] = rz_tgt / games if games > 0 else 0
                feature_dict['rz_receptions_per_game'] = rz_rec / games if games > 0 else 0
                feature_dict['rz_receiving_td_rate'] = rz_td / rz_tgt if rz_tgt > 0 else 0
                feature_dict['rz_yards_per_reception'] = rz_yds / rz_rec if rz_rec > 0 else 0
                feature_dict['rz_target_share'] = rz_tgt_pct / 100.0 if rz_tgt_pct > 0 else 0
                feature_dict['rz_rush_att_per_game'] = rz_rush_att / games if games > 0 else 0
                feature_dict['rz_rush_td_rate'] = rz_rush_td / rz_rush_att if rz_rush_att > 0 else 0
        else:
            # Use position average for redzone
            avg_redzone = pos_avg.get('redzone', {})
            if position == 'qb':
                feature_dict['rz_pass_att_per_game'] = avg_redzone.get('RZ ATT', 0) / 16
            elif position == 'rb':
                feature_dict['rz_carries_per_game'] = avg_redzone.get('RZ ATT', 0) / 16
            elif position in ['wr', 'te']:
                feature_dict['rz_targets_per_game'] = avg_redzone.get('RZ TGT', 0) / 16
        
        # Enhanced Injury features - using ALL available injury metrics
        injury_data = player_data.get('injury', {})
        if injury_data:
            feature_dict['injury_risk_score'] = injury_data.get('injury_risk_per_game', 0) / 100.0
            feature_dict['career_injuries'] = injury_data.get('career_injuries', 0)
            feature_dict['injuries_per_season'] = injury_data.get('injuries_per_season', 0)
            feature_dict['projected_games_missed'] = injury_data.get('projected_games_missed', 0)
            feature_dict['durability_score'] = injury_data.get('durability', 5) / 5.0
        else:
            # Default injury values
            feature_dict['injury_risk_score'] = 0.02  # 2% default risk
            feature_dict['career_injuries'] = 0
            feature_dict['injuries_per_season'] = 0
            feature_dict['projected_games_missed'] = 0
            feature_dict['durability_score'] = 1.0
        
        # Age feature
        age = player_data.get('age', 25)
        feature_dict['age'] = age
        
        # Durability feature (games played / season length)
        season_length = 17 if year >= 2021 else 16
        durability = feature_dict['games_played'] / season_length if feature_dict['games_played'] > 0 else 0
        feature_dict['durability'] = durability
        
        # Convert to vector with length assertion
        features = [feature_dict[feature] for feature in FEATURE_SCHEMAS[position]]
        
        # CRITICAL: Assert feature length consistency
        expected_length = len(FEATURE_SCHEMAS[position])
        actual_length = len(features)
        
        if actual_length != expected_length:
            raise ValueError(
                f"Feature length mismatch for {position}: expected {expected_length}, got {actual_length}. "
                f"Features: {features}"
            )
        
        return np.array(features)

    # Replace the old extract_features method
    def extract_features(self, player_data: Dict, position: str, year: int, player_name: str = "Unknown") -> np.ndarray:
        """
        Wrapper for the new schema-based feature extraction.
        """
        return self.extract_features_safe(player_data, position, year, player_name)
    
    # IMPROVEMENT #1: Add ADP Baseline and Spearman Rank Evaluation
    def adp_baseline(self, y_true: np.ndarray, adp_series: pd.Series) -> np.ndarray:
        """Convert ADP ranks to predicted PPG scale for baseline comparison."""
        # Invert ADP ranks (lower rank = higher expected PPG)
        z = -adp_series.fillna(adp_series.median())
        # Standardize to same scale as y_true
        z = (z - z.mean()) / (z.std() + 1e-6)
        return z * y_true.std() + y_true.mean()
    
    # IMPROVEMENT #4: Fixed ADP Player Alignment using DataLoader
    def get_adp_for_validation_fixed(self, data: Dict, val_players: List[str], position: str, val_year: int) -> pd.Series:
        """
        Get ADP data for validation players using consistent DataLoader matching logic.
        
        Args:
            data: Data dictionary
            val_players: List of validation player names (in order)
            position: Player position
            val_year: Year for ADP data
            
        Returns:
            Series with ADP ranks aligned exactly with val_players list
        """
        adp_ranks = []
        
        if val_year in data['adp']:
            adp_df = data['adp'][val_year]
            
            # Use DataLoader's player matching logic for consistency
            for player in val_players:
                player_found = False
                
                # Use the same matching logic as DataLoader
                for _, adp_row in adp_df.iterrows():
                    adp_name = self.data_loader.clean_player_name_for_matching(adp_row.get('Player', ''))
                    adp_pos = str(adp_row.get('POS', '')).upper()
                    
                    # Use exact same matching logic as other data sources
                    if (player == adp_name) and position.upper() in adp_pos:
                        adp_rank = self.data_loader.safe_float_conversion(adp_row.get('AVG', adp_row.get('Rank', 999)))
                        adp_ranks.append(adp_rank)
                        player_found = True
                        break
                
                if not player_found:
                    # Use median ADP rank for missing players
                    median_rank = adp_df['AVG'].median() if 'AVG' in adp_df.columns else 999
                    adp_ranks.append(median_rank)
                    print(f"    ⚠️  No ADP found for {player} ({position}), using median rank: {median_rank}")
        else:
            # Use default ranks if no ADP data
            adp_ranks = [999] * len(val_players)
            print(f"    ⚠️  No ADP data found for {val_year}")
        
        # CRITICAL: Ensure exact alignment with val_players
        if len(adp_ranks) != len(val_players):
            raise ValueError(
                f"ADP alignment mismatch: {len(adp_ranks)} ADP ranks vs {len(val_players)} validation players"
            )
        
        return pd.Series(adp_ranks, index=val_players)

    # Keep the old method for backward compatibility but mark as deprecated
    def get_adp_for_validation(self, data: Dict, val_players: List[str], position: str, val_year: int) -> pd.Series:
        """
        DEPRECATED: Use get_adp_for_validation_fixed() instead.
        """
        print("    ⚠️  Using deprecated ADP matching method. Consider using get_adp_for_validation_fixed()")
        return self.get_adp_for_validation_fixed(data, val_players, position, val_year)
    
    def prepare_training_data_with_seasons(self, data: Dict, position: str, train_years: List[int], val_years: List[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with proper temporal splits by season."""
        X, y, player_names = [], [], []
        
        # Determine which years to process
        if val_years is None:
            # Training mode: use train_years to predict next year
            process_years = train_years
        else:
            # Validation mode: use val_years to predict next year
            process_years = val_years
        
        print(f"    Processing years: {process_years}")
        
        for year in process_years:
            next_year = year + 1
            
            print(f"      Processing {year}→{next_year} for {position}")
            
            # Get current year stats
            if position not in data['stats'] or year not in data['stats'][position]:
                print(f"        ❌ No stats data for {position} {year}")
                continue
                
            current_stats = data['stats'][position][year]
            print(f"        ✓ Found {len(current_stats)} {position} stats for {year}")
            
            # Get next year results
            if next_year not in data['results']:
                print(f"        ❌ No results data for {next_year}")
                continue
                
            next_results = data['results'][next_year]
            print(f"        ✓ Found {len(next_results)} results for {next_year}")
            
            # Process each player
            matched_players = 0
            sample_stats_players = []
            sample_results_players = []
            
            for _, player_row in current_stats.iterrows():
                # Use the cleaned player name if available, otherwise clean it
                if 'Player_Clean' in player_row:
                    player_name = player_row['Player_Clean']
                else:
                    raw_name = player_row.get('Player', '')
                    player_name = self.clean_player_name(raw_name)
                
                # Debug first few names
                if len(sample_stats_players) < 3:
                    raw_name = player_row.get('Player', '')
                    print(f"        DEBUG STATS: '{raw_name}' -> '{player_name}'")
                
                if not player_name:
                    continue
                
                # Collect sample names for debugging
                if len(sample_stats_players) < 5:
                    sample_stats_players.append(player_name)
                
                # Find player in next year's results
                next_player = None
                for _, result_row in next_results.iterrows():
                    result_name = self.clean_player_name(result_row.get('Player', ''))
                    
                    # Collect sample names for debugging
                    if len(sample_results_players) < 5:
                        sample_results_players.append(result_name)
                    
                    # Use exact match since both names are cleaned the same way
                    if player_name == result_name:
                        next_player = result_row
                        break
                
                if next_player is None:
                    continue
                
                matched_players += 1
                
                # Extract features from current year
                player_data = {
                    'stats': player_row.to_dict(),
                    'advanced': {},
                    'redzone': {},
                    'age': 25  # Default age
                }
                
                # Add advanced metrics if available
                if position in data['advanced'] and year in data['advanced'][position]:
                    adv_df = data['advanced'][position][year]
                    for _, adv_row in adv_df.iterrows():
                        adv_name = self.clean_player_name(adv_row.get('PLAYER', adv_row.get('Player', '')))
                        if player_name in adv_name or adv_name in player_name:
                            player_data['advanced'] = adv_row.to_dict()
                            break
                
                # Add redzone data if available
                if position in data['redzone'] and year in data['redzone'][position]:
                    rz_df = data['redzone'][position][year]
                    for _, rz_row in rz_df.iterrows():
                        rz_name = self.clean_player_name(rz_row.get('PLAYER', rz_row.get('Player', '')))
                        if player_name in rz_name or rz_name in player_name:
                            player_data['redzone'] = rz_row.to_dict()
                            break
                
                # Add age if available
                if data['age'] is not None:
                    for _, age_row in data['age'].iterrows():
                        age_name = self.clean_player_name(age_row.get('PLAYER NAME', ''))
                        if player_name in age_name or age_name in player_name:
                            try:
                                player_data['age'] = float(age_row.get('AGE', 25))
                            except:
                                player_data['age'] = 25
                            break
                
                # Extract features using schema-based approach
                features = self.extract_features_safe(player_data, position, year, player_name)
                
                # Get target (next year's PPG)
                next_fpts = self.safe_float_conversion(next_player.get('AVG', next_player.get('FPTS/G', 0)))
                
                if len(features) > 0 and next_fpts > 0:
                    X.append(features)
                    y.append(next_fpts)
                    player_names.append(player_name)
            
            # Print sample names for debugging
            if len(sample_stats_players) > 0 and len(sample_results_players) > 0:
                print(f"        Sample stats players: {sample_stats_players[:3]}")
                print(f"        Sample results players: {sample_results_players[:3]}")
            elif len(sample_stats_players) > 0:
                print(f"        Sample stats players: {sample_stats_players[:3]}")
                print(f"        No results players found")
            elif len(sample_results_players) > 0:
                print(f"        No stats players found")
                print(f"        Sample results players: {sample_results_players[:3]}")
            else:
                print(f"        No players found in either dataset")
            
            print(f"        ✓ Matched {matched_players} players for {year}→{next_year}")
        
        print(f"    Generated {len(X)} samples from {len(process_years)} seasons")
        
        if X:
            X_array = np.array(X)
            # CRITICAL: Validate feature consistency
            stage_name = "validation" if val_years else "training"
            self.validate_feature_consistency(X_array, position, stage_name)
            return X_array, np.array(y), player_names
        else:
            return np.array([]), np.array([]), []
    
    # IMPROVEMENT #3: Increase Model Capacity
    def train_position_model_with_validation(self, position: str, data: Dict) -> Dict:
        """Train model for a specific position with proper temporal validation."""
        print(f"\nTraining {position.upper()} model with temporal validation...")
        
        # PHASE 1: Validation (Train on 2019-2023, validate on 2023→2024)
        print(f"  Phase 1: Validation")
        print(f"    Training on: 2019→2020, 2020→2021, 2021→2022, 2022→2023")
        print(f"    Validating on: 2023→2024")
        
        # FIXED: Calculate position averages using only training data (2019-2022)
        print(f"    Calculating position averages using 2019-2022 data...")
        self.position_averages = self.calculate_position_averages(data, max_year=2022)
        
        # Prepare training data (2019-2023)
        X_train, y_train, train_players = self.prepare_training_data_with_seasons(
            data, position, train_years=[2019, 2020, 2021, 2022]
        )
        
        # Prepare validation data (2023→2024)
        X_val, y_val, val_players = self.prepare_training_data_with_seasons(
            data, position, train_years=[2023], val_years=[2023]
        )
        
        if len(X_train) < 20:
            print(f"  ❌ Insufficient training data for {position}: {len(X_train)} samples")
            return {}
        
        if len(X_val) < 5:
            print(f"  ❌ Insufficient validation data for {position}: {len(X_val)} samples")
            return {}
        
        print(f"  ✓ Training samples: {len(X_train)}")
        print(f"  ✓ Validation samples: {len(X_val)}")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # IMPROVEMENT #3: Increased Model Capacity
        model = RandomForestRegressor(
            n_estimators=400,      # More trees for better performance
            max_depth=None,        # Let trees grow deep
            min_samples_leaf=2,    # Prevent overfitting
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val_scaled)
        val_mse = mean_squared_error(y_val, y_val_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        
        # IMPROVEMENT #4: Fixed ADP Player Alignment using DataLoader
        adp_series = self.get_adp_for_validation_fixed(data, val_players, position, 2024)
        adp_baseline_pred = self.adp_baseline(y_val, adp_series)
        adp_baseline_mse = mean_squared_error(y_val, adp_baseline_pred)
        adp_baseline_r2 = r2_score(y_val, adp_baseline_pred)
        
        # Calculate Spearman rank correlation
        spearman_corr = float(spearmanr(y_val, y_val_pred).correlation)
        adp_spearman_corr = float(spearmanr(y_val, adp_baseline_pred).correlation)
        
        # Calculate baseline on validation set (last year's PPG)
        baseline_val_pred = X_val[:, 0]  # First feature is PPG
        baseline_val_mse = mean_squared_error(y_val, baseline_val_pred)
        baseline_val_r2 = r2_score(y_val, baseline_val_pred)
        
        # IMPROVEMENT #4: Add divide-by-zero protection
        improvement = 0.0 if baseline_val_mse == 0 else (baseline_val_mse - val_mse) / baseline_val_mse * 100
        adp_improvement = 0.0 if adp_baseline_mse == 0 else (adp_baseline_mse - val_mse) / adp_baseline_mse * 100
        
        print(f"  ✓ Validation R²: {val_r2:.3f}, MSE: {val_mse:.3f}")
        print(f"  ✓ Baseline R²: {baseline_val_r2:.3f}, MSE: {baseline_val_mse:.3f}")
        print(f"  ✓ ADP Baseline R²: {adp_baseline_r2:.3f}, MSE: {adp_baseline_mse:.3f}")
        print(f"  ✓ Spearman Correlation: {spearman_corr:.3f}")
        print(f"  ✓ ADP Spearman Correlation: {adp_spearman_corr:.3f}")
        print(f"  ✓ Improvement vs Baseline: {improvement:.1f}%")
        print(f"  ✓ Improvement vs ADP: {adp_improvement:.1f}%")
        
        # Store validation results
        validation_results = {
            'val_r2': val_r2,
            'val_mse': val_mse,
            'baseline_val_r2': baseline_val_r2,
            'baseline_val_mse': baseline_val_mse,
            'adp_baseline_r2': adp_baseline_r2,
            'adp_baseline_mse': adp_baseline_mse,
            'spearman_corr': spearman_corr,
            'adp_spearman_corr': adp_spearman_corr,
            'improvement': improvement,
            'adp_improvement': adp_improvement,
            'val_predictions': list(zip(val_players, y_val, y_val_pred, adp_baseline_pred))
        }
        
        # PHASE 2: Final Training (All data for 2025 predictions)
        print(f"  Phase 2: Final Training")
        print(f"    Training on: 2019→2020, 2020→2021, 2021→2022, 2022→2023, 2023→2024")
        
        # FIXED: Calculate position averages using all available training data (2019-2023)
        print(f"    Calculating position averages using 2019-2023 data...")
        self.position_averages = self.calculate_position_averages(data, max_year=2023)
        
        # Prepare final training data (all years)
        X_final, y_final, final_players = self.prepare_training_data_with_seasons(
            data, position, train_years=[2019, 2020, 2021, 2022, 2023]
        )
        
        print(f"  ✓ Final training samples: {len(X_final)}")
        
        # Scale features for final model
        final_scaler = StandardScaler()
        X_final_scaled = final_scaler.fit_transform(X_final)
        
        # IMPROVEMENT #3: Increased Model Capacity for final model too
        final_model = RandomForestRegressor(
            n_estimators=400,      # More trees for better performance
            max_depth=None,        # Let trees grow deep
            min_samples_leaf=2,    # Prevent overfitting
            random_state=42,
            n_jobs=-1
        )
        
        final_model.fit(X_final_scaled, y_final)
        
        # Store results
        results = {
            'model': final_model,  # Use final model for predictions
            'scaler': final_scaler,
            'feature_names': self.get_feature_names(position),
            'validation': validation_results,
            'performance': {
                'val_r2': val_r2,
                'val_mse': val_mse,
                'baseline_val_r2': baseline_val_r2,
                'baseline_val_mse': baseline_val_mse,
                'adp_baseline_r2': adp_baseline_r2,
                'adp_baseline_mse': adp_baseline_mse,
                'spearman_corr': spearman_corr,
                'adp_spearman_corr': adp_spearman_corr,
                'improvement': improvement,
                'adp_improvement': adp_improvement
            }
        }
        
        return results
    
    def get_feature_names(self, position: str) -> List[str]:
        """Get feature names for a position using the schema."""
        if position not in FEATURE_SCHEMAS:
            raise ValueError(f"Unknown position: {position}")
        
        return FEATURE_SCHEMAS[position].copy()
    
    def analyze_feature_importance(self, position: str) -> pd.DataFrame:
        """
        Analyze feature importance for a trained model.
        
        Args:
            position: Position to analyze ('qb', 'rb', 'wr', 'te')
            
        Returns:
            DataFrame with feature names and importance scores, sorted by importance
        """
        if position not in self.models:
            print(f"❌ No trained model found for {position}")
            return pd.DataFrame()
        
        if position not in self.feature_names:
            print(f"❌ No feature names found for {position}")
            return pd.DataFrame()
        
        # Get feature importances from the trained model
        importances = self.models[position].feature_importances_
        feature_names = self.feature_names[position]
        
        # Create DataFrame with feature names and importance scores
        feature_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Print top 10 features
        print(f"\nTop 10 {position.upper()} Features:")
        print("-" * 50)
        for i, (_, row) in enumerate(feature_df.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Print bottom 10 features (least important)
        print(f"\nBottom 10 {position.upper()} Features:")
        print("-" * 50)
        for i, (_, row) in enumerate(feature_df.tail(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<35} {row['importance']:.4f}")
        
        # Calculate importance statistics
        total_importance = feature_df['importance'].sum()
        top_10_importance = feature_df.head(10)['importance'].sum()
        top_10_percentage = (top_10_importance / total_importance) * 100
        
        print(f"\nFeature Importance Summary for {position.upper()}:")
        print(f"  Total features: {len(feature_df)}")
        print(f"  Top 10 features account for: {top_10_percentage:.1f}% of total importance")
        print(f"  Most important feature: {feature_df.iloc[0]['feature']} ({feature_df.iloc[0]['importance']:.4f})")
        print(f"  Least important feature: {feature_df.iloc[-1]['feature']} ({feature_df.iloc[-1]['importance']:.4f})")
        
        return feature_df
    
    def analyze_all_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze feature importance for all trained models.
        
        Returns:
            Dictionary with position as key and feature importance DataFrame as value
        """
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        all_importance = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            if position in self.models:
                print(f"\nAnalyzing {position.upper()} feature importance...")
                importance_df = self.analyze_feature_importance(position)
                if not importance_df.empty:
                    all_importance[position] = importance_df
                    
                    # Save to CSV
                    output_file = f"feature_importance_{position}.csv"
                    importance_df.to_csv(output_file, index=False)
                    print(f"  ✓ Saved {position.upper()} feature importance to {output_file}")
        
        # Create combined feature importance summary
        if all_importance:
            combined_summary = []
            for position, df in all_importance.items():
                top_features = df.head(5)[['feature', 'importance']].copy()
                top_features['position'] = position.upper()
                combined_summary.append(top_features)
            
            if combined_summary:
                combined_df = pd.concat(combined_summary, ignore_index=True)
                combined_df.to_csv("feature_importance_summary.csv", index=False)
                print(f"\n  ✓ Saved combined feature importance summary to feature_importance_summary.csv")
        
        return all_importance
    
    # IMPROVEMENT #2: Add 2025 Inference Method
    def infer_2025(self, data: Dict, position: str) -> pd.DataFrame:
        """Generate 2025 predictions for a position using trained model."""
        if position not in self.models:
            print(f"❌ No trained model found for {position}")
            return pd.DataFrame()
        
        print(f"\nGenerating 2025 predictions for {position.upper()}...")
        
        # Get 2024 stats for all players
        if position not in data['stats'] or 2024 not in data['stats'][position]:
            print(f"❌ No 2024 stats found for {position}")
            return pd.DataFrame()
        
        stats_2024 = data['stats'][position][2024]
        predictions = []
        
        for _, player_row in stats_2024.iterrows():
            player_name = self.clean_player_name(player_row.get('Player', ''))
            if not player_name:
                continue
            
            # Extract features from 2024 data
            player_data = {
                'stats': player_row.to_dict(),
                'advanced': {},
                'redzone': {},
                'age': 25  # Default age
            }
            
            # Add advanced metrics if available
            if position in data['advanced'] and 2024 in data['advanced'][position]:
                adv_df = data['advanced'][position][2024]
                for _, adv_row in adv_df.iterrows():
                    adv_name = self.clean_player_name(adv_row.get('PLAYER', adv_row.get('Player', '')))
                    if player_name in adv_name or adv_name in player_name:
                        player_data['advanced'] = adv_row.to_dict()
                        break
            
            # Add redzone data if available
            if position in data['redzone'] and 2024 in data['redzone'][position]:
                rz_df = data['redzone'][position][2024]
                for _, rz_row in rz_df.iterrows():
                    rz_name = self.clean_player_name(rz_row.get('PLAYER', rz_row.get('Player', '')))
                    if player_name in rz_name or rz_name in player_name:
                        player_data['redzone'] = rz_row.to_dict()
                        break
            
            # Add age if available
            if data['age'] is not None:
                for _, age_row in data['age'].iterrows():
                    age_name = self.clean_player_name(age_row.get('PLAYER NAME', ''))
                    if player_name in age_name or age_name in player_name:
                        try:
                            player_data['age'] = float(age_row.get('AGE', 25))
                        except:
                            player_data['age'] = 25
                        break
            
            # Extract features
            features = self.extract_features_safe(player_data, position, 2024, player_name)
            
            if len(features) > 0:
                # Scale features
                features_scaled = self.scalers[position].transform([features])
                
                # Make prediction
                pred_ppg = self.models[position].predict(features_scaled)[0]
                
                predictions.append({
                    'player_name': player_name,
                    'pred_ppg_2025': pred_ppg,
                    'pos_rank_2025': 0  # Will be set after sorting
                })
        
        # Create DataFrame and sort by predicted PPG
        if predictions:
            df = pd.DataFrame(predictions)
            df = df.sort_values('pred_ppg_2025', ascending=False).reset_index(drop=True)
            df['pos_rank_2025'] = df.index + 1
            
            print(f"  ✓ Generated predictions for {len(df)} {position.upper()} players")
            return df
        else:
            print(f"  ❌ No predictions generated for {position}")
            return pd.DataFrame()
    
    def generate_all_2025_predictions(self, data: Dict) -> Dict[str, pd.DataFrame]:
        """Generate 2025 predictions for all positions."""
        print("\n" + "="*60)
        print("GENERATING 2025 PREDICTIONS")
        print("="*60)
        
        all_predictions = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            predictions_df = self.infer_2025(data, position)
            if not predictions_df.empty:
                all_predictions[position] = predictions_df
                
                # Save to CSV
                output_file = f"predictions_2025_{position}.csv"
                predictions_df.to_csv(output_file, index=False)
                print(f"  ✓ Saved {position.upper()} predictions to {output_file}")
        
        # Create combined rankings
        if all_predictions:
            combined_df = pd.concat([
                df.assign(position=pos.upper()) for pos, df in all_predictions.items()
            ], ignore_index=True)
            
            # Sort by predicted PPG for overall ranking
            combined_df = combined_df.sort_values('pred_ppg_2025', ascending=False).reset_index(drop=True)
            combined_df['overall_rank_2025'] = combined_df.index + 1
            
            # Save combined rankings
            combined_df.to_csv("predictions_2025_combined.csv", index=False)
            print(f"  ✓ Saved combined rankings to predictions_2025_combined.csv")
            
            all_predictions['combined'] = combined_df
        
        return all_predictions
    
    def train_all_models(self):
        """Train models for all positions with proper temporal validation."""
        print("="*60)
        print("SIMPLE ML SYSTEM TRAINING WITH TEMPORAL VALIDATION")
        print("="*60)
        
        data = self.load_data()
        
        print(f"\nTraining models with temporal validation...")
        for position in ['qb', 'rb', 'wr', 'te']:
            results = self.train_position_model_with_validation(position, data)
            if results:
                self.models[position] = results['model']
                self.scalers[position] = results['scaler']
                self.feature_names[position] = results['feature_names']
                self.performance_metrics[position] = results['performance']
                self.validation_results[position] = results['validation']
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        
        self.print_performance_summary()
        self.save_validation_results()
        
        # NEW: Analyze feature importance after training
        self.analyze_all_feature_importance()
        
        # Save debug logs from data loader
        self.data_loader.save_debug_logs("ml_training")
        
        # IMPROVEMENT #2: Generate 2025 predictions after training
        self.generate_all_2025_predictions(data)
    
    def print_performance_summary(self):
        """Print performance summary for all models."""
        print("\nMODEL PERFORMANCE SUMMARY:")
        print("-" * 60)
        
        for position, metrics in self.performance_metrics.items():
            print(f"\n{position.upper()}:")
            print(f"  Validation R²: {metrics['val_r2']:.3f}")
            print(f"  Validation MSE: {metrics['val_mse']:.3f}")
            print(f"  Baseline R²: {metrics['baseline_val_r2']:.3f}")
            print(f"  ADP Baseline R²: {metrics['adp_baseline_r2']:.3f}")
            print(f"  Spearman Correlation: {metrics['spearman_corr']:.3f}")
            print(f"  ADP Spearman Correlation: {metrics['adp_spearman_corr']:.3f}")
            print(f"  Improvement vs Baseline: {metrics['improvement']:.1f}%")
            print(f"  Improvement vs ADP: {metrics['adp_improvement']:.1f}%")
    
    def save_validation_results(self):
        """Save detailed validation results."""
        validation_file = f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(validation_file, 'w', encoding='utf-8') as f:
            f.write("TEMPORAL VALIDATION RESULTS (2023->2024)\n")
            f.write("=" * 60 + "\n\n")
            
            for position, results in self.validation_results.items():
                f.write(f"{position.upper()} VALIDATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"R² Score: {results['val_r2']:.3f}\n")
                f.write(f"MSE: {results['val_mse']:.3f}\n")
                f.write(f"Baseline R²: {results['baseline_val_r2']:.3f}\n")
                f.write(f"ADP Baseline R²: {results['adp_baseline_r2']:.3f}\n")
                f.write(f"Spearman Correlation: {results['spearman_corr']:.3f}\n")
                f.write(f"ADP Spearman Correlation: {results['adp_spearman_corr']:.3f}\n")
                f.write(f"Improvement vs Baseline: {results['improvement']:.1f}%\n")
                f.write(f"Improvement vs ADP: {results['adp_improvement']:.1f}%\n\n")
                
                f.write("TOP 10 VALIDATION PREDICTIONS:\n")
                f.write("Player, Actual PPG, Predicted PPG, ADP Baseline, Error\n")
                f.write("-" * 60 + "\n")
                
                # Sort by absolute error
                predictions = sorted(results['val_predictions'], key=lambda x: abs(x[1] - x[2]))
                for player, actual, predicted, adp_baseline in predictions[:10]:
                    error = actual - predicted
                    f.write(f"{player:<25} {actual:6.1f} {predicted:6.1f} {adp_baseline:6.1f} {error:+6.1f}\n")
                f.write("\n")
        
        print(f"\n✓ Validation results saved to: {validation_file}")
    
    def save_models(self, output_dir: str = "ml_models"):
        """Save trained models and metadata."""
        os.makedirs(output_dir, exist_ok=True)
        
        for position in self.models:
            # Save model
            model_path = f"{output_dir}/{position}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(self.models[position], f)
            
            # Save scaler
            scaler_path = f"{output_dir}/{position}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scalers[position], f)
            
            # Save metadata
            metadata = {
                'feature_names': self.feature_names[position],
                'performance': self.performance_metrics[position],
                'validation': self.validation_results[position],
                'position_averages': self.position_averages.get(position, {})
            }
            metadata_path = f"{output_dir}/{position}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
        
        print(f"\n✓ Models saved to {output_dir}/")

# Add this validation method
def validate_feature_consistency(self, X: np.ndarray, position: str, stage: str = "unknown"):
    """
    Validate that feature dimensions are consistent.
    
    Args:
        X: Feature matrix
        position: Position being processed
        stage: Training stage for logging
    """
    if position not in FEATURE_DIMENSIONS:
        raise ValueError(f"Unknown position: {position}")
    
    expected_dim = FEATURE_DIMENSIONS[position]
    
    if X.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch in {stage} for {position}: "
            f"expected {expected_dim}, got {X.shape[1]}. "
            f"Matrix shape: {X.shape}"
        )
    
    print(f"  ✓ {stage} feature validation passed for {position}: {X.shape[1]} features")

# Add this validation method
def validate_adp_alignment(self, val_players: List[str], adp_series: pd.Series, position: str):
    """
    Validate that ADP data aligns correctly with validation players.
    
    Args:
        val_players: List of validation player names
        adp_series: Series of ADP ranks
        position: Position being validated
    """
    if len(val_players) != len(adp_series):
        raise ValueError(
            f"ADP alignment validation failed for {position}: "
            f"{len(val_players)} players vs {len(adp_series)} ADP ranks"
        )
    
    # Check for reasonable ADP values
    median_adp = adp_series.median()
    if median_adp > 500:  # Suspicious if median is too high
        print(f"    ⚠️  Suspicious ADP median for {position}: {median_adp}")
    
    # Report alignment quality
    non_default_adp = (adp_series != 999).sum()
    alignment_rate = non_default_adp / len(val_players) * 100
    
    print(f"    ✓ ADP alignment for {position}: {alignment_rate:.1f}% matched ({non_default_adp}/{len(val_players)})")
    
    if alignment_rate < 50:
        print(f"    ⚠️  Low ADP alignment rate for {position}. Consider improving player matching.")

if __name__ == "__main__":
    trainer = SimpleMLTrainer()
    trainer.train_all_models()
    trainer.save_models()