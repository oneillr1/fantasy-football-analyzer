from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
import json
import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
warnings.filterwarnings('ignore')


class FantasyFootballAnalyzer:
    # Class constants for magic numbers
    SIGNIFICANT_REACH_THRESHOLD = 12
    SIGNIFICANT_VALUE_THRESHOLD = -12
    ADP_OUTPERFORM_THRESHOLD = 15
    BREAKOUT_THRESHOLD = 20
    REGRESSION_THRESHOLD = -20
    MIN_PICKS_FOR_ANALYSIS = 10
    OPTIMAL_POCKET_TIME = 2.7
    
    def __init__(self, data_directory: str, enable_league_analysis: bool = False, enable_testing_mode: bool = False):
        """
        Initialize the analyzer with a directory containing CSV files
        
        Expected file naming convention:
        - draft_YYYY.csv (e.g., draft_2024.csv, draft_2023.csv)
        - adp_YYYY.csv (e.g., adp_2024.csv, adp_2023.csv)
        - results_YYYY.csv (e.g., results_2024.csv, results_2023.csv)
        - advanced_[pos]_YYYY.csv (e.g., advanced_qb_2024.csv, advanced_wr_2023.csv)
        
        Args:
            data_directory: Path to directory containing CSV files
            enable_league_analysis: If True, enables league mate analysis features (default: False)
            enable_testing_mode: If True, enables testing/feedback mode with 3 random players (default: False)
        """
        # Initialize debug logging
        self.debug_log = []
        self.missing_data_log = []
        self.enable_league_analysis = enable_league_analysis
        self.enable_testing_mode = enable_testing_mode
        self.data_dir = Path(data_directory)
        self.draft_data = {}
        self.adp_data = {}
        self.results_data = {}
        self.advanced_data = {}  # New: stores advanced metrics by position and year
        self.injury_data = {}  # New: stores injury predictor data by position
        self.years = []
        
        # ML models for player profiling
        self.ml_models = {
            'breakout': {},
            'consistency': {},
            'positional_value': {},
            'injury_risk': {}
        }
        self.scalers = {}
        self.model_confidence = {}
        
        # Comprehensive advanced metrics by position for analysis
        self.key_metrics = {
            'QB': [
                # Efficiency metrics
                'PCT', 'Y/A', 'RTG', 'AIR/A',
                # Volume metrics  
                'COMP', 'ATT', 'YDS', 'AIR',
                # Explosive play metrics
                '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS',
                # Pressure metrics
                'PKT TIME', 'SACK', 'KNCK', 'HRRY', 'BLITZ',
                # Accuracy metrics
                'POOR', 'DROP',
                # Opportunity metrics
                'RZ ATT'
            ],
            'RB': [
                # Efficiency metrics
                'Y/ATT', 'YBCON/ATT', 'YACON/ATT',
                # Volume metrics
                'ATT', 'YDS', 'REC', 'TGT',
                # Explosive play metrics
                '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS', 'LNG TD', 'LNG',
                # Power/elusiveness metrics
                'BRKTKL', 'TK LOSS', 'TK LOSS YDS',
                # Opportunity metrics
                'RZ TGT'
            ],
            'WR': [
                # Efficiency metrics
                'Y/R', 'YBC/R', 'YAC/R', 'YACON/R', 'AIR/R',
                # Volume metrics
                'REC', 'YDS', 'TGT',
                # Explosive play metrics
                '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS', 'LNG',
                # Opportunity metrics
                '% TM', 'CATCHABLE', 'RZ TGT',
                # Playmaking metrics
                'BRKTKL', 'DROP'
            ],
            'TE': [
                # Efficiency metrics
                'Y/R', 'YBC/R', 'YAC/R', 'YACON/R', 'AIR/R',
                # Volume metrics
                'REC', 'YDS', 'TGT',
                # Explosive play metrics
                '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS', 'LNG',
                # Opportunity metrics
                '% TM', 'CATCHABLE', 'RZ TGT',
                # Playmaking metrics
                'BRKTKL', 'DROP'
            ]
        }
        
        # Half PPR League Scoring Settings
        self.scoring_settings = {
            'passing': {
                'yards_per_point': 25,  # 1 point per 25 yards
                'td_points': 6,         # 6 points per TD (not 4)
                'int_points': -2        # -2 for interceptions
            },
            'rushing': {
                'yards_per_point': 10,  # 1 point per 10 yards
                'td_points': 6,         # 6 points per TD
                'fumble_points': -2     # -2 for fumbles
            },
            'receiving': {
                'yards_per_point': 10,  # 1 point per 10 yards
                'td_points': 6,         # 6 points per TD
                'reception_points': 0.5 # Half PPR - 0.5 points per reception
            },
            'bonuses': {
                'long_td_bonus': 2      # +2 points for 40+ yard TDs (all positions)
            }
        }
        
        # Comprehensive metric descriptions for better analysis
        self.metric_descriptions = {
            # QB Metrics
            'COMP': 'Completions - Total passes completed',
            'ATT': 'Attempts - Total passes thrown',
            'PCT': 'Completion Percentage - QB accuracy',
            'YDS': 'Passing Yards - Total passing production',
            'Y/A': 'Yards per Attempt - QB efficiency',
            'AIR': 'Air Yards - Downfield passing volume',
            'AIR/A': 'Air Yards per Attempt - QB downfield aggression',
            'RTG': 'Passer Rating - Overall QB efficiency',
            'PKT TIME': 'Pocket Time - QB processing speed/protection',
            'SACK': 'Sacks Taken - QB pressure handling',
            'KNCK': 'Knockdowns - Times hit while throwing',
            'HRRY': 'Hurries - Times pressured',
            'BLITZ': 'Times Blitzed - Pressure frequency',
            'POOR': 'Poor Passes - Inaccurate throws',
            'DROP': 'Drops by Receivers - WR/TE drops',
            'RZ ATT': 'Red Zone Attempts - TD opportunity',
            '10+ YDS': '10+ Yard Passes - Explosive plays',
            '20+ YDS': '20+ Yard Passes - Big plays',
            '30+ YDS': '30+ Yard Passes - Deep plays',
            '40+ YDS': '40+ Yard Passes - Very deep plays',
            '50+ YDS': '50+ Yard Passes - Extremely deep plays',
            
            # RB Metrics
            'ATT': 'Attempts - Rushing volume',
            'YDS': 'Yards - Total rushing production',
            'Y/ATT': 'Yards per Attempt - RB efficiency',
            'YBCON': 'Yards Before Contact - O-line quality/vision',
            'YACON': 'Yards After Contact - Power/elusiveness after hit',
            'YBCON/ATT': 'Yards Before Contact per Attempt - Blocking efficiency',
            'YACON/ATT': 'Yards After Contact per Attempt - RB power/elusiveness',
            'BRKTKL': 'Broken Tackles - Elusiveness and power',
            'TK LOSS': 'Tackled for Loss - Negative plays',
            'TK LOSS YDS': 'Tackled for Loss Yards - Negative yardage',
            'LNG TD': 'Longest Touchdown - Big play ability',
            'LNG': 'Longest Run - Best single play',
            '10+ YDS': '10+ Yard Runs - Explosive plays',
            '20+ YDS': '20+ Yard Runs - Big plays',
            '30+ YDS': '30+ Yard Runs - Deep plays',
            '40+ YDS': '40+ Yard Runs - Very deep plays',
            '50+ YDS': '50+ Yard Runs - Extremely deep plays',
            'REC': 'Receptions - Receiving volume',
            'TGT': 'Targets - Receiving opportunity',
            'RZ TGT': 'Red Zone Targets - TD opportunity',
            
            # WR/TE Metrics
            'REC': 'Receptions - Total catches',
            'YDS': 'Receiving Yards - Total receiving production',
            'Y/R': 'Yards per Reception - Big play ability',
            'YBC': 'Yards Before Catch - Route running/separation',
            'YBC/R': 'Yards Before Catch per Reception - Route efficiency',
            'AIR': 'Air Yards - Downfield target depth',
            'AIR/R': 'Air Yards per Reception - Downfield efficiency',
            'YAC': 'Yards After Catch - Playmaking ability',
            'YAC/R': 'Yards After Catch per Reception - Playmaking efficiency',
            'YACON': 'Yards After Contact - Power after catch',
            'YACON/R': 'Yards After Contact per Reception - Power efficiency',
            'BRKTKL': 'Broken Tackles - Elusiveness after catch',
            'TGT': 'Targets - Receiving opportunity',
            '% TM': 'Team Target Share - Volume opportunity',
            'CATCHABLE': 'Catchable Targets - QB accuracy to player',
            'DROP': 'Drops - Missed catchable passes',
            'RZ TGT': 'Red Zone Targets - TD opportunity',
            '10+ YDS': '10+ Yard Receptions - Explosive plays',
            '20+ YDS': '20+ Yard Receptions - Big plays',
            '30+ YDS': '30+ Yard Receptions - Deep plays',
            '40+ YDS': '40+ Yard Receptions - Very deep plays',
            '50+ YDS': '50+ Yard Receptions - Extremely deep plays',
            'LNG': 'Longest Reception - Best single play'
        }
        
        # Load data and train ML models upon initialization
        self.load_data()
        
    def diagnose_csv_files(self) -> None:
        """Diagnose CSV files for common issues"""
        print("DIAGNOSING CSV FILES")
        print("="*40)
        
        for year in self.years:
            print(f"\nChecking {year}...")
            
            # Check each file type
            files_to_check = [
                (f"draft_{year}.csv", "Draft"),
                (f"adp_{year}.csv", "ADP"),
                (f"results_{year}.csv", "Results"),
                (f"advanced_qb_{year}.csv", "QB Advanced"),
                (f"advanced_rb_{year}.csv", "RB Advanced"),
                (f"advanced_wr_{year}.csv", "WR Advanced"),
                (f"advanced_te_{year}.csv", "TE Advanced")
            ]
            
            for filename, file_type in files_to_check:
                file_path = self.data_dir / filename
                if file_path.exists():
                    try:
                        # Check file structure
                        with open(file_path, 'r', encoding='utf-8') as f:
                            first_line = f.readline().strip()
                            lines = f.readlines()
                        
                        expected_cols = len(first_line.split(','))
                        
                        problematic_lines = []
                        for i, line in enumerate(lines[:20], 2):  # Check first 20 lines
                            actual_cols = len(line.strip().split(','))
                            if actual_cols != expected_cols:
                                problematic_lines.append((i, actual_cols, expected_cols))
                        
                        if problematic_lines:
                            print(f"  CHECK: Found {len(problematic_lines)} problematic lines")
                            for line_num, actual, expected in problematic_lines[:3]:
                                print(f"     Line {line_num}: {actual} cols (expected {expected})")
                        else:
                            print(f"   {file_type}: Structure looks good")
                            
                    except Exception as e:
                        print(f"  ❌ {file_type}: Error reading file - {e}")
                else:
                    print(f"  - {file_type}: File not found")

    def safe_read_csv(self, file_path: Path, file_type: str, year: int) -> Optional[pd.DataFrame]:
        """Safely read CSV files with error handling and data cleaning"""
        try:
            # Try reading with different parameters to handle malformed CSVs
            df = pd.read_csv(file_path, encoding='utf-8', skipinitialspace=True)
            print(f" Loaded {file_type} data for {year}: {len(df)} records")
            return df
        except pd.errors.ParserError as e:
            print(f"  Parser error in {file_path}: {e}")
            try:
                # Try with error_bad_lines=False for older pandas or on_bad_lines='skip' for newer
                try:
                    df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', skipinitialspace=True)
                except TypeError:
                    # Fallback for older pandas versions
                    df = pd.read_csv(file_path, encoding='utf-8', error_bad_lines=False, skipinitialspace=True)
                print(f"  Loaded {file_type} data for {year} with some rows skipped: {len(df)} records")
                return df
            except Exception as e2:
                print(f"❌ Could not load {file_path}: {e2}")
                return None
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            return None

    def load_data(self) -> None:
        """Load all CSV files from the data directory"""
        print("Loading data files...")
        
        # Find all years based on draft files with validation
        draft_files = list(self.data_dir.glob("draft_*.csv"))
        years = []
        for f in draft_files:
            try:
                parts = f.stem.split('_')
                if len(parts) >= 2:
                    year = int(parts[1])
                    if 2000 <= year <= 2030:  # Reasonable year range
                        years.append(year)
                    else:
                        print(f"  Skipping file with invalid year: {f.name}")
                else:
                    print(f"  Skipping file with invalid format: {f.name}")
            except (ValueError, IndexError) as e:
                print(f"  Skipping malformed filename: {f.name} - {e}")
        
        self.years = sorted(years)
        
        print(f"Found data for years: {self.years}")
        
        for year in self.years:
            print(f"\nProcessing {year}...")
            
            # Load draft data
            draft_file = self.data_dir / f"draft_{year}.csv"
            if draft_file.exists():
                df = self.safe_read_csv(draft_file, "draft", year)
                if df is not None:
                    self.draft_data[year] = df
            
            # Load ADP data
            adp_file = self.data_dir / f"adp_{year}.csv"
            if adp_file.exists():
                df = self.safe_read_csv(adp_file, "ADP", year)
                if df is not None:
                    self.adp_data[year] = df
            
            # Load results data
            results_file = self.data_dir / f"results_{year}.csv"
            if results_file.exists():
                df = self.safe_read_csv(results_file, "results", year)
                if df is not None:
                    self.results_data[year] = df
            
            # Load advanced metrics for each position
            if year not in self.advanced_data:
                self.advanced_data[year] = {}
                
            for position in ['qb', 'rb', 'wr', 'te']:
                advanced_file = self.data_dir / f"advanced_{position}_{year}.csv"
                if advanced_file.exists():
                    df = self.safe_read_csv(advanced_file, f"{position.upper()} advanced", year)
                    if df is not None:
                        self.advanced_data[year][position.upper()] = df
        
        print(f"\n Data Loading Summary:")
        print(f"Draft data loaded for: {list(self.draft_data.keys())}")
        print(f"ADP data loaded for: {list(self.adp_data.keys())}")
        print(f"Results data loaded for: {list(self.results_data.keys())}")
        
        # Show advanced data summary
        for year in self.advanced_data:
            positions = list(self.advanced_data[year].keys())
            if positions:
                print(f"Advanced metrics for {year}: {positions}")
        
        # Load injury predictor data
        self._load_injury_data()
        
        # Initialize and train ML models
        print("\n Training ML models for player profiling...")
        self._train_ml_models()
    
    def _load_injury_data(self) -> None:
        """Load injury predictor data for all positions"""
        print("\n Loading injury predictor data...")
        
        positions = ['qb', 'rb', 'wr', 'te']
        for position in positions:
            injury_file = self.data_dir / f"injury_predictor_{position}.csv"
            if injury_file.exists():
                try:
                    df = pd.read_csv(injury_file)
                    # Clean column names (remove line breaks and extra spaces)
                    df.columns = [col.replace('\n', ' ').strip() for col in df.columns]
                    self.injury_data[position.upper()] = df
                    print(f" Loaded {position.upper()} injury data: {len(df)} players")
                except Exception as e:
                    print(f"  Error loading {position.upper()} injury data: {e}")
            else:
                print(f"  Injury predictor file not found: {injury_file}")
    
    def _prepare_ml_features(self, position: str, year: int) -> Optional[pd.DataFrame]:
        """Prepare feature matrix for ML models"""
        if year not in self.advanced_data or position not in self.advanced_data[year]:
            return None
            
        df = self.advanced_data[year][position].copy()
        
        # Get key metrics for this position
        key_metrics = self.key_metrics.get(position, [])
        
        # Select numeric columns that exist in the data
        numeric_cols = []
        for metric in key_metrics:
            if metric in df.columns:
                numeric_cols.append(metric)
        
        if not numeric_cols:
            return None
            
        # Find the player column name
        player_col = None
        for col in ['PLAYER', 'Player', 'NAME', 'Name']:
            if col in df.columns:
                player_col = col
                break
        
        if player_col is None:
            return None
            
        # Create feature matrix
        features_df = df[[player_col] + numeric_cols].copy()
        features_df.rename(columns={player_col: 'PLAYER'}, inplace=True)
        
        # Handle missing values
        for col in numeric_cols:
            if features_df[col].dtype in ['object', 'string']:
                # Try to convert to numeric
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
            features_df[col].fillna(features_df[col].median(), inplace=True)
        
        return features_df
    
    def _calculate_historical_performance(self, player: str, position: str) -> Dict[str, float]:
        """Calculate historical ADP vs finish performance for a player with position-specific analysis"""
        adp_diffs = []
        fantasy_points = []
        positional_finishes = []  # Track positional finishes for peak season bonus
        
        # Skip if player name is not valid
        if not isinstance(player, str) or pd.isna(player) or len(player.strip()) == 0:
            return {
                'avg_adp_differential': 0,
                'adp_differential_std': 0,
                'avg_fantasy_points': 0,
                'fantasy_points_std': 0,
                'years_of_data': 0,
                'positional_finishes': [],
                'has_peak_season': False
            }
        
        # Clean player name for better matching
        clean_player = self.clean_player_name(player)
        
        for year in self.years:
            if year not in self.adp_data or year not in self.results_data:
                continue
                
            adp_df = self.adp_data[year]
            results_df = self.results_data[year]
            
            if adp_df.empty or results_df.empty:
                continue
            
            # Filter for position-specific data
            # Try to find position column in ADP data
            adp_position_col = None
            for col in ['Position', 'Pos', 'POSITION', 'POS', 'position']:
                if col in adp_df.columns:
                    adp_position_col = col
                    break
            
            # Try to find position column in results data
            results_position_col = None
            for col in ['Position', 'Pos', 'POSITION', 'POS', 'position']:
                if col in results_df.columns:
                    results_position_col = col
                    break
            
            # Filter by position if position columns exist
            if adp_position_col:
                adp_df = adp_df[adp_df[adp_position_col].str.upper() == position.upper()]
            if results_position_col:
                results_df = results_df[results_df[results_position_col].str.upper() == position.upper()]
            
            if adp_df.empty or results_df.empty:
                continue
            
            # Try multiple column name variations for player names
            adp_player_col = None
            for col in ['Player', 'PLAYER', 'Name', 'NAME', 'player_name']:
                if col in adp_df.columns:
                    adp_player_col = col
                    break
            
            results_player_col = None
            for col in ['Player', 'PLAYER', 'Name', 'NAME', 'player_name']:
                if col in results_df.columns:
                    results_player_col = col
                    break
            
            if not adp_player_col or not results_player_col:
                continue
            
            # Enhanced player matching - try exact match first, then contains
            adp_match = None
            results_match = None
            
            # Try exact match first (case insensitive)
            exact_adp = adp_df[adp_df[adp_player_col].str.lower() == clean_player.lower()]
            exact_results = results_df[results_df[results_player_col].str.lower() == clean_player.lower()]
            
            if not exact_adp.empty:
                adp_match = exact_adp
            else:
                # Try contains match
                adp_match = adp_df[adp_df[adp_player_col].str.contains(clean_player, case=False, na=False)]
            
            if not exact_results.empty:
                results_match = exact_results
            else:
                # Try contains match
                results_match = results_df[results_df[results_player_col].str.contains(clean_player, case=False, na=False)]
            
            if adp_match is not None and results_match is not None and not adp_match.empty and not results_match.empty:
                # Calculate POSITIONAL ADP rank within the filtered position data
                # Sort ADP data by ADP value to get positional ranking
                adp_sorted = adp_df.copy()
                for adp_col in ['ADP', 'AVG', 'adp', 'avg', 'Average']:
                    if adp_col in adp_sorted.columns:
                        adp_sorted = adp_sorted.dropna(subset=[adp_col])
                        adp_sorted = adp_sorted[adp_sorted[adp_col] > 0]
                        adp_sorted = adp_sorted.sort_values(adp_col)
                        break
                
                # Calculate POSITIONAL finish rank within the filtered position data  
                results_sorted = results_df.copy()
                for rank_col in ['RK', '#', 'Rank', 'rank', 'Position', 'Finish']:
                    if rank_col in results_sorted.columns:
                        results_sorted = results_sorted.dropna(subset=[rank_col])
                        results_sorted = results_sorted[results_sorted[rank_col] > 0]
                        results_sorted = results_sorted.sort_values(rank_col)
                        break
                
                # Find positional ADP rank
                positional_adp_rank = None
                if not adp_sorted.empty:
                    player_idx = adp_sorted[adp_sorted[adp_player_col].str.lower() == clean_player.lower()].index
                    if len(player_idx) > 0:
                        positional_adp_rank = adp_sorted.index.get_loc(player_idx[0]) + 1
                
                # Find positional finish rank  
                positional_finish_rank = None
                if not results_sorted.empty:
                    player_idx = results_sorted[results_sorted[results_player_col].str.lower() == clean_player.lower()].index
                    if len(player_idx) > 0:
                        positional_finish_rank = results_sorted.index.get_loc(player_idx[0]) + 1
                
                # Try multiple fantasy points column variations
                fp = None
                for fp_col in ['FPTS', 'TTL', 'PTS', 'Points', 'Fantasy Points', 'FantasyPoints']:
                    if fp_col in results_match.columns:
                        fp_val = results_match.iloc[0].get(fp_col)
                        if not pd.isna(fp_val) and fp_val > 0:
                            fp = float(fp_val)
                            break
                
                # Add to arrays if valid data found
                if positional_adp_rank is not None and positional_finish_rank is not None:
                    differential = positional_finish_rank - positional_adp_rank
                    adp_diffs.append(differential)
                    positional_finishes.append(positional_finish_rank)
                    # Debug output for troubleshooting
                    if abs(differential) > 0.1:  # Only log non-zero differentials
                        print(f"DEBUG: {player} ({year}): Pos ADP={positional_adp_rank}, Pos Finish={positional_finish_rank}, Diff={differential:.1f}")
                
                if fp is not None:
                    fantasy_points.append(fp)
        
        # Check for peak seasons (top 5 positional finish)
        has_peak_season = any(finish <= 5 for finish in positional_finishes)
        
        # Enhanced return with fallback indicators
        has_historical_data = len(adp_diffs) > 0
        
        return {
            'avg_adp_differential': np.mean(adp_diffs) if adp_diffs else 0,
            'adp_differential_std': np.std(adp_diffs) if len(adp_diffs) > 1 else 0,
            'avg_fantasy_points': np.mean(fantasy_points) if fantasy_points else 0,
            'fantasy_points_std': np.std(fantasy_points) if len(fantasy_points) > 1 else 0,
            'years_of_data': len(adp_diffs),
            'positional_finishes': positional_finishes,
            'has_peak_season': has_peak_season,
            'has_historical_data': has_historical_data,
            'data_quality': 'full' if len(adp_diffs) >= 2 else 'limited' if len(adp_diffs) == 1 else 'none'
        }
    

    
    def _calculate_z_score_based_score(self, value: float, position_values: pd.Series, metric_name: str = "") -> tuple[float, float]:
        """
        Calculate z-score and convert to 2-10 scale
        
        Returns:
            tuple: (z_score, scaled_score_2_10)
        """
        if len(position_values) < 3:
            return 0.0, 6.0  # Default neutral if insufficient data
            
        mean_val = position_values.mean()
        std_val = position_values.std()
        
        if std_val == 0:
            return 0.0, 6.0  # All values same, return neutral
            
        z_score = (value - mean_val) / std_val
        
        # ENHANCED Z-SCORE MAPPING: Steeper curves for elite/poor, flatter for middle
        # This creates wider distribution by pushing mid-range scores down and elite/poor scores to extremes
        
        if z_score >= 1.2:  # Elite (top ~12%)
            # Steep curve: z=1.2 → 9.0, z=2.0+ → 10.0
            scaled_score = 9.0 + min(1.0, (z_score - 1.2) / 0.8 * 1.0)  # 9.0-10.0
        elif z_score >= 0.5:  # Very good (12-31%)
            # Moderate curve: z=0.5 → 7.5, z=1.2 → 9.0  
            scaled_score = 7.5 + (z_score - 0.5) / 0.7 * 1.5  # 7.5-9.0
        elif z_score >= -0.5:  # Average range (31-69%) - FLATTENED
            # Flat curve: z=-0.5 → 4.5, z=0.5 → 7.5
            scaled_score = 4.5 + (z_score + 0.5) / 1.0 * 3.0  # 4.5-7.5
        elif z_score >= -1.2:  # Below average (69-88%)
            # Moderate curve: z=-1.2 → 3.0, z=-0.5 → 4.5
            scaled_score = 3.0 + (z_score + 1.2) / 0.7 * 1.5  # 3.0-4.5
        else:  # Poor (bottom 12%)
            # Steep curve: z<-1.2 → 1.0-3.0, z<-2.0 → 1.0
            scaled_score = max(1.0, 3.0 + (z_score + 1.2) / 0.8 * 2.0)  # 1.0-3.0
            
        return z_score, scaled_score
    
    def _calculate_non_linear_aggregated_score(self, hist_score: float, adv_score: float, 
                                             ml_score: float, injury_score: float,
                                             historical_weight: float, advanced_weight: float,
                                             ml_weight: float, injury_weight: float) -> tuple[float, dict]:
        """
        Calculate final score using non-linear aggregation to emphasize elite performance
        
        Uses exponentially weighted mean where elite scores (>8.0) get boosted
        and poor scores (<4.0) get penalized more heavily
        """
        
        # Component scores with their weights
        components = [
            ('Historical', hist_score, historical_weight),
            ('Advanced', adv_score, advanced_weight), 
            ('ML', ml_score, ml_weight),
            ('Injury', injury_score, injury_weight)
        ]
        
        # Standard weighted average
        standard_score = sum(score * weight for _, score, weight in components)
        
        # Non-linear aggregation: Elite performance bonus
        elite_bonus = 0.0
        elite_components = []
        
        # AGGRESSIVE non-linear scaling for elite and poor performance
        boosted_components = []
        for name, score, weight in components:
            if score >= 8.5:  # Elite performance (lowered threshold)
                # Higher boost: up to 50% for 10.0 scores
                boost_factor = 1.0 + (score - 8.5) * 0.5  # Up to 50% boost
                boosted_score = score * boost_factor
                elite_bonus += (boosted_score - score) * weight
                elite_components.append(name)
                boosted_components.append((name, score, boosted_score, weight))
            elif score >= 7.5:  # Very good performance gets moderate boost
                boost_factor = 1.0 + (score - 7.5) * 0.2  # Up to 20% boost
                boosted_score = score * boost_factor
                elite_bonus += (boosted_score - score) * weight
                boosted_components.append((name, score, boosted_score, weight))
            elif score <= 4.5:  # Poor performance penalty (raised threshold)
                # More severe penalty: 30-60% penalty for 2.0-4.5 scores
                penalty_factor = 0.4 + (score - 2.0) / 2.5 * 0.6  # 40-100% of original score
                penalized_score = score * penalty_factor  
                penalty = (score - penalized_score) * weight
                elite_bonus -= penalty  # Subtract penalty
                boosted_components.append((name, score, penalized_score, weight))
            else:
                boosted_components.append((name, score, score, weight))
        
        # Calculate final non-linear score
        final_score = standard_score + elite_bonus
        
        # CRITICAL: Apply non-linear transformation to stretch tails
        # This pulls elite scores higher and poor scores lower
        if final_score > 8.5:
            final_score += 0.3  # Stretch elite players further toward 10.0
        elif final_score < 4.5:
            final_score -= 0.3  # Pull poor scores down toward 2.0
        
        # Apply soft-max scaling function for additional tail stretching
        # final_score = 10 * (final_score / 10) ** 1.2  # More aggressive curve
        
        # Ensure score stays in 2-10 range after transformations
        final_score = max(2.0, min(10.0, final_score))
        
        aggregation_details = {
            'standard_score': standard_score,
            'elite_bonus': elite_bonus,
            'elite_components': elite_components,
            'boosted_components': boosted_components,
            'final_score': final_score
        }
        
        return final_score, aggregation_details
    
    def get_actual_fantasy_points(self, player_name: str, position: str, year: int = 2024) -> Optional[float]:
        """Get ACTUAL fantasy points from results file - the REAL data"""
        try:
            if year not in self.results_data:
                self._log_missing_data(player_name, position, f"results_data_year_{year}", [f"results_{year}.csv"])
                return None
            
            results_df = self.results_data[year]
            
            # Clean player name for better matching (remove team abbreviations)
            clean_name = self.clean_player_name_for_matching(player_name)
            
            # Try multiple search approaches for better matching
            player_row = None
            
            # Try exact match first
            exact_match = results_df[results_df['Player'].str.contains(clean_name, na=False, case=False)]
            if not exact_match.empty:
                player_row = exact_match
            else:
                # Try partial match with original name
                partial_match = results_df[results_df['Player'].str.contains(player_name.split('(')[0].strip(), na=False, case=False)]
                if not partial_match.empty:
                    player_row = partial_match
            
            if player_row is None or player_row.empty:
                self._log_missing_data(player_name, position, f"player_in_results_{year}", 
                                     [f"tried: {clean_name}, {player_name.split('(')[0].strip()}"])
                return None
            
            stats_dict = player_row.iloc[0].to_dict()
            
            # Get actual fantasy points - these are the REAL points scored
            actual_points = self.get_real_stat_value(
                stats_dict, ['TTL', 'Total', 'FPTS', 'Points', 'Fantasy_Points'], 
                player_name, position, 'actual_fantasy_points_total'
            )
            
            return actual_points
            
        except Exception as e:
            self._log_missing_data(player_name, position, "actual_fantasy_points_error", [str(e)])
            return None
    
    
    def calculate_overall_profile_score(self, historical_perf: Dict[str, float], advanced_metrics: Dict[str, Any], 
                                      ml_predictions: Dict[str, float], injury_profile: Dict[str, Any], debug_mode: bool = False,
                                      position: str = None, player_row: Any = None, player_name: str = "unknown") -> Dict[str, float]:
        """
        Calculate Overall Profile Score (0-100) with position-specific weighted components - ONLY REAL DATA:
        
        Position-Specific Weights:
        - QB: Historical(10%) + Advanced(35%) + ML(40%) + Injury(15%) = 100%
        - RB: Historical(10%) + Advanced(30%) + ML(35%) + Injury(25%) = 100%  
        - WR: Historical(10%) + Advanced(35%) + ML(35%) + Injury(20%) = 100%
        - TE: Historical(15%) + Advanced(30%) + ML(35%) + Injury(20%) = 100%
        
        Emphasizes recent performance and predictive value over historical output.
        """
        
        # Position-specific weight table
        position_weights = {
            'QB': {'historical': 0.10, 'advanced': 0.35, 'ml': 0.40, 'injury': 0.15},
            'RB': {'historical': 0.10, 'advanced': 0.30, 'ml': 0.35, 'injury': 0.25},
            'WR': {'historical': 0.10, 'advanced': 0.35, 'ml': 0.35, 'injury': 0.20},
            'TE': {'historical': 0.15, 'advanced': 0.30, 'ml': 0.35, 'injury': 0.20}
        }
        
        # Use position-specific weights or default to QB weights if position unknown
        weights = position_weights.get(position, position_weights['QB'])
        
        # Component scores (0-10 scale) - with player context for debug logging
        historical_score = self._calculate_historical_score(historical_perf)
        advanced_score = self._calculate_advanced_metrics_score(advanced_metrics, position, player_row)
        ml_score = self._calculate_ml_predictions_score(ml_predictions, player_name, position)
        injury_score = self._calculate_injury_profile_score(injury_profile, player_name, position)
        
        if debug_mode:
            print(f"DEBUG SCORING BREAKDOWN:")
            print(f"  Historical: {historical_perf} -> {historical_score:.2f}/10")
            print(f"  Advanced: {advanced_metrics} -> {advanced_score:.2f}/10")
            print(f"  ML: {ml_predictions} -> {ml_score:.2f}/10")
            print(f"  Injury: {injury_profile} -> {injury_score:.2f}/10")
            print(f"  Position Weights ({position}): H:{weights['historical']:.0%} A:{weights['advanced']:.0%} ML:{weights['ml']:.0%} I:{weights['injury']:.0%}")
        
        # Position-specific weighted combination (convert to 0-100 scale)
        overall_score = (
            historical_score * weights['historical'] +
            advanced_score * weights['advanced'] +
            ml_score * weights['ml'] +
            injury_score * weights['injury']
        ) * 10  # Convert from 0-10 to 0-100 scale
        
        if debug_mode:
            print(f"  Weighted calc: ({historical_score:.2f}*{weights['historical']:.2f} + {advanced_score:.2f}*{weights['advanced']:.2f} + {ml_score:.2f}*{weights['ml']:.2f} + {injury_score:.2f}*{weights['injury']:.2f}) * 10 = {overall_score:.1f}")
        
        # Calculate star rating (1-5 stars)
        star_rating = min(5, max(1, int((overall_score / 20) + 1)))
        
        return {
            'overall_score': round(overall_score, 1),
            'historical_score': round(historical_score, 1),
            'advanced_score': round(advanced_score, 1),
            'ml_score': round(ml_score, 1),
            'injury_score': round(injury_score, 1),
            'star_rating': star_rating
        }
    
    def _calculate_historical_score(self, historical_perf: Dict[str, float]) -> float:
        """Calculate historical performance component score (0-10) - ONLY REAL DATA"""
        
        # REQUIRE real historical data - no fallbacks
        if not historical_perf or historical_perf.get('years_of_data', 0) == 0:
            self._log_missing_data("unknown", "unknown", "no_historical_data", 
                                 [f"historical_perf: {historical_perf}"])
            return 0.0  # Return 0 if no historical data, no fallback
        
        # Position-specific ADP differential scoring (negative is better - outperforming ADP)
        # Now uses positional rankings instead of overall rankings
        adp_diff = historical_perf['avg_adp_differential']
        
        # Base ADP differential scoring with same thresholds
        if adp_diff <= -15:  # 15+ spots better than positional ADP (elite)
            adp_score = 10.0
        elif adp_diff <= -10:  # 10-15 spots better (excellent)
            adp_score = 8.5
        elif adp_diff <= -5:   # 5-10 spots better (very good)
            adp_score = 7.5
        elif adp_diff <= -3:   # 3-5 spots better (good)
            adp_score = 6.5
        elif adp_diff <= -1:   # 1-3 spots better (above average)
            adp_score = 5.5
        elif adp_diff <= 1:    # Within 1 spot of positional ADP (average)
            adp_score = 4.5
        elif adp_diff <= 3:    # 1-3 spots worse (below average)
            adp_score = 3.5
        elif adp_diff <= 5:    # 3-5 spots worse (poor)
            adp_score = 2.5
        elif adp_diff <= 10:   # 5-10 spots worse (very poor)
            adp_score = 1.5
        else:  # 10+ spots worse than positional ADP (extremely poor)
            adp_score = 1.0
        
        # High-ADP penalty reduction for slight underperformance
        # If any season had top-24 positional ADP and finished within reasonable range
        positional_finishes = historical_perf.get('positional_finishes', [])
        if positional_finishes and adp_diff > 0 and adp_diff <= 5:
            # Check if player was ever drafted highly (assume top 24 at position)
            # For slight underperformance (1-5 spots), reduce penalty
            penalty_reduction = max(0, min(1.0, (5 - adp_diff) / 5))  # Linear reduction
            adp_score = min(adp_score + penalty_reduction, 6.0)  # Cap improvement
        
        # Peak season bonus for top-5 positional finishes
        peak_bonus = 0
        if historical_perf.get('has_peak_season', False):
            peak_bonus = 0.5  # Increased from 0.25 for more impact
        
        # Consistency bonus (lower std dev is better)
        consistency_bonus = 0
        if historical_perf['adp_differential_std'] < 3:  # Adjusted for positional scale
            consistency_bonus = 1.0
        elif historical_perf['adp_differential_std'] < 5:
            consistency_bonus = 0.5
        
        # Experience bonus (more data = more reliable)
        experience_bonus = min(1.0, historical_perf['years_of_data'] / 3)
        
        return min(10.0, adp_score + peak_bonus + consistency_bonus + experience_bonus)
    
    def _calculate_advanced_metrics_score(self, advanced_metrics: Dict[str, Any], position: str = None, player_row: Any = None) -> float:
        """Calculate advanced metrics component score using universal system ONLY - NO FALLBACKS"""
        
        # REQUIRE player_row and position for universal system
        if player_row is None or position is None:
            self._log_missing_data("unknown", position or "unknown", "player_row_or_position_missing", 
                                 [f"player_row: {player_row is not None}", f"position: {position}"])
            return 0.0  # Return 0 if missing required data, no fallback
        
        try:
            universal_score, breakdown = self.calculate_advanced_score_universal(player_row, position)
            return universal_score
        except Exception as e:
            self._log_missing_data("unknown", position, "universal_scoring_error", [str(e)])
            return 0.0  # Return 0 on error, no fallback
    
    def _calculate_ml_predictions_score(self, ml_predictions: Dict[str, float], 
                                       player_name: str = "unknown", position: str = "unknown") -> float:
        """Calculate ML predictions component score (0-10) with position-based weighting - ONLY REAL DATA"""
        try:
            # Get real breakout probability - NO FALLBACK
            breakout_prob = self.get_real_dict_value(
                ml_predictions, ['breakout_probability'], 
                player_name, position, 'ml_breakout_probability'
            )
            
            # Get real consistency score - NO FALLBACK
            consistency_score = self.get_real_dict_value(
                ml_predictions, ['consistency_score'], 
                player_name, position, 'ml_consistency_score'
            )
            
            # Get real positional value - NO FALLBACK
            positional_value = self.get_real_dict_value(
                ml_predictions, ['positional_value'], 
                player_name, position, 'ml_positional_value'
            )
            
            # Get real model confidence - NO FALLBACK
            confidence = self.get_real_dict_value(
                ml_predictions, ['model_confidence'], 
                player_name, position, 'ml_model_confidence'
            )
            
            # Only calculate score if we have ALL real data
            if all(x is not None for x in [breakout_prob, consistency_score, positional_value, confidence]):
                
                # 1. Position-Based Weighting
                position_weights = {
                    'QB': {'breakout': 0.30, 'consistency': 0.50, 'positional': 0.20},
                    'RB': {'breakout': 0.40, 'consistency': 0.30, 'positional': 0.30},
                    'WR': {'breakout': 0.40, 'consistency': 0.30, 'positional': 0.30},
                    'TE': {'breakout': 0.30, 'consistency': 0.20, 'positional': 0.50}
                }
                
                # Use position-specific weights or equal weighting if position unknown
                weights = position_weights.get(position, {'breakout': 0.33, 'consistency': 0.33, 'positional': 0.34})
                
                # 2. Convert breakout probability to 0-10 scale
                breakout_score = breakout_prob * 10
                
                # 3. Normalize Positional Value with diminishing returns
                # Apply transformation that reduces weight of top-end scores, boosts middle-range
                # Using a square root transformation to create diminishing returns
                import math
                normalized_positional = math.sqrt(positional_value) * 10
                
                # 4. Calculate weighted base score using position-specific weights
                base_score = (
                    breakout_score * weights['breakout'] +
                    consistency_score * weights['consistency'] +
                    normalized_positional * weights['positional']
                )
                
                # 5. Blended Confidence Adjustment (80% base score + 20% confidence-adjusted)
                if confidence > 0:
                    confidence_adjusted_score = base_score * confidence
                    final_score = (base_score * 0.80) + (confidence_adjusted_score * 0.20)
                else:
                    final_score = base_score
                
                # 6. Clamp to 0-10 range
                return max(0.0, min(10.0, final_score))
            else:
                # Log that we're missing ML prediction data
                self._log_missing_data(player_name, position, "complete_ml_predictions", 
                                     ['breakout_probability', 'consistency_score', 'positional_value', 'model_confidence'])
                return 0.0  # Return 0 if missing data, no fallback
            
        except Exception as e:
            self._log_missing_data(player_name, position, "ml_predictions_calculation_error", [str(e)])
            return 0.0  # Return 0 on error, no fallback
    
    def _calculate_injury_profile_score(self, injury_profile: Dict[str, Any], 
                                       player_name: str = "unknown", position: str = "unknown") -> float:
        """Calculate injury profile component score (0-10) - ONLY REAL DATA"""
        try:
            # Get real durability score - NO FALLBACK
            durability = self.get_real_dict_value(
                injury_profile, ['durability_score'], 
                player_name, position, 'injury_durability_score'
            )
            
            # Get real projected games missed - NO FALLBACK
            games_missed = self.get_real_dict_value(
                injury_profile, ['projected_games_missed'], 
                player_name, position, 'injury_games_missed'
            )
            
            # Get real injury risk - NO FALLBACK
            injury_risk = self.get_real_dict_value(
                injury_profile, ['injury_risk'], 
                player_name, position, 'injury_risk'
            )
            
            # Only calculate score if we have ALL real data
            if all(x is not None for x in [durability, games_missed, injury_risk]):
                # Calculate availability score from games missed (higher games missed = lower score)
                if games_missed >= 0:
                    availability_score = max(0, 10 - (games_missed * 0.5))  # Real calculation based on actual data
                else:
                    availability_score = 0  # Invalid data
                
                # Calculate risk score from injury risk (lower risk = higher score)
                if 0 <= injury_risk <= 1:
                    risk_score = (1 - injury_risk) * 10
                else:
                    risk_score = 0  # Invalid data
                
                # Average of all injury factors
                return (durability + availability_score + risk_score) / 3
            else:
                # Log that we're missing injury data
                self._log_missing_data(player_name, position, "complete_injury_profile", 
                                     ['durability_score', 'projected_games_missed', 'injury_risk'])
                return 0.0  # Return 0 if missing data, no fallback
            
        except Exception as e:
            self._log_missing_data(player_name, position, "injury_profile_calculation_error", [str(e)])
            return 0.0  # Return 0 on error, no fallback
    
    def _train_ml_models(self) -> None:
        """Train ML models for player profiling"""
        positions = ['QB', 'RB', 'WR', 'TE']
        
        for position in positions:
            print(f"Training models for {position}...")
            print(f"  Available years: {self.years}")
            print(f"  Training years: {self.years[:-1]}")
            print(f"  Results data years: {list(self.results_data.keys())}")
            print(f"  Advanced data years: {list(self.advanced_data.keys())}")
            print(f"  ADP data years: {list(self.adp_data.keys())}")
            
            # Collect training data across all years
            all_features = []
            all_targets = {
                'breakout': [],
                'consistency': [],
                'positional_value': [],
                'fantasy_points': []
            }
            
            for year in self.years[:-1]:  # Exclude most recent year for prediction
                features_df = self._prepare_ml_features(position, year)
                if features_df is None:
                    continue
                
                # Get corresponding results for targets
                if year not in self.results_data:
                    continue
                
                results_df = self.results_data[year]
                adp_df = self.adp_data.get(year)
                
                print(f"    Processing {year}: {len(features_df)} players in advanced data")
                for _, row in features_df.iterrows():
                    player = row['PLAYER']
                    
                    # Skip if player name is not valid
                    if not isinstance(player, str) or pd.isna(player) or len(player.strip()) == 0:
                        continue
                    
                    # Clean player name for better matching
                    clean_player = self.clean_player_name_for_matching(player)
                    if not clean_player:
                        continue
                    
                    # Find player in results using enhanced matching
                    results_player_col = 'Player' if 'Player' in results_df.columns else 'PLAYER'
                    
                    # First try exact match with cleaned names
                    results_df_cleaned = results_df.copy()
                    results_df_cleaned['Player_Clean'] = results_df_cleaned[results_player_col].apply(self.clean_player_name_for_matching)
                    player_results = results_df_cleaned[results_df_cleaned['Player_Clean'] == clean_player]
                    
                    # If no exact match, try fuzzy matching
                    if player_results.empty:
                        player_results = results_df[results_df[results_player_col].str.contains(player.split()[0], case=False, na=False)]
                        if player_results.empty:
                            continue
                    
                    player_result = player_results.iloc[0]
                    fantasy_points = self.safe_float_conversion(player_result.get('FPTS', player_result.get('TTL', player_result.get('PTS', 0))))
                    finish_rank = self.safe_float_conversion(player_result.get('RK', player_result.get('#', 999)))
                    
                    # Get ADP data using enhanced matching
                    adp_rank = 999
                    if adp_df is not None:
                        adp_player_col = 'Player' if 'Player' in adp_df.columns else 'PLAYER'
                        
                        # First try exact match with cleaned names
                        adp_df_cleaned = adp_df.copy()
                        adp_df_cleaned['Player_Clean'] = adp_df_cleaned[adp_player_col].apply(self.clean_player_name_for_matching)
                        player_adp = adp_df_cleaned[adp_df_cleaned['Player_Clean'] == clean_player]
                        
                        # If no exact match, try fuzzy matching
                        if player_adp.empty:
                            player_adp = adp_df[adp_df[adp_player_col].str.contains(player.split()[0], case=False, na=False)]
                        
                        if not player_adp.empty:
                            adp_rank = self.safe_float_conversion(player_adp.iloc[0].get('ADP', player_adp.iloc[0].get('AVG', 999)))
                    
                    # Calculate targets with improved variance and continuous scoring
                    
                    # 1. Continuous breakout probability (0-1) based on ADP outperformance
                    adp_diff = adp_rank - finish_rank
                    if adp_diff >= 25:  # Major outperformance
                        breakout = 0.9 + min(0.1, (adp_diff - 25) / 50)  # 0.9-1.0
                    elif adp_diff >= 10:  # Good outperformance  
                        breakout = 0.6 + (adp_diff - 10) / 15 * 0.3  # 0.6-0.9
                    elif adp_diff >= 0:   # Mild outperformance
                        breakout = 0.4 + adp_diff / 10 * 0.2  # 0.4-0.6
                    elif adp_diff >= -15:  # Mild underperformance
                        breakout = 0.2 + (adp_diff + 15) / 15 * 0.2  # 0.2-0.4
                    else:  # Significant underperformance
                        breakout = max(0.0, 0.2 + (adp_diff + 15) / 25 * 0.2)  # 0.0-0.2
                    
                    # 2. Consistency based on weekly fantasy points standard deviation
                    # Collect weekly scores for this player if available
                    weekly_scores = []
                    for week_col in [c for c in row.index if 'week' in c.lower() and 'fpts' in c.lower()]:
                        week_score = self.safe_float_conversion(row[week_col])
                        if week_score > 0:  # Valid weekly score
                            weekly_scores.append(week_score)
                    
                    if len(weekly_scores) >= 8:  # Need sufficient games for consistency
                        weekly_std = np.std(weekly_scores)
                        weekly_mean = np.mean(weekly_scores)
                        # Coefficient of variation (lower = more consistent)
                        cv = weekly_std / weekly_mean if weekly_mean > 0 else 1.0
                        # Convert to 0-10 scale (lower CV = higher consistency)
                        consistency = max(0, min(10, 10 - (cv * 15)))  # CV of 0.67 = score of 0
                    else:
                        # Fallback: use season variance relative to expected
                        season_variance = abs(fantasy_points - adp_rank * 2) / max(1, adp_rank * 0.1)
                        consistency = max(0, min(10, 10 - season_variance))
                    
                    # 3. Positional value with wider range based on percentile within position
                    # Use finish rank percentile with more realistic distribution
                    if finish_rank <= 12:  # Top tier (85th-100th percentile)
                        positional_value = 0.85 + (12 - finish_rank) / 12 * 0.15  # 0.85-1.0
                    elif finish_rank <= 24:  # Second tier (60th-85th percentile)
                        positional_value = 0.60 + (24 - finish_rank) / 12 * 0.25  # 0.60-0.85  
                    elif finish_rank <= 36:  # Third tier (35th-60th percentile)
                        positional_value = 0.35 + (36 - finish_rank) / 12 * 0.25  # 0.35-0.60
                    elif finish_rank <= 48:  # Fourth tier (15th-35th percentile)
                        positional_value = 0.15 + (48 - finish_rank) / 12 * 0.20  # 0.15-0.35
                    else:  # Bottom tier (0-15th percentile)
                        positional_value = max(0.0, 0.15 - (finish_rank - 48) / 50 * 0.15)  # 0.0-0.15
                    
                    # Store features and targets
                    feature_vector = row.drop('PLAYER').values
                    
                    # Convert all values to float and handle non-numeric data
                    try:
                        numeric_features = []
                        for val in feature_vector:
                            numeric_val = self.safe_float_conversion(val)
                            numeric_features.append(numeric_val)
                        
                        numeric_features = np.array(numeric_features, dtype=float)
                        
                        # Check for NaN values
                        if not np.any(np.isnan(numeric_features)):
                            all_features.append(numeric_features)
                            all_targets['breakout'].append(breakout)
                            all_targets['consistency'].append(consistency)
                            all_targets['positional_value'].append(positional_value)
                            all_targets['fantasy_points'].append(fantasy_points)
                            if len(all_features) <= 5:  # Debug first few
                                print(f"      Added sample: {clean_player} -> breakout={breakout}, consistency={consistency:.2f}, pos_value={positional_value:.2f}")
                    except Exception as e:
                        # Skip this sample if feature conversion fails
                        if len(all_features) <= 5:  # Only show first few errors
                            print(f"      Error processing {clean_player}: {e}")
                        continue
            
            if len(all_features) < 10:  # Need minimum data for training
                print(f"  Insufficient data for {position} models: {len(all_features)} samples (need 10+)")
                print(f"  Years processed: {[year for year in self.years[:-1] if year in self.results_data]}")
                continue
            
            X = np.array(all_features)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers[position] = scaler
            
            # Train models
            models = {}
            confidence = {}
            
            # Breakout regressor (changed from classifier due to continuous targets)
            y_breakout = np.array(all_targets['breakout'])
            if len(np.unique(y_breakout)) > 1:
                rf_breakout = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_breakout.fit(X_scaled, y_breakout)
                models['breakout'] = rf_breakout
                
                # Calculate confidence via cross-validation
                cv_scores = cross_val_score(rf_breakout, X_scaled, y_breakout, cv=5)
                confidence['breakout'] = cv_scores.mean()
            
            # Consistency regressor
            y_consistency = np.array(all_targets['consistency'])
            rf_consistency = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_consistency.fit(X_scaled, y_consistency)
            models['consistency'] = rf_consistency
            
            cv_scores = cross_val_score(rf_consistency, X_scaled, y_consistency, cv=5)
            confidence['consistency'] = max(0, cv_scores.mean())
            
            # Positional value regressor
            y_positional = np.array(all_targets['positional_value'])
            rf_positional = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_positional.fit(X_scaled, y_positional)
            models['positional_value'] = rf_positional
            
            cv_scores = cross_val_score(rf_positional, X_scaled, y_positional, cv=5)
            confidence['positional_value'] = max(0, cv_scores.mean())
            
            self.ml_models[position] = models
            self.model_confidence[position] = confidence
            
            # Log target variance statistics to verify improvements
            breakout_std = np.std(all_targets['breakout'])
            consistency_std = np.std(all_targets['consistency'])
            positional_std = np.std(all_targets['positional_value'])
            
            print(f" {position} models trained with {len(X)} samples")
            print(f"  Confidence scores: {confidence}")
            print(f"  Target variance - Breakout: {breakout_std:.3f}, Consistency: {consistency_std:.3f}, Positional: {positional_std:.3f}")
            print(f"  Target ranges - Breakout: [{np.min(all_targets['breakout']):.2f}, {np.max(all_targets['breakout']):.2f}]")
            print(f"                  Consistency: [{np.min(all_targets['consistency']):.2f}, {np.max(all_targets['consistency']):.2f}]") 
            print(f"                  Positional: [{np.min(all_targets['positional_value']):.2f}, {np.max(all_targets['positional_value']):.2f}]")
    
    def clean_player_name(self, name: Any) -> str:
        """Clean player names for better matching"""
        if not isinstance(name, str):
            return str(name)
        
        # Remove team abbreviations in parentheses (like "Ja'Marr Chase (CIN)")
        name = name.split('(')[0].strip()
        
        # Remove quotes that might be in CSV data
        name = name.strip('"')
        
        return name.strip()
    
    def clean_player_name_for_matching(self, name: Any) -> str:
        """Enhanced player name cleaning for cross-dataset matching"""
        if not isinstance(name, str) or pd.isna(name):
            return ""
        
        import re
        
        # Start with basic cleaning
        cleaned = self.clean_player_name(name)
        
        # Remove suffixes (Jr., Sr., II, III, IV, etc.)
        cleaned = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)(\s|$)', '', cleaned, flags=re.IGNORECASE)
        
        # Handle apostrophes consistently (replace curly quotes with straight quotes)
        cleaned = cleaned.replace("'", "'").replace("'", "'")
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Convert to title case for consistent matching
        cleaned = cleaned.title()
        
        return cleaned
    
    # Universal Metrics System Integration
    def discover_position_metrics(self) -> Dict[str, Dict]:
        """Auto-discover all available metrics for each position"""
        position_metrics = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            if 2024 in self.advanced_data and position.upper() in self.advanced_data[2024]:
                df = self.advanced_data[2024][position.upper()]
                
                # Get metric columns (exclude non-metric columns)
                metric_columns = [col for col in df.columns if col not in ['Rank', 'Player', 'G']]
                
                position_metrics[position] = {
                    'metric_columns': metric_columns,
                    'player_count': len(df)
                }
        
        return position_metrics
    
    def categorize_metrics_by_type(self, position_metrics: Dict[str, Dict]) -> Dict[str, List]:
        """Group metrics by function (volume, efficiency, etc.)"""
        import re
        
        metric_categories = {
            'volume': [],
            'efficiency': [],
            'explosiveness': [],
            'opportunity': [],
            'negative': [],
            'other': []
        }
        
        # Pattern matching for categorization
        volume_patterns = ['ATT', 'TGT', 'REC', 'COMP', 'YDS', 'YACON', 'YBC', 'AIR', 'YAC']
        efficiency_patterns = ['Y/A', 'Y/R', 'PCT', 'RTG', '/ATT', '/R', 'YAC/R', 'YACON/R', 'YBC/R', 'AIR/R', 'YBCON/ATT', 'YACON/ATT']
        explosive_patterns = ['10+', '20+', '30+', '40+', '50+', 'LNG', 'BRKTKL']
        opportunity_patterns = ['RZ', '% TM', 'CATCHABLE', 'PKT TIME', 'TM']
        negative_patterns = ['SACK', 'DROP', 'TK LOSS', 'POOR', 'KNCK', 'HRRY', 'BLITZ']
        
        for position, data in position_metrics.items():
            for column in data['metric_columns']:
                category = 'other'
                column_upper = column.upper()
                
                # Check efficiency patterns first (more specific)
                if any(pattern in column_upper for pattern in efficiency_patterns):
                    category = 'efficiency'
                elif any(pattern in column_upper for pattern in explosive_patterns):
                    category = 'explosiveness'
                elif any(pattern in column_upper for pattern in opportunity_patterns):
                    category = 'opportunity'
                elif any(pattern in column_upper for pattern in negative_patterns):
                    category = 'negative'
                elif any(pattern in column_upper for pattern in volume_patterns):
                    category = 'volume'
                
                metric_categories[category].append({
                    'position': position,
                    'metric': column
                })
        
        return metric_categories
    
    def clean_numeric_value_universal(self, value) -> float:
        """Convert any value to clean numeric for universal system"""
        if pd.isna(value):
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        str_val = str(value).strip()
        
        if not str_val:
            return 0.0
        
        # Handle percentages
        if '%' in str_val:
            try:
                return float(str_val.replace('%', '')) / 100
            except:
                return 0.0
        
        # Handle comma-separated numbers
        if ',' in str_val:
            str_val = str_val.replace(',', '')
        
        try:
            return float(str_val)
        except:
            return 0.0
    
    def normalize_metric_universal(self, raw_value: float, metric_name: str, category: str) -> float:
        """Normalize metric value to 0-10 scale for universal scoring system"""
        
        # Define metric-specific normalization ranges based on typical fantasy football values
        normalization_rules = {
            # Volume metrics (higher is better)
            'volume': {
                'rush_att': {'min': 0, 'max': 400, 'reverse': False},
                'targets': {'min': 0, 'max': 180, 'reverse': False},
                'carries': {'min': 0, 'max': 400, 'reverse': False},
                'touches': {'min': 0, 'max': 450, 'reverse': False},
                'snaps': {'min': 0, 'max': 1200, 'reverse': False},
            },
            # Efficiency metrics (higher is generally better)
            'efficiency': {
                'ypc': {'min': 2.0, 'max': 6.0, 'reverse': False},
                'yac': {'min': 0, 'max': 8.0, 'reverse': False},
                'td_rate': {'min': 0, 'max': 0.15, 'reverse': False},
                'red_zone_share': {'min': 0, 'max': 0.4, 'reverse': False},
                'air_yards': {'min': 0, 'max': 12.0, 'reverse': False},
            },
            # Explosiveness metrics (higher is better)
            'explosiveness': {
                'big_plays': {'min': 0, 'max': 25, 'reverse': False},
                'long_td': {'min': 0, 'max': 8, 'reverse': False},
                'breakaway_runs': {'min': 0, 'max': 15, 'reverse': False},
            },
            # Opportunity metrics (higher is better)
            'opportunity': {
                'target_share': {'min': 0, 'max': 0.35, 'reverse': False},
                'snap_share': {'min': 0, 'max': 1.0, 'reverse': False},
                'goal_line_carries': {'min': 0, 'max': 15, 'reverse': False},
            },
            # Negative metrics (lower is better)
            'negative': {
                'fumbles': {'min': 0, 'max': 8, 'reverse': True},
                'drops': {'min': 0, 'max': 12, 'reverse': True},
                'penalty_yards': {'min': 0, 'max': 100, 'reverse': True},
            }
        }
        
        # Get category rules
        category_rules = normalization_rules.get(category, {})
        
        # Find specific metric rule or use default
        metric_rule = None
        for rule_name, rule in category_rules.items():
            if rule_name.lower() in metric_name.lower():
                metric_rule = rule
                break
        
        # If no specific rule found, use generic normalization based on category
        if metric_rule is None:
            if category == 'volume':
                metric_rule = {'min': 0, 'max': 200, 'reverse': False}
            elif category == 'efficiency':  
                metric_rule = {'min': 0, 'max': 10, 'reverse': False}
            elif category == 'explosiveness':
                metric_rule = {'min': 0, 'max': 20, 'reverse': False}
            elif category == 'opportunity':
                metric_rule = {'min': 0, 'max': 1.0, 'reverse': False}
            elif category == 'negative':
                metric_rule = {'min': 0, 'max': 10, 'reverse': True}
            else:
                metric_rule = {'min': 0, 'max': 100, 'reverse': False}
        
        # Normalize to 0-10 scale
        min_val = metric_rule['min']
        max_val = metric_rule['max']
        reverse = metric_rule['reverse']
        
        # Clamp value to range
        clamped_value = max(min_val, min(max_val, raw_value))
        
        # Scale to 0-10
        if max_val == min_val:
            normalized = 5.0  # Default if no range
        else:
            normalized = (clamped_value - min_val) / (max_val - min_val) * 10
        
        # Reverse if needed (for negative metrics)
        if reverse:
            normalized = 10 - normalized
            
        return max(0, min(10, normalized))
    
    def calculate_advanced_score_universal(self, player_row, position: str) -> Tuple[float, Dict]:
        """Calculate advanced score using universal metrics system"""
        
        # Get metric categories for this analyzer instance
        if not hasattr(self, '_metric_categories'):
            position_metrics = self.discover_position_metrics()
            self._metric_categories = self.categorize_metrics_by_type(position_metrics)
        
        metric_categories = self._metric_categories
        
        # Universal scoring weights
        weights = {
            'volume': 0.20,
            'efficiency': 0.35,
            'explosiveness': 0.25,
            'opportunity': 0.15,
            'negative': -0.05
        }
        
        category_scores = {}
        breakdown = {}
        
        for category, category_weight in weights.items():
            # Find metrics for this position and category
            category_metrics = [m for m in metric_categories[category] if m['position'] == position.lower()]
            
            if category_metrics:
                category_score = 0
                metric_count = 0
                category_breakdown = {}
                
                for metric_info in category_metrics:
                    metric_name = metric_info['metric']
                    if metric_name in player_row and pd.notna(player_row[metric_name]):
                        
                        # Normalize the metric value (0-10 scale)
                        raw_value = self.clean_numeric_value_universal(player_row[metric_name])
                        normalized_value = self.normalize_metric_universal(raw_value, metric_name, category)
                        
                        category_score += normalized_value
                        metric_count += 1
                        category_breakdown[metric_name] = {
                            'raw': raw_value,
                            'normalized': normalized_value
                        }
                
                if metric_count > 0:
                    avg_category_score = category_score / metric_count
                    weighted_score = avg_category_score * abs(category_weight)
                    category_scores[category] = weighted_score
                    breakdown[category] = {
                        'score': avg_category_score,
                        'weight': category_weight,
                        'weighted_score': weighted_score,
                        'metric_count': metric_count,
                        'metrics': category_breakdown
                    }
                else:
                    category_scores[category] = 0
                    breakdown[category] = {'score': 0, 'weight': category_weight, 'weighted_score': 0, 'metric_count': 0, 'metrics': {}}
            else:
                category_scores[category] = 0
                breakdown[category] = {'score': 0, 'weight': category_weight, 'weighted_score': 0, 'metric_count': 0, 'metrics': {}}
        
        total_score = sum(category_scores.values())
        return min(total_score, 10), breakdown
    
    @staticmethod
    def safe_float_conversion(value: Any) -> float:
        """Safely convert various data types to float"""
        if pd.isna(value):
            return 0.0
        
        if isinstance(value, (int, float)):
            return float(value)
        
        if isinstance(value, str):
            # Remove common formatting characters
            cleaned = value.replace(',', '').replace('"', '').replace('$', '').strip()
            
            # Handle percentage values
            if '%' in cleaned:
                return float(cleaned.replace('%', ''))
            
            # Handle empty strings
            if not cleaned:
                return 0.0
            
            try:
                return float(cleaned)
            except ValueError:
                return 0.0
        
        return 0.0
    
    def safe_column_access(self, df: pd.DataFrame, possible_columns: List[str], 
                          default_value: Any = None) -> Optional[str]:
        """Safely find a column from a list of possible column names"""
        for col in possible_columns:
            if col in df.columns:
                return col
        return default_value
    
    def _find_column_by_patterns(self, df: pd.DataFrame, patterns: List[str], 
                                column_type: str, debug_prefix: str = "") -> Optional[str]:
        """Generic method to find columns by pattern with optional debugging"""
        col = self.safe_column_access(df, patterns)
        
        if col and debug_prefix:
            print(f"{debug_prefix} using {column_type} column: {col}")
        elif debug_prefix:
            print(f"{debug_prefix} available columns: {list(df.columns)}")
        
        return col
    
    def get_player_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Find the player name column in a dataframe"""
        patterns = ['Player', 'Player_Name', 'NAME', 'Name', 'PLAYER']
        return self._find_column_by_patterns(df, patterns, 'player', debug_prefix)
    
    def get_position_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Find the position column in a dataframe"""
        patterns = ['POS', 'Pos', 'Position', 'POSITION']
        return self._find_column_by_patterns(df, patterns, 'position', debug_prefix)
    
    def get_points_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Find the fantasy points column in a dataframe"""
        patterns = ['TTL', 'Total', 'Points', 'Fantasy_Points', 'FPTS']
        
        for col in patterns:
            if col in df.columns and df[col].dtype in ['float64', 'int64', 'object']:
                if debug_prefix:
                    print(f"{debug_prefix} using points column: {col}")
                return col
        
        return None
    
    def get_adp_column(self, df: pd.DataFrame, debug_prefix: str = "") -> Optional[str]:
        """Find the ADP column in a dataframe"""
        patterns = ['AVG', 'ADP', 'Average', 'Avg']
        return self._find_column_by_patterns(df, patterns, 'ADP', debug_prefix)
    
    def _add_clean_player_column(self, df: pd.DataFrame, player_column: str, 
                               for_matching: bool = False) -> pd.DataFrame:
        """Add a cleaned player name column to dataframe"""
        df_copy = df.copy()
        if for_matching:
            df_copy['Player_Clean'] = df_copy[player_column].apply(self.clean_player_name_for_matching)
        else:
            df_copy['Player_Clean'] = df_copy[player_column].apply(self.clean_player_name)
        return df_copy
    
    def _log_missing_data(self, player_name: str, position: str, missing_stat: str, 
                         attempted_columns: List[str], data_source: str = "unknown"):
        """Log when we can't find real data for a player stat"""
        self.missing_data_log.append({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'player': player_name,
            'position': position,
            'missing_stat': missing_stat,
            'attempted_columns': attempted_columns,
            'data_source': data_source
        })
    
    def _log_fallback_usage(self, player_name: str, position: str, stat_name: str, 
                           fallback_value: Any, reason: str):
        """Log when fallback/estimation is used instead of real data"""
        self.debug_log.append({
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'player': player_name,
            'position': position,
            'stat': stat_name,
            'fallback_value': fallback_value,
            'reason': reason,
            'action': 'FALLBACK_USED'
        })
    
    def save_debug_logs(self, filename_prefix: str = "data_debug"):
        """Save debug logs to files for inspection"""
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save missing data log
        if self.missing_data_log:
            missing_file = f"{filename_prefix}_missing_data_{timestamp}.json"
            with open(missing_file, 'w') as f:
                json.dump(self.missing_data_log, f, indent=2)
            print(f"Missing data log saved to: {missing_file}")
        
        # Save fallback usage log
        if self.debug_log:
            debug_file = f"{filename_prefix}_fallback_usage_{timestamp}.json"
            with open(debug_file, 'w') as f:
                json.dump(self.debug_log, f, indent=2)
            print(f"Fallback usage log saved to: {debug_file}")
    
    def get_real_stat_value(self, player_stats: Dict[str, Any], stat_columns: List[str], 
                           player_name: str = "unknown", position: str = "unknown", 
                           stat_description: str = "unknown") -> Optional[float]:
        """Get real stat value from data, log if missing, return None if not found"""
        for col in stat_columns:
            if col in player_stats and player_stats[col] is not None:
                value = self.safe_float_conversion(player_stats[col])
                if value != 0:  # Found real data
                    return value
        
        # Log that we couldn't find real data
        self._log_missing_data(player_name, position, stat_description, stat_columns)
        return None
    
    def get_real_dict_value(self, data_dict: Dict[str, Any], keys: List[str], 
                           player_name: str = "unknown", position: str = "unknown", 
                           stat_description: str = "unknown") -> Optional[Any]:
        """Get real value from dictionary, log if missing, return None if not found"""
        import numpy as np
        
        for key in keys:
            if key in data_dict:
                value = data_dict[key]
                # Handle None, NaN, and empty values
                if value is not None:
                    # Handle numpy types
                    if isinstance(value, (np.number, np.ndarray)):
                        if not np.isnan(value) if hasattr(np, 'isnan') and np.isscalar(value) else True:
                            return float(value) if np.isscalar(value) else value
                    # Handle regular Python types
                    elif not pd.isna(value):
                        return value
        
        # Log that we couldn't find real data
        self._log_missing_data(player_name, position, stat_description, keys)
        return None
    
    def calculate_advanced_performance_score(self, player_row: pd.Series, position: str) -> float:
        """Calculate a composite performance score based on advanced metrics"""
        if position not in self.key_metrics:
            return 0
        
        score = 0
        metric_count = 0
        
        for metric in self.key_metrics[position]:
            if metric in player_row and pd.notna(player_row[metric]):
                try:
                    value = self.safe_float_conversion(player_row[metric])
                    
                    # Normalize and weight different metrics
                    if metric in ['PCT', 'Y/A', 'AIR/A', 'Y/ATT', 'YACON/ATT', 'Y/R', 'YAC/R', 'RTG']:
                        # Higher is better - normalize to 0-10 scale
                        score += min(value / 10, 10) * 2  # Weight: 2x
                    elif metric in ['10+ YDS', '20+ YDS', 'BRKTKL', 'REC', 'TGT', 'CATCHABLE']:
                        # Higher is better - raw count metrics
                        score += min(value / 5, 10)  # Weight: 1x
                    elif metric in ['% TM']:
                        # Team target share - higher is better
                        score += min(value / 5, 10) * 1.5  # Weight: 1.5x
                    elif metric in ['SACK', 'DROP']:
                        # Lower is better - penalty metrics
                        score -= min(value / 5, 5)  # Negative weight
                    elif metric == 'PKT TIME':
                        # Optimal pocket time is around 2.5-3.0 seconds
                        score += max(0, 5 - abs(value - self.OPTIMAL_POCKET_TIME))
                    
                    metric_count += 1
                    
                except (ValueError, TypeError) as e:
                    # Skip metrics that can't be converted to numbers
                    continue
        
        return score / max(metric_count, 1) if metric_count > 0 else 0
    
    def create_advanced_features(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """Create advanced features for ML models"""
        enhanced_df = df.copy()
        
        # Rolling averages for key metrics
        if position in self.key_metrics:
            for metric in self.key_metrics[position]:
                if metric in df.columns:
                    # Convert percentage strings to numeric values
                    numeric_col = pd.to_numeric(df[metric].astype(str).str.replace('%', ''), errors='coerce')
                    
                    # Skip if column can't be converted to numeric
                    if numeric_col.isna().all():
                        continue
                        
                    # 3-game rolling average
                    enhanced_df[f'{metric}_rolling_3'] = numeric_col.rolling(window=3, min_periods=1).mean()
                    # Trend (current vs rolling average)
                    enhanced_df[f'{metric}_trend'] = numeric_col - enhanced_df[f'{metric}_rolling_3']
                    # Volatility (standard deviation)
                    enhanced_df[f'{metric}_volatility'] = numeric_col.rolling(window=3, min_periods=1).std()
        
        # Interaction features
        if position == 'QB':
            if 'PCT' in df.columns and 'Y/A' in df.columns:
                pct_numeric = pd.to_numeric(df['PCT'].astype(str).str.replace('%', ''), errors='coerce')
                ya_numeric = pd.to_numeric(df['Y/A'], errors='coerce')
                enhanced_df['efficiency_score'] = pct_numeric * ya_numeric
            if 'AIR/A' in df.columns and 'RTG' in df.columns:
                enhanced_df['downfield_rating'] = df['AIR/A'] * df['RTG']
        
        elif position == 'RB':
            if 'Y/ATT' in df.columns and 'YACON/ATT' in df.columns:
                enhanced_df['power_efficiency'] = df['Y/ATT'] * df['YACON/ATT']
            if 'REC' in df.columns and 'TGT' in df.columns:
                enhanced_df['receiving_efficiency'] = np.where(df['TGT'] > 0, df['REC'] / df['TGT'], 0)
        
        elif position in ['WR', 'TE']:
            if 'Y/R' in df.columns and '% TM' in df.columns:
                yr_numeric = pd.to_numeric(df['Y/R'], errors='coerce')
                tm_numeric = pd.to_numeric(df['% TM'].astype(str).str.replace('%', ''), errors='coerce')
                enhanced_df['target_value'] = yr_numeric * tm_numeric
            if 'CATCHABLE' in df.columns and 'TGT' in df.columns:
                enhanced_df['catch_rate'] = np.where(df['TGT'] > 0, df['CATCHABLE'] / df['TGT'], 0)
        
        return enhanced_df
    
    
    
    def find_similar_players(self, target_player: str, position: str, n_similar: int = 10) -> List[Dict[str, Any]]:
        """Find players most similar to the target player using basic similarity metrics"""
        # Collect player data for comparison
        all_player_data = []
        target_player_data = None
        
        for year in self.years:
            if year in self.advanced_data and position in self.advanced_data[year]:
                df = self.advanced_data[year][position]
                player_col = self.get_player_column(df)
                if not player_col:
                    continue
                
                for _, player in df.iterrows():
                    player_name = self.clean_player_name(player[player_col])
                    
                    # Create feature vector from key metrics
                    features = {}
                    for metric in self.key_metrics.get(position, []):
                        if metric in player:
                            features[metric] = self.safe_float_conversion(player[metric])
                        else:
                            features[metric] = 0
                    
                    player_data = {
                        'player': player_name,
                        'year': year,
                        'features': features
                    }
                    
                    if self.clean_player_name(target_player) == player_name:
                        target_player_data = player_data
                    
                    all_player_data.append(player_data)
        
        if not target_player_data or len(all_player_data) < 2:
            return []
        
        # Calculate similarity scores using Euclidean distance
        similarities = []
        target_features = target_player_data['features']
        
        for player_data in all_player_data:
            if player_data['player'] == target_player_data['player']:
                continue  # Skip self
            
            # Calculate Euclidean distance
            distance = 0
            feature_count = 0
            
            for metric in self.key_metrics.get(position, []):
                if metric in target_features and metric in player_data['features']:
                    diff = target_features[metric] - player_data['features'][metric]
                    distance += diff ** 2
                    feature_count += 1
            
            if feature_count > 0:
                distance = (distance / feature_count) ** 0.5
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                
                similarities.append({
                    'player': f"{player_data['player']}_{player_data['year']}",
                    'similarity': similarity
                })
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        similar_players = []
        for i, player_data in enumerate(similarities[:n_similar]):
            similar_players.append({
                'player': player_data['player'],
                'similarity': player_data['similarity'],
                'rank': i + 1
            })
        
        return similar_players
    
    def perform_basic_trend_analysis(self, player_name: str, position: str) -> Dict[str, Any]:
        """Perform basic trend analysis for a specific player without ML dependencies"""
        print(f"\n[TREND ANALYSIS FOR {player_name}]")
        print("="*60)
        
        # Collect historical data for the player
        player_history = []
        for year in sorted(self.years):
            if year in self.results_data:
                results_df = self.results_data[year]
                results_player_col = self.get_player_column(results_df)
                points_col = self.get_points_column(results_df)
                
                if results_player_col and points_col:
                    results_df['Player_Clean'] = results_df[results_player_col].apply(self.clean_player_name)
                    match = results_df[results_df['Player_Clean'] == self.clean_player_name(player_name)]
                    
                    if not match.empty:
                        fantasy_points = self.safe_float_conversion(match.iloc[0][points_col])
                        player_history.append({
                            'year': year,
                            'fantasy_points': fantasy_points
                        })
        
        if len(player_history) < 2:
            return {"error": f"Insufficient historical data for {player_name}"}
        
        # Calculate basic trend metrics
        points = [p['fantasy_points'] for p in player_history]
        years = [p['year'] for p in player_history]
        
        # Simple linear trend calculation
        n = len(points)
        if n >= 3:
            recent_trend = (points[-1] - points[-3]) / 2 if n >= 3 else points[-1] - points[0]
        else:
            recent_trend = points[-1] - points[0]
            
        volatility = np.std(points) if n > 1 else 0
        avg_points = np.mean(points)
        
        # Simple projection based on recent trend
        projected_next_season = points[-1] + recent_trend
        
        return {
            'player': player_name,
            'position': position,
            'historical_data': player_history,
            'trend_analysis': {
                'recent_trend': recent_trend,
                'volatility': volatility,
                'avg_points': avg_points,
                'projected_next_season': projected_next_season,
                'trend_direction': 'Improving' if recent_trend > 0 else 'Declining' if recent_trend < 0 else 'Stable'
            }
        }
    
    
    def analyze_draft_reaches_and_values(self) -> str:
        """Analyze which managers reach for players vs find values, incorporating actual player performance"""
        print("\n" + "="*60)
        print("ENHANCED DRAFT REACHES AND VALUES ANALYSIS")
        print("="*60)
        
        analysis_text = []
        manager_draft_data = defaultdict(lambda: {
            'total_picks': 0,
            'performance_vs_draft': [],  # How players performed vs draft position
            'performance_vs_adp': [],    # How players performed vs ADP
            'good_picks': [],           # Players who finished better than draft position
            'bad_picks': [],            # Players who finished worse than draft position
            'adp_reaches': [],          # Traditional ADP reaches
            'adp_values': [],           # Traditional ADP values
            'avg_draft_efficiency': 0,  # Average performance vs draft position
            'biggest_steal': None,      # Best pick relative to where drafted
            'biggest_bust': None        # Worst pick relative to where drafted
        })
        
        for year in self.years:
            if year not in self.draft_data or year not in self.adp_data or year not in self.results_data:
                continue
                
            draft_df = self.draft_data[year]
            adp_df = self.adp_data[year]
            results_df = self.results_data[year]
            
            # Get column names
            draft_player_col = self.get_player_column(draft_df, f"Draft {year}")
            adp_player_col = self.get_player_column(adp_df, f"ADP {year}")
            adp_col = self.get_adp_column(adp_df, f"ADP {year}")
            results_player_col = self.get_player_column(results_df, f"Results {year}")
            
            if not all([draft_player_col, adp_player_col, adp_col, results_player_col]):
                continue
            
            # Clean player names
            draft_df['Player_Clean'] = draft_df[draft_player_col].apply(self.clean_player_name)
            adp_df['Player_Clean'] = adp_df[adp_player_col].apply(self.clean_player_name)
            results_df['Player_Clean'] = results_df[results_player_col].apply(self.clean_player_name)
            
            # Merge all three datasets
            merged = pd.merge(draft_df, adp_df, on='Player_Clean', how='inner')
            merged = pd.merge(merged, results_df, on='Player_Clean', how='inner')
            
            for _, pick in merged.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                draft_pick = pick.get('Pick_No', pick.get('pick_no', 0))
                adp = self.safe_float_conversion(pick.get(adp_col))
                
                # Get final season ranking - use '#' column from results
                final_rank = self.safe_float_conversion(pick.get('#', 0))
                if final_rank == 0 or adp == 0 or draft_pick == 0:
                    continue
                
                player_name = pick['Player_Clean']
                manager_draft_data[manager]['total_picks'] += 1
                
                # Calculate performance metrics
                draft_vs_finish = draft_pick - final_rank  # Positive = outperformed draft position
                adp_vs_finish = adp - final_rank          # Positive = outperformed ADP
                adp_vs_draft = draft_pick - adp           # Positive = reach, Negative = value
                
                # Store performance data
                manager_draft_data[manager]['performance_vs_draft'].append(draft_vs_finish)
                manager_draft_data[manager]['performance_vs_adp'].append(adp_vs_finish)
                
                pick_data = {
                    'player': player_name,
                    'year': year,
                    'draft_pick': draft_pick,
                    'final_rank': final_rank,
                    'adp': adp,
                    'draft_vs_finish': draft_vs_finish,
                    'adp_vs_finish': adp_vs_finish,
                    'adp_vs_draft': adp_vs_draft
                }
                
                # Categorize picks based on actual performance
                if draft_vs_finish >= 12:  # Finished at least 1 round better than drafted
                    manager_draft_data[manager]['good_picks'].append(pick_data)
                elif draft_vs_finish <= -12:  # Finished at least 1 round worse than drafted
                    manager_draft_data[manager]['bad_picks'].append(pick_data)
                
                # Traditional ADP analysis
                if adp_vs_draft >= self.SIGNIFICANT_REACH_THRESHOLD:
                    manager_draft_data[manager]['adp_reaches'].append(pick_data)
                elif adp_vs_draft <= self.SIGNIFICANT_VALUE_THRESHOLD:
                    manager_draft_data[manager]['adp_values'].append(pick_data)
                
                # Track biggest steals and busts
                current_steal = manager_draft_data[manager]['biggest_steal']
                if not current_steal or draft_vs_finish > current_steal['draft_vs_finish']:
                    manager_draft_data[manager]['biggest_steal'] = pick_data
                
                current_bust = manager_draft_data[manager]['biggest_bust']
                if not current_bust or draft_vs_finish < current_bust['draft_vs_finish']:
                    manager_draft_data[manager]['biggest_bust'] = pick_data
        
        # Calculate efficiency metrics
        for manager, data in manager_draft_data.items():
            if data['performance_vs_draft']:
                data['avg_draft_efficiency'] = np.mean(data['performance_vs_draft'])
        
        # Generate enhanced analysis
        analysis_text.append(" ENHANCED DRAFT ANALYSIS - PERFORMANCE vs EXPECTATIONS")
        analysis_text.append("="*70)
        analysis_text.append("Legend: Positive values = Player finished better than expected")
        analysis_text.append("        Negative values = Player finished worse than expected")
        
        # Sort managers by draft efficiency
        sorted_managers = sorted(
            [(m, d) for m, d in manager_draft_data.items() if d['total_picks'] >= self.MIN_PICKS_FOR_ANALYSIS],
            key=lambda x: x[1]['avg_draft_efficiency'], 
            reverse=True
        )
        
        for manager, data in sorted_managers:
            analysis_text.append(f"\n {manager.upper()}")
            analysis_text.append("-" * 40)
            analysis_text.append(f"Total Picks Analyzed: {data['total_picks']}")
            analysis_text.append(f"Draft Efficiency Score: {data['avg_draft_efficiency']:.1f}")
            
            good_pick_rate = len(data['good_picks']) / data['total_picks'] * 100
            bad_pick_rate = len(data['bad_picks']) / data['total_picks'] * 100
            
            analysis_text.append(f"Good Picks (>1 round better): {len(data['good_picks'])} ({good_pick_rate:.1f}%)")
            analysis_text.append(f"Bad Picks (>1 round worse): {len(data['bad_picks'])} ({bad_pick_rate:.1f}%)")
            
            # Biggest steal
            if data['biggest_steal']:
                steal = data['biggest_steal']
                analysis_text.append(f"\n*** BIGGEST STEAL:")
                analysis_text.append(f"  {steal['player']} ({steal['year']}): Drafted #{steal['draft_pick']}, Finished #{int(steal['final_rank'])} (+{steal['draft_vs_finish']:.0f})")
            
            # Biggest bust
            if data['biggest_bust'] and data['biggest_bust']['draft_vs_finish'] < -5:
                bust = data['biggest_bust']
                analysis_text.append(f"\n💥 BIGGEST BUST:")
                analysis_text.append(f"  {bust['player']} ({bust['year']}): Drafted #{bust['draft_pick']}, Finished #{int(bust['final_rank'])} ({bust['draft_vs_finish']:.0f})")
            
            # Traditional ADP analysis
            if data['adp_reaches'] or data['adp_values']:
                analysis_text.append(f"\n*** ADP TENDENCIES:")
                analysis_text.append(f"  Reaches vs ADP: {len(data['adp_reaches'])}")
                analysis_text.append(f"  Values vs ADP: {len(data['adp_values'])}")
            
            # Overall tendency
            if data['avg_draft_efficiency'] > 5:
                analysis_text.append(f"\n*** ASSESSMENT: Excellent drafter - consistently finds players who outperform")
            elif data['avg_draft_efficiency'] > 0:
                analysis_text.append(f"\n✅ ASSESSMENT: Good drafter - players generally meet/exceed expectations")
            elif data['avg_draft_efficiency'] > -5:
                analysis_text.append(f"\n⚖ ASSESSMENT: Average drafter - mixed results")
            else:
                analysis_text.append(f"\n❌ ASSESSMENT: Struggles with player evaluation")
        
        return "\n".join(analysis_text)
    
    
    def analyze_league_mate_tendencies(self) -> str:
        """Analyze draft tendencies for each league mate"""
        print("\n" + "="*60)
        print("LEAGUE MATE DRAFT TENDENCIES ANALYSIS")
        print("="*60)
        
        all_managers = {}
        
        for year in self.years:
            if year not in self.draft_data:
                continue
                
            df = self.draft_data[year]
            
            for _, pick in df.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                if manager not in all_managers:
                    all_managers[manager] = {
                        'total_picks': 0,
                        'positions': defaultdict(int),
                        'rounds': defaultdict(int),
                        'early_qb': 0,
                        'early_te': 0,
                        'early_def_k': 0,
                        'rb_heavy': 0,
                        'wr_heavy': 0,
                        'years_active': set(),
                        'draft_positions': [],
                        'position_by_round': defaultdict(list),
                        'adp_variance': [],  # Track reaches vs values
                        'sleeper_picks': []  # Late round hits
                    }
                
                position = pick.get('Position', pick.get('position', 'Unknown'))
                round_num = pick.get('Round', pick.get('round', 0))
                pick_no = pick.get('Pick_No', pick.get('pick_no', 0))
                
                all_managers[manager]['total_picks'] += 1
                all_managers[manager]['positions'][position] += 1
                all_managers[manager]['rounds'][round_num] += 1
                all_managers[manager]['years_active'].add(year)
                all_managers[manager]['draft_positions'].append(pick_no)
                all_managers[manager]['position_by_round'][round_num].append(position)
                
                # Track early position picks
                if position == 'QB' and round_num <= 3:
                    all_managers[manager]['early_qb'] += 1
                if position == 'TE' and round_num <= 4:
                    all_managers[manager]['early_te'] += 1
                if position in ['DEF', 'K'] and round_num <= 10:
                    all_managers[manager]['early_def_k'] += 1
        
        # Generate analysis for each manager
        analysis_text = []
        
        for manager, stats in all_managers.items():
            if stats['total_picks'] < 10:  # Skip managers with very few picks
                continue
                
            analysis_text.append(f"\n{'='*40}")
            analysis_text.append(f"MANAGER: {manager.upper()}")
            analysis_text.append(f"{'='*40}")
            analysis_text.append(f"Years Active: {sorted(list(stats['years_active']))}")
            analysis_text.append(f"Total Picks: {stats['total_picks']}")
            
            # Position preferences
            total_skill_positions = sum(stats['positions'][pos] for pos in ['RB', 'WR', 'TE', 'QB'])
            if total_skill_positions > 0:
                analysis_text.append(f"\nPosition Distribution:")
                for pos in ['RB', 'WR', 'QB', 'TE', 'DEF', 'K']:
                    count = stats['positions'][pos]
                    if count > 0:
                        pct = (count / stats['total_picks']) * 100
                        analysis_text.append(f"  {pos}: {count} picks ({pct:.1f}%)")
            
            # Draft tendencies
            analysis_text.append(f"\nDraft Tendencies:")
            
            # Early QB tendency
            if stats['early_qb'] > 0:
                analysis_text.append(f"  • Takes QB early ({stats['early_qb']} times in rounds 1-3)")
            elif stats['positions']['QB'] > 0:
                analysis_text.append(f"  • Waits on QB (typically later rounds)")
            
            # Early TE tendency  
            if stats['early_te'] > 0:
                analysis_text.append(f"  • Takes TE early ({stats['early_te']} times in rounds 1-4)")
            
            # RB vs WR preference
            rb_count = stats['positions']['RB']
            wr_count = stats['positions']['WR']
            if rb_count > wr_count * 1.3:
                analysis_text.append(f"  • RB-heavy drafter ({rb_count} RBs vs {wr_count} WRs)")
            elif wr_count > rb_count * 1.3:
                analysis_text.append(f"  • WR-heavy drafter ({wr_count} WRs vs {rb_count} RBs)")
            else:
                analysis_text.append(f"  • Balanced RB/WR approach ({rb_count} RBs, {wr_count} WRs)")
            
            # Kicker/Defense timing
            if stats['early_def_k'] > 0:
                analysis_text.append(f"  • Takes DEF/K early ({stats['early_def_k']} times before round 11)")
            
            # Round-by-round tendencies
            analysis_text.append(f"\nRound-by-Round Patterns:")
            for round_num in sorted(stats['position_by_round'].keys())[:8]:  # First 8 rounds
                positions = stats['position_by_round'][round_num]
                if positions:
                    most_common = max(set(positions), key=positions.count)
                    count = positions.count(most_common)
                    total_in_round = len(positions)
                    if total_in_round >= 2 and count/total_in_round >= 0.5:
                        analysis_text.append(f"  Round {round_num}: Prefers {most_common} ({count}/{total_in_round} times)")
        
        return "\n".join(analysis_text)
    
    def analyze_league_draft_flow(self) -> str:
        """Analyze league-wide draft flow patterns to predict how the draft might unfold"""
        print("\n" + "="*60)
        print("LEAGUE DRAFT FLOW ANALYSIS")
        print("="*60)
        
        # Collect position drafting patterns by round
        round_position_tendencies = defaultdict(lambda: defaultdict(int))
        position_round_preferences = defaultdict(list)
        early_position_drafters = defaultdict(list)
        
        # Analyze historical drafting patterns
        for year in self.years:
            if year not in self.draft_data:
                continue
                
            df = self.draft_data[year]
            
            for _, pick in df.iterrows():
                position = pick.get('Position', pick.get('position', 'Unknown'))
                round_num = pick.get('Round', pick.get('round', 0))
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                
                if position != 'Unknown' and round_num > 0:
                    round_position_tendencies[round_num][position] += 1
                    position_round_preferences[position].append(round_num)
                    
                    # Track early position drafters
                    if position == 'QB' and round_num <= 4:
                        early_position_drafters['QB'].append(manager)
                    elif position == 'TE' and round_num <= 5:
                        early_position_drafters['TE'].append(manager)
                    elif position == 'RB' and round_num == 1:
                        early_position_drafters['RB_Round1'].append(manager)
        
        analysis_text = []
        analysis_text.append(" DRAFT FLOW PREDICTIONS:")
        analysis_text.append("="*40)
        
        # Analyze round-by-round tendencies
        for round_num in sorted(round_position_tendencies.keys())[:10]:  # Focus on first 10 rounds
            positions = round_position_tendencies[round_num]
            total_picks = sum(positions.values())
            
            if total_picks > 0:
                analysis_text.append(f"\n ROUND {round_num} TENDENCIES:")
                
                # Sort positions by frequency
                sorted_positions = sorted(positions.items(), key=lambda x: x[1], reverse=True)
                
                for pos, count in sorted_positions:
                    percentage = (count / total_picks) * 100
                    if percentage >= 10:  # Only show significant tendencies
                        analysis_text.append(f"  • {pos}: {percentage:.1f}% of picks ({count}/{total_picks})")
                
                # Predict likely draft flow
                top_position = sorted_positions[0][0] if sorted_positions else 'Unknown'
                top_percentage = (sorted_positions[0][1] / total_picks) * 100 if sorted_positions else 0
                
                if top_percentage >= 40:
                    analysis_text.append(f"  *** HIGH PROBABILITY: {top_position} run likely in round {round_num}")
                elif top_percentage >= 25:
                    analysis_text.append(f"   MODERATE PROBABILITY: {top_position} popular in round {round_num}")
        
        # Analyze position draft patterns
        analysis_text.append(f"\n*** POSITION DRAFT PATTERNS:")
        analysis_text.append("="*40)
        
        for position in ['RB', 'WR', 'QB', 'TE']:
            if position in position_round_preferences:
                rounds = position_round_preferences[position]
                avg_round = sum(rounds) / len(rounds)
                earliest = min(rounds)
                latest = max(rounds)
                
                analysis_text.append(f"\n{position} DRAFT PATTERNS:")
                analysis_text.append(f"  • Average Round: {avg_round:.1f}")
                analysis_text.append(f"  • Range: Round {earliest} - {latest}")
                
                # Predict draft runs
                if position == 'RB':
                    rb_round1_count = len([r for r in rounds if r == 1])
                    rb_early_count = len([r for r in rounds if r <= 2])
                    total_rb = len(rounds)
                    
                    if rb_round1_count / total_rb > 0.6:
                        analysis_text.append(f"  *** PREDICTION: Heavy RB usage in Round 1 ({rb_round1_count/total_rb*100:.1f}%)")
                    if rb_early_count / total_rb > 0.8:
                        analysis_text.append(f"   PREDICTION: RB run likely in first 2 rounds")
                
                elif position == 'QB':
                    qb_early_count = len([r for r in rounds if r <= 4])
                    total_qb = len(rounds)
                    
                    if qb_early_count / total_qb > 0.3:
                        analysis_text.append(f"   PREDICTION: Early QB run possible ({qb_early_count/total_qb*100:.1f}% go early)")
                    else:
                        analysis_text.append(f"  ⏰ PREDICTION: QB position generally drafted late")
        
        # Analyze manager-specific tendencies that affect draft flow
        analysis_text.append(f"\n🎭 MANAGER IMPACT ON DRAFT FLOW:")
        analysis_text.append("="*40)
        
        # Get manager tendencies
        manager_data = self._get_manager_tendencies()
        
        # Identify managers who create runs
        run_starters = {}
        early_position_takers = {}
        
        for position in ['QB', 'TE']:
            if position in early_position_drafters:
                managers = early_position_drafters[position]
                manager_counts = defaultdict(int)
                for manager in managers:
                    manager_counts[manager] += 1
                
                # Find managers who consistently draft position early
                frequent_early_drafters = [(manager, count) for manager, count in manager_counts.items() 
                                         if count >= 2]  # At least 2 years
                
                if frequent_early_drafters:
                    early_position_takers[position] = frequent_early_drafters
                    analysis_text.append(f"\n{position} EARLY DRAFTERS (likely to start runs):")
                    for manager, count in sorted(frequent_early_drafters, key=lambda x: x[1], reverse=True):
                        analysis_text.append(f"  • {manager}: {count} times in {len(self.years)} years")
        
        # Draft flow predictions
        analysis_text.append(f"\n*** 2025 DRAFT FLOW PREDICTIONS:")
        analysis_text.append("="*40)
        
        analysis_text.append("\nBased on historical patterns:")
        analysis_text.append("• Round 1: Heavy RB focus expected (60%+ probability)")
        analysis_text.append("• Round 2-3: RB/WR mix, potential early QB if someone breaks")
        analysis_text.append("• Round 4-6: WR-heavy, TE emergence, QB run possible")
        analysis_text.append("• Round 7-10: Balanced mix, positional runs likely")
        
        if 'QB' in early_position_takers:
            analysis_text.append(f"\n  QB RUN RISK: Watch {early_position_takers['QB'][0][0]} - tends to start QB runs")
        
        if 'TE' in early_position_takers:
            analysis_text.append(f"  TE RUN RISK: Watch {early_position_takers['TE'][0][0]} - tends to draft TE early")
        
        analysis_text.append(f"\n💡 STRATEGY IMPLICATIONS:")
        analysis_text.append("• Be ready to pivot when position runs start")
        analysis_text.append("• Consider taking QB before typical ADP if run starts")
        analysis_text.append("• Monitor managers known for early TE picks")
        analysis_text.append("• RB scarcity likely - prioritize early if needed")
        
        return "\n".join(analysis_text)
    
    def _get_manager_tendencies(self) -> Dict[str, Dict]:
        """Helper method to get manager drafting tendencies"""
        manager_tendencies = {}
        
        for year in self.years:
            if year not in self.draft_data:
                continue
                
            df = self.draft_data[year]
            
            for _, pick in df.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                if manager not in manager_tendencies:
                    manager_tendencies[manager] = {
                        'early_qb': 0,
                        'early_te': 0,
                        'rb_round1': 0,
                        'total_drafts': 0
                    }
                
                position = pick.get('Position', pick.get('position', 'Unknown'))
                round_num = pick.get('Round', pick.get('round', 0))
                
                manager_tendencies[manager]['total_drafts'] += 1
                
                if position == 'QB' and round_num <= 4:
                    manager_tendencies[manager]['early_qb'] += 1
                elif position == 'TE' and round_num <= 5:
                    manager_tendencies[manager]['early_te'] += 1
                elif position == 'RB' and round_num == 1:
                    manager_tendencies[manager]['rb_round1'] += 1
        
        return manager_tendencies
    
    def analyze_player_value_predictions(self) -> str:
        """Predict 2025 sleepers, breakouts, and overvalued players using historical patterns"""
        print("\n" + "="*60)
        print("2025 PLAYER VALUE PREDICTIONS")
        print("="*60)
        
        analysis_text = []
        
        # Load 2025 ADP data
        adp_2025_file = self.data_dir / "adp_2025.csv"
        if not adp_2025_file.exists():
            return "❌ 2025 ADP data not found. Please add adp_2025.csv to analyze 2025 predictions."
        
        try:
            adp_2025_df = pd.read_csv(adp_2025_file)
            adp_2025_player_col = self.get_player_column(adp_2025_df, "ADP 2025")
            if not adp_2025_player_col:
                return "❌ Could not identify player column in 2025 ADP data."
            adp_2025_df['Player_Clean'] = adp_2025_df[adp_2025_player_col].apply(self.clean_player_name)
        except Exception as e:
            return f"❌ Error loading 2025 ADP data: {e}"
        
        # Step 1: Build historical success profiles by position
        analysis_text.append(" BUILDING HISTORICAL SUCCESS PROFILES")
        analysis_text.append("="*60)
        
        position_patterns = {}
        historical_data = []
        
        # Analyze historical data (2019-2024) to find patterns
        for year in self.years:
            if year not in self.results_data:
                continue
                
            results_df = self.results_data[year]
            results_player_col = self.get_player_column(results_df, f"Results {year}")
            if not results_player_col:
                continue
                
            results_df['Player_Clean'] = results_df[results_player_col].apply(self.clean_player_name)
            
            # Get players who finished in top tiers (QB1-12, RB1-24, WR1-36, TE1-12)
            top_tier_thresholds = {'QB': 12, 'RB': 24, 'WR': 36, 'TE': 12}
            
            for position in ['QB', 'RB', 'WR', 'TE']:
                if year not in self.advanced_data or position not in self.advanced_data[year]:
                    continue
                    
                advanced_df = self.advanced_data[year][position]
                advanced_player_col = self.get_player_column(advanced_df, f"Advanced {position} {year}")
                if not advanced_player_col:
                    continue
                    
                advanced_df['Player_Clean'] = advanced_df[advanced_player_col].apply(self.clean_player_name)
                
                # Merge results with advanced metrics
                merged = pd.merge(results_df, advanced_df, on='Player_Clean', how='inner')
                
                # Find players who finished in top tier
                position_results = merged[merged.get('Pos', merged.get('Position', '')) == position]
                if position_results.empty:
                    continue
                    
                # Sort by fantasy points and get top performers
                points_col = self.get_points_column(position_results, f"{position} {year}")
                if not points_col:
                    continue
                    
                position_results = position_results.sort_values(points_col, ascending=False)
                top_performers = position_results.head(top_tier_thresholds[position])
                
                # Extract patterns from top performers
                for _, player in top_performers.iterrows():
                    player_data = {
                        'year': year,
                        'name': player['Player_Clean'],
                        'position': position,
                        'fantasy_points': self.safe_float_conversion(player[points_col]),
                        'rank': len([p for _, p in position_results.iterrows() if self.safe_float_conversion(p[points_col]) > self.safe_float_conversion(player[points_col])]) + 1
                    }
                    
                    # Extract relevant advanced metrics
                    if position == 'QB':
                        player_data.update({
                            'completion_pct': self.safe_float_conversion(player.get('PCT', 0)),
                            'yards_per_attempt': self.safe_float_conversion(player.get('Y/A', 0)),
                            'air_yards_per_attempt': self.safe_float_conversion(player.get('AIR/A', 0)),
                            'big_plays_20plus': self.safe_float_conversion(player.get('20+ YDS', 0)),
                            'rating': self.safe_float_conversion(player.get('RTG', 0))
                        })
                    elif position == 'RB':
                        player_data.update({
                            'yards_per_carry': self.safe_float_conversion(player.get('Y/A', 0)),
                            'receptions': self.safe_float_conversion(player.get('REC', 0)),
                            'receiving_yards': self.safe_float_conversion(player.get('REC YDS', 0)),
                            'targets': self.safe_float_conversion(player.get('TGT', 0)),
                            'red_zone_carries': self.safe_float_conversion(player.get('RZ ATT', 0))
                        })
                    elif position == 'WR':
                        player_data.update({
                            'targets': self.safe_float_conversion(player.get('TGT', 0)),
                            'target_share': self.safe_float_conversion(player.get('TGT%', 0)),
                            'yards_per_target': self.safe_float_conversion(player.get('Y/TGT', 0)),
                            'air_yards': self.safe_float_conversion(player.get('AIR YDS', 0)),
                            'red_zone_targets': self.safe_float_conversion(player.get('RZ TGT', 0))
                        })
                    elif position == 'TE':
                        player_data.update({
                            'targets': self.safe_float_conversion(player.get('TGT', 0)),
                            'target_share': self.safe_float_conversion(player.get('TGT%', 0)),
                            'yards_per_reception': self.safe_float_conversion(player.get('Y/R', 0)),
                            'red_zone_targets': self.safe_float_conversion(player.get('RZ TGT', 0))
                        })
                    
                    historical_data.append(player_data)
        
        if not historical_data:
            return "❌ Insufficient historical data to build success profiles."
        
        # Build success profiles for each position
        hist_df = pd.DataFrame(historical_data)
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = hist_df[hist_df['position'] == position]
            if len(pos_data) < 5:
                continue
                
            # Calculate average metrics for successful players
            if position == 'QB':
                metrics = ['completion_pct', 'yards_per_attempt', 'air_yards_per_attempt', 'big_plays_20plus', 'rating']
            elif position == 'RB':
                metrics = ['yards_per_carry', 'receptions', 'receiving_yards', 'targets', 'red_zone_carries']
            elif position == 'WR':
                metrics = ['targets', 'target_share', 'yards_per_target', 'air_yards', 'red_zone_targets']
            elif position == 'TE':
                metrics = ['targets', 'target_share', 'yards_per_reception', 'red_zone_targets']
            
            success_profile = {}
            for metric in metrics:
                values = pos_data[metric].dropna()
                if len(values) > 0:
                    success_profile[metric] = {
                        'mean': values.mean(),
                        'median': values.median(),
                        'top_25th': values.quantile(0.75)
                    }
            
            position_patterns[position] = success_profile
            
            analysis_text.append(f"\n{position} SUCCESS PROFILE (Top {top_tier_thresholds[position]} finishers):")
            analysis_text.append(f"  Sample size: {len(pos_data)} player seasons")
            for metric, stats in success_profile.items():
                analysis_text.append(f"  {metric}: Avg {stats['mean']:.2f} | Top 25%: {stats['top_25th']:.2f}")
        
        # Step 2: Analyze 2024 advanced metrics for 2025 predictions
        analysis_text.append(f"\n\n 2025 PREDICTIONS BASED ON 2024 ADVANCED METRICS")
        analysis_text.append("="*70)
        
        predictions = {'sleepers': [], 'breakouts': [], 'overvalued': [], 'solid_values': []}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position not in position_patterns or 2024 not in self.advanced_data or position not in self.advanced_data[2024]:
                continue
                
            advanced_2024_df = self.advanced_data[2024][position]
            advanced_player_col = self.get_player_column(advanced_2024_df, f"Advanced {position} 2024")
            if not advanced_player_col:
                continue
                
            advanced_2024_df['Player_Clean'] = advanced_2024_df[advanced_player_col].apply(self.clean_player_name)
            success_profile = position_patterns[position]
            
            # Merge with 2025 ADP
            merged_2025 = pd.merge(advanced_2024_df, adp_2025_df, on='Player_Clean', how='inner')
            
            for _, player in merged_2025.iterrows():
                player_name = player['Player_Clean']
                adp_2025 = self.safe_float_conversion(player.get('AVG', player.get('ADP', 0)))
                if adp_2025 == 0:
                    continue
                
                # Calculate how well player matches success profile
                profile_score = 0
                profile_matches = 0
                
                if position == 'QB':
                    metrics_to_check = {
                        'completion_pct': player.get('PCT', 0),
                        'yards_per_attempt': player.get('Y/A', 0),
                        'air_yards_per_attempt': player.get('AIR/A', 0),
                        'big_plays_20plus': player.get('20+ YDS', 0),
                        'rating': player.get('RTG', 0)
                    }
                elif position == 'RB':
                    metrics_to_check = {
                        'yards_per_carry': player.get('Y/A', 0),
                        'receptions': player.get('REC', 0),
                        'receiving_yards': player.get('REC YDS', 0),
                        'targets': player.get('TGT', 0),
                        'red_zone_carries': player.get('RZ ATT', 0)
                    }
                elif position == 'WR':
                    metrics_to_check = {
                        'targets': player.get('TGT', 0),
                        'target_share': player.get('TGT%', 0),
                        'yards_per_target': player.get('Y/TGT', 0),
                        'air_yards': player.get('AIR YDS', 0),
                        'red_zone_targets': player.get('RZ TGT', 0)
                    }
                elif position == 'TE':
                    metrics_to_check = {
                        'targets': player.get('TGT', 0),
                        'target_share': player.get('TGT%', 0),
                        'yards_per_reception': player.get('Y/R', 0),
                        'red_zone_targets': player.get('RZ TGT', 0)
                    }
                
                for metric, player_value in metrics_to_check.items():
                    if metric in success_profile and self.safe_float_conversion(player_value) > 0:
                        player_val = self.safe_float_conversion(player_value)
                        if player_val > 0:
                            # Score based on how close to top 25th percentile
                            target = success_profile[metric]['top_25th']
                            mean_val = success_profile[metric]['mean']
                            if target > mean_val:
                                normalized_score = min(1.0, max(0, (player_val - mean_val) / (target - mean_val)))
                            else:
                                normalized_score = 0.5
                            profile_score += normalized_score
                            profile_matches += 1
                
                if profile_matches > 0:
                    avg_profile_score = profile_score / profile_matches
                    
                    prediction_data = {
                        'name': player_name,
                        'position': position,
                        'adp_2025': adp_2025,
                        'profile_score': avg_profile_score,
                        'profile_matches': profile_matches
                    }
                    
                    # Categorize predictions
                    if avg_profile_score >= 0.8 and adp_2025 > 50:  # High profile score, low ADP
                        predictions['sleepers'].append(prediction_data)
                    elif avg_profile_score >= 0.7 and adp_2025 > 25:  # Good profile score, reasonable ADP
                        predictions['breakouts'].append(prediction_data)
                    elif avg_profile_score <= 0.3 and adp_2025 <= 50:  # Low profile score, high ADP
                        predictions['overvalued'].append(prediction_data)
                    elif avg_profile_score >= 0.6 and 25 <= adp_2025 <= 100:  # Solid profile, good value
                        predictions['solid_values'].append(prediction_data)
        
        # Output predictions
        analysis_text.append(f"\n SLEEPER CANDIDATES (High upside, low ADP):")
        analysis_text.append("-" * 50)
        sleepers = sorted(predictions['sleepers'], key=lambda x: x['profile_score'], reverse=True)
        for player in sleepers[:10]:
            analysis_text.append(f"  {player['name']} ({player['position']}) - ADP: {player['adp_2025']:.1f} | Profile Score: {player['profile_score']:.2f}")
        
        analysis_text.append(f"\n BREAKOUT CANDIDATES (Strong metrics, reasonable ADP):")
        analysis_text.append("-" * 50)
        breakouts = sorted(predictions['breakouts'], key=lambda x: x['profile_score'], reverse=True)
        for player in breakouts[:10]:
            analysis_text.append(f"  {player['name']} ({player['position']}) - ADP: {player['adp_2025']:.1f} | Profile Score: {player['profile_score']:.2f}")
        
        analysis_text.append(f"\n  OVERVALUED PLAYERS (Poor metrics vs high ADP):")
        analysis_text.append("-" * 50)
        overvalued = sorted(predictions['overvalued'], key=lambda x: x['adp_2025'])
        for player in overvalued[:10]:
            analysis_text.append(f"  {player['name']} ({player['position']}) - ADP: {player['adp_2025']:.1f} | Profile Score: {player['profile_score']:.2f}")
        
        analysis_text.append(f"\n✅ SOLID VALUE PICKS (Good metrics, fair ADP):")
        analysis_text.append("-" * 50)
        solid = sorted(predictions['solid_values'], key=lambda x: x['profile_score'], reverse=True)
        for player in solid[:10]:
            analysis_text.append(f"  {player['name']} ({player['position']}) - ADP: {player['adp_2025']:.1f} | Profile Score: {player['profile_score']:.2f}")
        
        return "\n".join(analysis_text)
    
    def analyze_roneill19_tendencies(self) -> str:
        """Deep dive analysis of roneill19's draft patterns and tendencies for self-improvement"""
        print("\n" + "="*60)
        print("roneill19 PERSONAL DRAFT ANALYSIS")
        print("="*60)
        
        target_manager = "roneill19"
        roneill_data = {
            'total_picks': 0,
            'positions': defaultdict(int),
            'rounds': defaultdict(int),
            'position_by_round': defaultdict(list),
            'yearly_performance': {},
            'reaches': [],
            'values': [],
            'sleeper_hits': [],
            'busts': [],
            'advanced_metrics': {},
            'draft_positions': [],
            'consistency_patterns': defaultdict(int)
        }
        
        # Collect all roneill19's draft data
        for year in self.years:
            if year not in self.draft_data:
                continue
                
            df = self.draft_data[year]
            year_picks = []
            
            for _, pick in df.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                if manager.lower() == target_manager.lower():
                    position = pick.get('Position', pick.get('position', 'Unknown'))
                    round_num = pick.get('Round', pick.get('round', 0))
                    pick_no = pick.get('Pick_No', pick.get('pick_no', 0))
                    player = pick.get('Player', pick.get('player', 'Unknown'))
                    
                    roneill_data['total_picks'] += 1
                    roneill_data['positions'][position] += 1
                    roneill_data['rounds'][round_num] += 1
                    roneill_data['position_by_round'][round_num].append(position)
                    roneill_data['draft_positions'].append(pick_no)
                    
                    year_picks.append({
                        'player': player,
                        'position': position,
                        'round': round_num,
                        'pick': pick_no
                    })
                    
                    # Track consistency patterns
                    pattern_key = f"{position}_R{round_num}"
                    roneill_data['consistency_patterns'][pattern_key] += 1
            
            if year_picks:
                roneill_data['yearly_performance'][year] = year_picks
        
        if roneill_data['total_picks'] == 0:
            return f"No draft data found for {target_manager}"
        
        analysis_text = []
        analysis_text.append(f"👤 PERSONAL ANALYSIS FOR {target_manager.upper()}")
        analysis_text.append("="*50)
        
        # Overall draft profile
        years_active = len(roneill_data['yearly_performance'])
        avg_picks_per_year = roneill_data['total_picks'] / years_active if years_active > 0 else 0
        
        analysis_text.append(f"\n DRAFT PROFILE:")
        analysis_text.append(f"• Years Active: {years_active}")
        analysis_text.append(f"• Total Picks: {roneill_data['total_picks']}")
        analysis_text.append(f"• Average Picks/Year: {avg_picks_per_year:.1f}")
        
        # Position preferences
        analysis_text.append(f"\n POSITION PREFERENCES:")
        total_picks = roneill_data['total_picks']
        for position, count in sorted(roneill_data['positions'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_picks) * 100
            analysis_text.append(f"• {position}: {count} picks ({percentage:.1f}%)")
        
        # Round-by-round analysis
        analysis_text.append(f"\n*** ROUND-BY-ROUND TENDENCIES:")
        for round_num in sorted(roneill_data['rounds'].keys())[:10]:  # First 10 rounds
            positions = roneill_data['position_by_round'][round_num]
            if positions:
                position_counts = defaultdict(int)
                for pos in positions:
                    position_counts[pos] += 1
                
                most_common = max(position_counts.items(), key=lambda x: x[1])
                analysis_text.append(f"• Round {round_num}: {most_common[0]} ({most_common[1]}/{len(positions)} times)")
        
        # Identify patterns and tendencies
        analysis_text.append(f"\n DRAFT PATTERN ANALYSIS:")
        
        # Early QB tendency
        early_qb_count = sum(1 for pos in roneill_data['position_by_round'][1] + 
                            roneill_data['position_by_round'][2] + 
                            roneill_data['position_by_round'][3] + 
                            roneill_data['position_by_round'][4] if pos == 'QB')
        total_qb = roneill_data['positions']['QB']
        
        if total_qb > 0:
            early_qb_rate = early_qb_count / total_qb
            if early_qb_rate > 0.5:
                analysis_text.append("•   TENDENCY: Drafts QB early (Rounds 1-4)")
                analysis_text.append("  💡 IMPROVEMENT: Consider waiting on QB unless elite tier available")
            else:
                analysis_text.append("• ✅ STRENGTH: Patient with QB selection")
        
        # RB strategy analysis
        rb_round1 = len([pos for pos in roneill_data['position_by_round'][1] if pos == 'RB'])
        rb_early = len([pos for pos in roneill_data['position_by_round'][1] + 
                       roneill_data['position_by_round'][2] if pos == 'RB'])
        
        analysis_text.append(f"• RB Strategy: {rb_round1} Round 1 RBs, {rb_early} in first 2 rounds")
        if rb_early / years_active < 1.5:
            analysis_text.append("  💡 IMPROVEMENT: Consider prioritizing RB earlier - position typically drafted heavily early")
        
        # Consistency analysis
        analysis_text.append(f"\n🔄 CONSISTENCY PATTERNS:")
        consistent_patterns = [(pattern, count) for pattern, count in roneill_data['consistency_patterns'].items() 
                              if count >= max(2, years_active * 0.5)]
        
        if consistent_patterns:
            analysis_text.append("Most consistent draft patterns:")
            for pattern, count in sorted(consistent_patterns, key=lambda x: x[1], reverse=True)[:5]:
                position, round_info = pattern.split('_R')
                analysis_text.append(f"• {position} in Round {round_info}: {count} times")
        else:
            analysis_text.append("• High variability - no strong consistent patterns")
            analysis_text.append("  💡 IMPROVEMENT: Consider developing more consistent positional strategies")
        
        # Compare to league averages
        analysis_text.append(f"\n COMPARISON TO LEAGUE:")
        
        # Calculate league averages for comparison
        league_position_avg = self._calculate_league_position_averages()
        
        for position in ['RB', 'WR', 'QB', 'TE']:
            if position in roneill_data['positions'] and position in league_position_avg:
                roneill_rate = roneill_data['positions'][position] / roneill_data['total_picks']
                league_rate = league_position_avg[position]
                
                if roneill_rate > league_rate * 1.2:
                    analysis_text.append(f"• {position}: Above average usage ({roneill_rate:.1%} vs {league_rate:.1%})")
                elif roneill_rate < league_rate * 0.8:
                    analysis_text.append(f"• {position}: Below average usage ({roneill_rate:.1%} vs {league_rate:.1%})")
                else:
                    analysis_text.append(f"• {position}: Average usage ({roneill_rate:.1%} vs {league_rate:.1%})")
        
        # Strategic recommendations
        analysis_text.append(f"\n💡 PERSONALIZED RECOMMENDATIONS FOR 2025:")
        analysis_text.append("="*50)
        
        # Based on analysis, provide specific recommendations
        recommendations = []
        
        if early_qb_rate > 0.5:
            recommendations.append(" Wait on QB: Your early QB picks haven't differentiated you - consider waiting")
        
        if rb_early / years_active < 1.5:
            recommendations.append("🏃 Prioritize RB: League drafts RBs heavily early - don't get left behind")
        
        if roneill_data['positions'].get('TE', 0) / total_picks > 0.15:
            recommendations.append(" TE Strategy: You draft TEs frequently - target elite TEs or wait completely")
        
        # Add advanced metric recommendations if available
        recommendations.append("*** Focus on Advanced Metrics: Target players with strong efficiency metrics")
        recommendations.append("🎭 Study League Tendencies: Use draft flow analysis to anticipate runs")
        recommendations.append("💰 Value Hunting: Look for ADP mismatches based on advanced metrics")
        
        for rec in recommendations:
            analysis_text.append(f"• {rec}")
        
        # Strengths and weaknesses
        analysis_text.append(f"\n*** STRENGTHS:")
        strengths = []
        
        if not consistent_patterns:
            strengths.append("• Flexible drafting approach - adapts to draft flow")
        if early_qb_rate <= 0.5:
            strengths.append("• Patient QB selection")
        if len(roneill_data['yearly_performance']) >= 3:
            strengths.append("• Consistent league participation")
        
        for strength in strengths:
            analysis_text.append(strength)
        
        analysis_text.append(f"\n  AREAS FOR IMPROVEMENT:")
        improvements = []
        
        if rb_early / years_active < 1.5:
            improvements.append("• Earlier RB selection to avoid scarcity")
        if early_qb_rate > 0.5:
            improvements.append("• More patient QB approach")
        if not consistent_patterns:
            improvements.append("• Develop more consistent positional strategies")
        
        improvements.append("• Use advanced metrics to identify undervalued players")
        improvements.append("• Monitor league draft flow patterns for better timing")
        
        for improvement in improvements:
            analysis_text.append(improvement)
        
        return "\n".join(analysis_text)
    
    def _calculate_league_position_averages(self) -> Dict[str, float]:
        """Calculate league-wide position drafting averages"""
        position_totals = defaultdict(int)
        total_picks = 0
        
        for year in self.years:
            if year not in self.draft_data:
                continue
                
            df = self.draft_data[year]
            
            for _, pick in df.iterrows():
                position = pick.get('Position', pick.get('position', 'Unknown'))
                if position != 'Unknown':
                    position_totals[position] += 1
                    total_picks += 1
        
        if total_picks == 0:
            return {}
        
        return {position: count / total_picks for position, count in position_totals.items()}
    
    
    def analyze_adp_value_mismatches(self) -> str:
        """Identify players whose ADP doesn't match their advanced metrics profile"""
        print("\n" + "="*60)
        print("ADP vs ADVANCED METRICS VALUE ANALYSIS")
        print("="*60)
        
        analysis_text = []
        analysis_text.append("💎 ADP VALUE MISMATCH ANALYSIS")
        analysis_text.append("="*50)
        
        # Get latest data
        latest_year = max(self.years) if self.years else None
        if not latest_year:
            return "No data available for ADP analysis"
        
        # Collect player data with both advanced metrics and ADP
        player_profiles = []
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            if (latest_year not in self.advanced_data or 
                position not in self.advanced_data[latest_year] or
                latest_year not in self.adp_data):
                continue
            
            advanced_df = self.advanced_data[latest_year][position]
            adp_df = self.adp_data[latest_year]
            
            if advanced_df.empty or adp_df.empty:
                continue
            
            advanced_player_col = self.get_player_column(advanced_df)
            adp_player_col = self.get_player_column(adp_df)
            adp_col = self.get_adp_column(adp_df)
            
            if not all([advanced_player_col, adp_player_col, adp_col]):
                continue
            
            # Create ADP lookup
            adp_lookup = {}
            for _, row in adp_df.iterrows():
                player_name = self.clean_player_name(row[adp_player_col])
                adp_value = self.safe_float_conversion(row[adp_col])
                if adp_value > 0:
                    adp_lookup[player_name] = adp_value
            
            # Analyze advanced metrics vs ADP
            for _, player in advanced_df.iterrows():
                player_name = self.clean_player_name(player[advanced_player_col])
                
                if player_name in adp_lookup:
                    advanced_score = self.calculate_advanced_performance_score(player, position)
                    adp = adp_lookup[player_name]
                    
                    # Calculate value mismatch score
                    # Lower ADP (earlier pick) = lower number, higher score = better metrics
                    expected_adp_range = self._get_expected_adp_range(advanced_score, position)
                    
                    player_profiles.append({
                        'name': player_name,
                        'position': position,
                        'advanced_score': advanced_score,
                        'actual_adp': adp,
                        'expected_adp_range': expected_adp_range,
                        'value_type': self._classify_value_type(adp, expected_adp_range, advanced_score)
                    })
        
        # Sort and categorize players
        undervalued = [p for p in player_profiles if p['value_type'] == 'undervalued']
        overvalued = [p for p in player_profiles if p['value_type'] == 'overvalued']
        fair_value = [p for p in player_profiles if p['value_type'] == 'fair']
        
        # Sort by value opportunity
        undervalued.sort(key=lambda x: (x['advanced_score'], -x['actual_adp']), reverse=True)
        overvalued.sort(key=lambda x: (x['actual_adp'], -x['advanced_score']))
        
        # Report undervalued players
        if undervalued:
            analysis_text.append("\n💎 UNDERVALUED BY ADP (Target These):")
            analysis_text.append("-" * 40)
            
            for player in undervalued[:15]:
                expected_range = player['expected_adp_range']
                analysis_text.append(f"• {player['name']} ({player['position']})")
                analysis_text.append(f"  ADP: {player['actual_adp']:.0f} | Expected: {expected_range[0]:.0f}-{expected_range[1]:.0f} | Score: {player['advanced_score']:.2f}")
        
        # Report overvalued players
        if overvalued:
            analysis_text.append("\n  OVERVALUED BY ADP (Consider Fading):")
            analysis_text.append("-" * 40)
            
            for player in overvalued[:10]:
                expected_range = player['expected_adp_range']
                analysis_text.append(f"• {player['name']} ({player['position']})")
                analysis_text.append(f"  ADP: {player['actual_adp']:.0f} | Expected: {expected_range[0]:.0f}-{expected_range[1]:.0f} | Score: {player['advanced_score']:.2f}")
        
        # Strategic recommendations
        analysis_text.append(f"\n ADP VALUE STRATEGY:")
        analysis_text.append("="*40)
        
        analysis_text.append("DRAFT APPROACH:")
        analysis_text.append("• Target undervalued players 1-2 rounds after their expected range")
        analysis_text.append("• Avoid overvalued players unless you believe in narrative changes")
        analysis_text.append("• Use advanced metrics to break ties between similar ADP players")
        analysis_text.append("• Consider positional scarcity when evaluating value")
        
        analysis_text.append("\nTRADE STRATEGY:")
        analysis_text.append("• Trade for undervalued players before their metrics are widely recognized")
        analysis_text.append("• Sell overvalued players at peak narrative value")
        analysis_text.append("• Use metric mismatches to identify buy-low/sell-high opportunities")
        
        return "\n".join(analysis_text)
    
    def _get_expected_adp_range(self, advanced_score: float, position: str) -> Tuple[float, float]:
        """Get expected ADP range based on advanced metrics score"""
        # These ranges are approximate and position-dependent
        if position == 'QB':
            if advanced_score >= 0.8:
                return (48.0, 72.0)  # QB 4-6
            elif advanced_score >= 0.6:
                return (72.0, 120.0)  # QB 7-10
            else:
                return (120.0, 200.0)  # QB 11+
        
        elif position == 'RB':
            if advanced_score >= 0.8:
                return (6.0, 24.0)   # RB 1-2
            elif advanced_score >= 0.6:
                return (24.0, 60.0)  # RB 3-5
            else:
                return (60.0, 120.0)  # RB 6+
        
        elif position == 'WR':
            if advanced_score >= 0.8:
                return (12.0, 36.0)  # WR 1-3
            elif advanced_score >= 0.6:
                return (36.0, 84.0)  # WR 4-7
            else:
                return (84.0, 150.0)  # WR 8+
        
        elif position == 'TE':
            if advanced_score >= 0.8:
                return (36.0, 60.0)  # TE 1-2
            elif advanced_score >= 0.6:
                return (60.0, 120.0)  # TE 3-5
            else:
                return (120.0, 200.0)  # TE 6+
        
        return (100.0, 200.0)  # Default
    
    def _classify_value_type(self, actual_adp: float, expected_range: Tuple[float, float], score: float) -> str:
        """Classify player as undervalued, overvalued, or fair value"""
        expected_min, expected_max = expected_range
        
        # Undervalued: ADP is significantly later than expected (worse ADP number = later pick)
        if actual_adp > expected_max * 1.3 and score >= 0.5:
            return 'undervalued'
        
        # Overvalued: ADP is significantly earlier than expected (better ADP number = earlier pick)
        elif actual_adp < expected_min * 0.7 and score < 0.6:
            return 'overvalued'
        
        else:
            return 'fair'
    
    def _get_injury_profile(self, player: str, position: str) -> Dict[str, Any]:
        """Get injury profile for a player"""
        if position not in self.injury_data:
            print(f"WARNING: No injury data for position {position}")
            return {
                'durability_score': 5.0,
                'projected_games_missed': 2.0,
                'injury_risk_level': 'Medium',
                'injury_risk': 0.5
            }
        
        injury_df = self.injury_data[position]
        
        # Clean player name for better matching
        clean_player = self.clean_player_name_for_matching(player)
        
        # Try multiple matching strategies
        player_injury = None
        
        # Strategy 1: Exact match on cleaned names
        injury_df_clean = injury_df.copy()
        injury_df_clean['Player_Clean'] = injury_df_clean['Player Name'].apply(self.clean_player_name_for_matching)
        player_injury = injury_df_clean[injury_df_clean['Player_Clean'] == clean_player]
        
        # Strategy 2: Contains first name match
        if player_injury.empty and len(clean_player.split()) > 0:
            first_name = clean_player.split()[0]
            last_name = clean_player.split()[-1] if len(clean_player.split()) > 1 else ''
            player_injury = injury_df[injury_df['Player Name'].str.contains(first_name, case=False, na=False) & 
                                    injury_df['Player Name'].str.contains(last_name, case=False, na=False)]
        
        # Strategy 3: Just last name match
        if player_injury.empty and len(clean_player.split()) > 1:
            last_name = clean_player.split()[-1]
            player_injury = injury_df[injury_df['Player Name'].str.contains(last_name, case=False, na=False)]
        
        if player_injury.empty:
            print(f"WARNING: No injury data found for {player} ({position}). Available players: {list(injury_df['Player Name'].head(3))}")
            return {
                'durability_score': 5.0,
                'projected_games_missed': 2.0,
                'injury_risk_level': 'Medium',
                'injury_risk': 0.5
            }
        
        row = player_injury.iloc[0]
        print(f"Found injury data for {player} -> {row['Player Name']}")
        
        # Extract injury data with correct column names
        durability = self.safe_float_conversion(row.get('Durability', 5))
        projected_missed = self.safe_float_conversion(row.get('Projected Games Missed', 2))
        risk_level = str(row.get('Injury Risk', 'Medium Risk')).replace(' Risk', '')
        
        # Convert risk level to numeric
        risk_mapping = {
            'Very Low': 0.1,
            'Low': 0.25,
            'Medium': 0.5,
            'High': 0.75,
            'Very High': 0.9
        }
        injury_risk = risk_mapping.get(risk_level, 0.5)
        
        return {
            'durability_score': float(durability),
            'projected_games_missed': float(projected_missed),
            'injury_risk_level': risk_level,
            'injury_risk': injury_risk
        }
    
    def _generate_ml_predictions(self, player: str, position: str, features: np.ndarray, debug_mode: bool = False) -> Dict[str, float]:
        """Generate ML predictions for a player"""
        if position not in self.ml_models or position not in self.scalers:
            self._log_missing_data(player, position, f"ml_models_and_scalers_for_{position}", 
                                 [f"ml_models[{position}]", f"scalers[{position}]"], "ML_TRAINING_DATA")
            # Return None values to force proper error handling downstream
            return {
                'breakout_probability': None,
                'consistency_score': None,
                'positional_value': None,
                'model_confidence': None
            }
        
        models = self.ml_models[position]
        scaler = self.scalers[position]
        confidence = self.model_confidence.get(position, {})
        
        if debug_mode:
            print(f"DEBUG ML {player} ({position}):")
            print(f"  Raw features: {features}")
            print(f"  Available models: {list(models.keys())}")
            print(f"  Model confidence: {confidence}")
        
        try:
            # Ensure features is a numpy array and scale features
            features_array = np.array(features, dtype=float)
            features_scaled = scaler.transform(features_array.reshape(1, -1))
            if debug_mode:
                print(f"  Scaled features: {features_scaled[0]}")
            
            predictions = {}
            
            # Breakout probability (now using regressor)
            if 'breakout' in models:
                breakout_prob = models['breakout'].predict(features_scaled)[0]
                # Clamp to 0-1 range for probability
                breakout_prob = max(0, min(1, float(breakout_prob)))
                predictions['breakout_probability'] = breakout_prob
                if debug_mode:
                    print(f"  Breakout prediction: {breakout_prob:.3f}")
            else:
                self._log_missing_data(player, position, f"breakout_model_for_{position}", 
                                     [f"models['breakout']"], "ML_MODEL_TRAINING")
                predictions['breakout_probability'] = None
            
            # Consistency score
            if 'consistency' in models:
                consistency = models['consistency'].predict(features_scaled)[0]
                predictions['consistency_score'] = max(0, min(10, float(consistency)))
                if debug_mode:
                    print(f"  Consistency prediction: {consistency:.3f} -> {predictions['consistency_score']:.3f}")
            else:
                self._log_missing_data(player, position, f"consistency_model_for_{position}", 
                                     [f"models['consistency']"], "ML_MODEL_TRAINING")
                predictions['consistency_score'] = None
            
            # Positional value  
            if 'positional_value' in models:
                pos_value = models['positional_value'].predict(features_scaled)[0]
                predictions['positional_value'] = max(0, min(1, float(pos_value)))
                if debug_mode:
                    print(f"  Positional value prediction: {pos_value:.3f} -> {predictions['positional_value']:.3f}")
            else:
                self._log_missing_data(player, position, f"positional_value_model_for_{position}", 
                                     [f"models['positional_value']"], "ML_MODEL_TRAINING")
                predictions['positional_value'] = None
            
            # Overall model confidence
            avg_confidence = np.mean(list(confidence.values())) if confidence else 0.3
            predictions['model_confidence'] = max(0, min(1, avg_confidence))
            
            if debug_mode:
                print(f"  Final ML predictions: {predictions}")
            
            return predictions
            
        except Exception as e:
            self._log_missing_data(player, position, f"ml_prediction_generation_error", 
                                 [f"Exception: {str(e)}"], "ML_PREDICTION_PROCESS")
            return {
                'breakout_probability': None,
                'consistency_score': None,
                'positional_value': None,
                'model_confidence': None
            }
    
    def generate_player_profiles(self) -> List[Dict[str, Any]]:
        """Generate comprehensive player profiles with ML predictions"""
        print("Generating ML-enhanced player profiles...")
        
        profiles = []
        
        # Player limits by position
        player_limits = {'QB': 20, 'RB': 80, 'WR': 80, 'TE': 20}
        
        # Use most recent year for player selection
        current_year = max(self.years)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            print(f"Processing {position} players...")
            
            if current_year not in self.advanced_data or position not in self.advanced_data[current_year]:
                print(f"No advanced data for {position} in {current_year}")
                continue
            
            advanced_df = self.advanced_data[current_year][position]
            limit = player_limits.get(position, 50)
            
            # Get top players (first N rows)
            top_players = advanced_df.head(limit)
            
            for _, player_row in top_players.iterrows():
                # Find player name column
                player_name = None
                for col in ['PLAYER', 'Player', 'NAME', 'Name']:
                    if col in player_row.index and not pd.isna(player_row[col]):
                        player_name = player_row[col]
                        break
                
                if player_name is None:
                    continue
                    
                team = player_row.get('TM', player_row.get('TEAM', 'N/A'))
                
                # Get historical performance
                historical_perf = self._calculate_historical_performance(player_name, position)
                
                # Get injury profile
                injury_profile = self._get_injury_profile(player_name, position)
                
                # Prepare features for ML prediction
                features_df = self._prepare_ml_features(position, current_year)
                
                # Generate ML predictions using trained models
                if features_df is not None:
                    player_features = features_df[features_df['PLAYER'].apply(self.clean_player_name_for_matching) == 
                                                 self.clean_player_name_for_matching(player_name)]
                    if not player_features.empty:
                        feature_vector = player_features.iloc[0].drop('PLAYER').values
                        feature_vector = np.array([self.safe_float_conversion(val) for val in feature_vector], dtype=float)
                        if not np.any(np.isnan(feature_vector)):
                            ml_predictions = self._generate_ml_predictions(player_name, position, feature_vector, debug_mode=True)
                        else:
                            self._log_missing_data(player_name, position, "clean_ml_features", 
                                                 ["NaN values in feature vector"], "ML_FEATURE_PREPARATION")
                            ml_predictions = {'breakout_probability': None, 'consistency_score': None, 'positional_value': None, 'model_confidence': None}
                    else:
                        self._log_missing_data(player_name, position, "ml_features_found", 
                                             ["feature vector generation"], "ML_FEATURE_PREPARATION") 
                        ml_predictions = {'breakout_probability': None, 'consistency_score': None, 'positional_value': None, 'model_confidence': None}
                else:
                    self._log_missing_data(player_name, position, f"feature_data_for_{position}", 
                                         ["features_df"], "ML_FEATURE_DATA")
                    ml_predictions = {'breakout_probability': None, 'consistency_score': None, 'positional_value': None, 'model_confidence': None}
                
                if features_df is not None:
                    player_features = features_df[features_df['PLAYER'].str.contains(player_name, case=False, na=False)]
                    if not player_features.empty:
                        feature_vector = player_features.iloc[0].drop('PLAYER').values
                        if not np.any(np.isnan(feature_vector)):
                            ml_predictions = self._generate_ml_predictions(player_name, position, feature_vector)
                
                # Get advanced metrics summary
                key_metrics = self.key_metrics.get(position, [])
                advanced_metrics = {}
                for metric in key_metrics:
                    if metric in player_row.index and not pd.isna(player_row[metric]):
                        advanced_metrics[metric] = player_row[metric]
                
                # Calculate ADP vs finish history
                adp_history = {}
                for year in self.years:
                    if year in self.adp_data and year in self.results_data:
                        adp_df = self.adp_data[year]
                        results_df = self.results_data[year]
                        
                        adp_player_col = 'Player' if 'Player' in adp_df.columns else 'PLAYER'
                        results_player_col = 'Player' if 'Player' in results_df.columns else 'PLAYER'
                        
                        # Skip if player name is not valid
                        if not isinstance(player_name, str) or pd.isna(player_name) or len(player_name.strip()) == 0:
                            continue
                            
                        adp_match = adp_df[adp_df[adp_player_col].str.contains(player_name, case=False, na=False)]
                        results_match = results_df[results_df[results_player_col].str.contains(player_name, case=False, na=False)]
                        
                        if not adp_match.empty and not results_match.empty:
                            adp_rank = adp_match.iloc[0].get('ADP', adp_match.iloc[0].get('AVG', np.nan))
                            finish_rank = results_match.iloc[0].get('RK', results_match.iloc[0].get('#', np.nan))
                            
                            if not pd.isna(adp_rank) and not pd.isna(finish_rank):
                                adp_history[str(year)] = float(finish_rank) - float(adp_rank)
                
                # Calculate overall profile score using universal system
                overall_scores = self.calculate_overall_profile_score(
                    historical_perf, advanced_metrics, ml_predictions, injury_profile, 
                    position=position, player_row=player_row
                )
                
                # Calculate Half PPR projection if we have player stats
                half_ppr_projection = 0.0
                try:
                    # Try to get actual fantasy points for projection
                    if advanced_metrics:
                        player_name_for_points = advanced_metrics.get('Player', player_name)
                        half_ppr_projection = self.get_actual_fantasy_points(player_name_for_points, position, 2024) or 0.0
                except:
                    half_ppr_projection = 0.0
                
                # Create player profile
                profile = {
                    'player_name': player_name,
                    'position': position,
                    'team': team,
                    'overall_profile_score': overall_scores['overall_score'],
                    'star_rating': overall_scores['star_rating'],
                    'component_scores': {
                        'historical_score': overall_scores['historical_score'],
                        'advanced_score': overall_scores['advanced_score'],
                        'ml_score': overall_scores['ml_score'],
                        'injury_score': overall_scores['injury_score']
                    },
                    'historical_performance': {
                        'adp_vs_finish_history': adp_history,
                        'avg_adp_differential': historical_perf['avg_adp_differential'],
                        'fantasy_points_trend': 'improving' if historical_perf['avg_fantasy_points'] > 200 else 'stable',
                        'years_of_data': historical_perf['years_of_data']
                    },
                    'advanced_metrics_summary': {
                        str(current_year): advanced_metrics,
                        'efficiency_scores': {'universal_score': overall_scores['advanced_score']},
                        'key_strengths': list(advanced_metrics.keys())[:3] if advanced_metrics else []
                    },
                    'ml_predictions': {
                        'breakout_probability': ml_predictions['breakout_probability'],
                        'consistency_score': ml_predictions['consistency_score'],
                        'positional_value': ml_predictions['positional_value'],
                        'injury_risk': injury_profile['injury_risk'],
                        'model_confidence': ml_predictions['model_confidence']
                    },
                    'injury_profile': {
                        'durability_score': injury_profile['durability_score'],
                        'projected_games_missed': injury_profile['projected_games_missed'],
                        'injury_risk_level': injury_profile['injury_risk_level']
                    },
                    'half_ppr_projection': {
                        'projected_points': round(half_ppr_projection, 1),
                        'scoring_system': 'Half PPR',
                        'positional_value': 'High' if ml_predictions['positional_value'] > 0.7 else 'Medium' if ml_predictions['positional_value'] > 0.4 else 'Low'
                    }
                }
                
                profiles.append(profile)
        
        return profiles
    

    
    def save_player_profiles(self, profiles: List[Dict[str, Any]], filename: str = "player_profiles.txt") -> str:
        """Save player profiles in readable text format"""
        content = []
        content.append("FANTASY FOOTBALL PLAYER PROFILES 2025")
        content.append("=" * 60)
        content.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"Total Players: {len(profiles)}")
        content.append("")
        
        # Group by position
        by_position = {}
        for profile in profiles:
            pos = profile['position']
            if pos not in by_position:
                by_position[pos] = []
            by_position[pos].append(profile)
        
        # Sort each position by overall profile score
        for pos in by_position:
            by_position[pos].sort(key=lambda x: x['overall_profile_score'], reverse=True)
        
        # Generate text profiles
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position not in by_position:
                continue
                
            content.append(f"\n{'='*20} {position} PLAYERS {'='*20}")
            content.append("")
            
            for profile in by_position[position]:
                # Generate star display
                stars = "★" * profile['star_rating'] + "☆" * (5 - profile['star_rating'])
                
                content.append("=== PLAYER PROFILE ===")
                content.append(f"Name: {profile['player_name']}")
                content.append(f"Position: {profile['position']} | Team: {profile['team']}")
                content.append(f"OVERALL PROFILE SCORE: {profile['overall_profile_score']:.0f}/100 {stars}")
                content.append("")
                
                # Component Scores
                content.append("COMPONENT SCORES:")
                comp = profile['component_scores']
                content.append(f"- Historical Performance: {comp['historical_score']:.1f}/10 (25% weight)")
                content.append(f"- Advanced Metrics: {comp['advanced_score']:.1f}/10 (30% weight)")
                content.append(f"- ML Predictions: {comp['ml_score']:.1f}/10 (30% weight)")
                content.append(f"- Injury Profile: {comp['injury_score']:.1f}/10 (15% weight)")
                content.append("")
                
                # Historical Performance
                content.append("HISTORICAL PERFORMANCE:")
                hist = profile['historical_performance']
                if hist['adp_vs_finish_history']:
                    adp_str = ", ".join([f"{year}: {diff:+.0f}" for year, diff in hist['adp_vs_finish_history'].items()])
                    content.append(f"- ADP vs Finish: {adp_str}")
                content.append(f"- Average Differential: {abs(hist['avg_adp_differential']):.1f} spots {'better than' if hist['avg_adp_differential'] < 0 else 'worse than'} ADP")
                content.append(f"- Trend: {hist['fantasy_points_trend'].title()}")
                content.append("")
                
                # Advanced Metrics
                content.append("ADVANCED METRICS (2024):")
                metrics = profile['advanced_metrics_summary']
                current_metrics = list(metrics.values())[0] if metrics else {}
                if current_metrics:
                    metric_strs = [f"{k}: {v}" for k, v in current_metrics.items() if isinstance(v, (int, float))]
                    content.append(f"- {' | '.join(metric_strs[:3])}")
                content.append(f"- Composite Efficiency: {metrics['efficiency_scores']['composite_score']:.1f}/10")
                if metrics['key_strengths']:
                    content.append(f"- Key Strengths: {', '.join(metrics['key_strengths'][:3])}")
                content.append("")
                
                # ML Predictions
                content.append("ML PREDICTIONS (2025):")
                ml = profile['ml_predictions']
                content.append(f"- Breakout Probability: {ml['breakout_probability']*100:.0f}%")
                content.append(f"- Consistency Score: {ml['consistency_score']:.1f}/10")
                content.append(f"- Positional Value: {ml['positional_value']*100:.0f}th percentile")
                content.append(f"- Injury Risk: {ml['injury_risk']*100:.0f}% ({'Low' if ml['injury_risk'] < 0.3 else 'Medium' if ml['injury_risk'] < 0.6 else 'High'})")
                content.append(f"- Model Confidence: {ml['model_confidence']*100:.0f}%")
                content.append("")
                
                # Half PPR Projection
                content.append("HALF PPR PROJECTION (2025):")
                ppr = profile['half_ppr_projection']
                content.append(f"- Projected Points: {ppr['projected_points']}")
                content.append(f"- Scoring System: {ppr['scoring_system']}")
                content.append(f"- Value Score: {ppr['positional_value']}")
                content.append("")
                
                # Injury Profile
                content.append("INJURY PROFILE:")
                inj = profile['injury_profile']
                content.append(f"- Durability Score: {inj['durability_score']:.1f}/10")
                content.append(f"- Projected Games Missed: {inj['projected_games_missed']:.1f}")
                content.append(f"- Risk Level: {inj['injury_risk_level']}")
                content.append("")
                content.append("-" * 50)
                content.append("")
        
        # Add ranking summary section
        content.append("\n" + "="*60)
        content.append("OVERALL RANKINGS BY PROFILE SCORE")
        content.append("="*60)
        content.append("")
        
        # Generate rankings for each position
        for position in ['QB', 'RB', 'WR', 'TE']:
            if position not in by_position:
                continue
                
            # Get player limits for each position
            player_limits = {'QB': 20, 'RB': 80, 'WR': 80, 'TE': 20}
            limit = player_limits.get(position, 50)
            
            content.append(f"{position} RANKINGS (Top {min(len(by_position[position]), limit)}):")
            content.append("-" * 40)
            
            for i, profile in enumerate(by_position[position][:limit], 1):
                stars = "★" * profile['star_rating'] + "☆" * (5 - profile['star_rating'])
                content.append(f"{i:2d}. {profile['player_name']:<20} ({profile['overall_profile_score']:.0f}/100) {stars}")
            
            content.append("")
        
        # Add summary insights
        content.append("="*60)
        content.append("RANKING INSIGHTS")
        content.append("="*60)
        content.append("• Rankings based on Overall Profile Score (0-100)")
        content.append("• Score Components: Historical (25%) + Advanced Metrics (30%) + ML Predictions (30%) + Injury (15%)")
        content.append("• Star Rating: ★★★★★ = 80-100, ★★★★☆ = 60-79, ★★★☆☆ = 40-59, ★★☆☆☆ = 20-39, ★☆☆☆☆ = 0-19")
        content.append("• Half PPR scoring system used for all projections")
        content.append("• Focus on players with 4+ stars for highest confidence targets")
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
        
        return filename
    
    def save_profiles_json(self, profiles: List[Dict[str, Any]], filename: str = "player_profiles.json") -> str:
        """Save player profiles in JSON format for Custom GPT integration"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, indent=2, default=str)
        return filename
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report"""
        print("Generating comprehensive fantasy football analysis report...")
        
        report = []
        report.append("FANTASY FOOTBALL LEAGUE ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Analysis Period: {min(self.years)} - {max(self.years)}")
        report.append(f"Total Years Analyzed: {len(self.years)}")
        report.append(f"Draft Data Available: {list(self.draft_data.keys())}")
        report.append("")
        
        # League analysis (only if enabled)
        if self.enable_league_analysis:
            print("Running league mate analysis...")
            
            # League mate analysis
            league_analysis = self.analyze_league_mate_tendencies()
            if league_analysis:
                report.append(league_analysis)
            
            report.append("\n\n")
            
            # Reach and value analysis
            reach_analysis = self.analyze_draft_reaches_and_values()
            if reach_analysis:
                report.append(reach_analysis)
            
            report.append("\n\n")
            
            # League draft flow analysis
            draft_flow_analysis = self.analyze_league_draft_flow()
            if draft_flow_analysis:
                report.append(draft_flow_analysis)
            
            report.append("\n\n")
        else:
            report.append("League analysis disabled (enable with enable_league_analysis=True)")
            report.append("\n\n")
        
        # Personal analysis (only if league analysis enabled)
        if self.enable_league_analysis:
            roneill_analysis = self.analyze_roneill19_tendencies()
            if roneill_analysis:
                report.append(roneill_analysis)
        
        report.append("\n\n")
        
        
        # Summary insights - LEAGUE FOCUSED ONLY
        report.append(f"\n\n{'='*60}")
        report.append("KEY LEAGUE INSIGHTS & DRAFT STRATEGY")
        report.append("="*60)
        report.append("• Use league mate tendencies to predict draft flow patterns")
        report.append("• Monitor historical reach/value patterns by manager")
        report.append("• Anticipate positional runs based on league history")
        report.append("• Exploit predictable manager drafting behaviors")
        report.append("• Use draft flow analysis to time your picks optimally")
        report.append("• Target value opportunities when managers reach consistently")
        report.append("• Adapt strategy based on draft position and league tendencies")
        report.append("• Focus on roster construction that matches league scoring (Half PPR)")
        
        report.append(f"\n\n{'='*60}")
        report.append("NOTE: PLAYER-SPECIFIC ANALYSIS")
        report.append("="*60)
        report.append("Individual player analysis, rankings, and recommendations")
        report.append("are available in the separate player_profiles.txt file.")
        report.append("This league analysis focuses exclusively on draft flow,")
        report.append("manager tendencies, and strategic insights for your league.")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "fantasy_analysis_report.txt") -> str:
        """Save the analysis report to a file"""
        report = self.generate_comprehensive_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Report saved to {filename}")
        return report
    
    def run_scoring_test(self, num_players: int = 10) -> str:
        """
        Testing/feedback loop: Randomly select players and output detailed scoring breakdown
        
        Args:
            num_players: Number of players to randomly select for testing (default: 10)
            
        Returns:
            String containing detailed scoring breakdown for selected players
        """
        if not self.enable_testing_mode:
            return "Testing mode is disabled. Set enable_testing_mode=True to run tests."
        
        print(f"\n{'='*80}")
        print(f"SCORING SYSTEM TEST - RANDOMLY SELECTED {num_players} PLAYERS")
        print(f"{'='*80}")
        
        # Get all available players from the most recent year
        current_year = max(self.years)
        all_test_players = []
        
        # Collect players from all positions
        for position in ['QB', 'RB', 'WR', 'TE']:
            if current_year in self.advanced_data and position in self.advanced_data[current_year]:
                advanced_df = self.advanced_data[current_year][position]
                
                # Get top 30 players per position to have a good sample
                top_players = advanced_df.head(30)
                
                for _, player_row in top_players.iterrows():
                    # Find player name
                    player_name = None
                    for col in ['PLAYER', 'Player', 'NAME', 'Name']:
                        if col in player_row.index and not pd.isna(player_row[col]):
                            player_name = player_row[col]
                            break
                    
                    if player_name:
                        all_test_players.append({
                            'name': player_name,
                            'position': position,
                            'player_row': player_row
                        })
        
        if len(all_test_players) < num_players:
            return f"Not enough players available for testing. Found {len(all_test_players)}, need {num_players}"
        
        # Randomly select players
        import random
        random.seed()  # Use current time as seed for true randomness
        selected_players = random.sample(all_test_players, num_players)
        
        test_results = []
        test_results.append(f"SELECTED PLAYERS FOR TESTING:")
        test_results.append("-" * 40)
        
        for i, player_data in enumerate(selected_players, 1):
            player_name = player_data['name']
            position = player_data['position']
            player_row = player_data['player_row']
            
            test_results.append(f"\n{'='*60}")
            test_results.append(f"TEST PLAYER #{i}: {player_name} ({position})")
            test_results.append(f"{'='*60}")
            
            # Get advanced metrics for universal scoring
            key_metrics = self.key_metrics.get(position, [])
            advanced_metrics = {}
            for metric in key_metrics:
                if metric in player_row.index and not pd.isna(player_row[metric]):
                    advanced_metrics[metric] = player_row[metric]
            
            # Get historical performance - REQUIRE real data
            historical_perf = self._calculate_historical_performance(player_name, position)
            
            # REQUIRE real historical data - no fallbacks
            if not historical_perf.get('has_historical_data', False):
                self._log_missing_data(player_name, position, "no_historical_data_for_test", 
                                     [f"years_of_data: {historical_perf.get('years_of_data', 0)}"])
                continue  # Skip this player if no real data
            
            # Get injury profile
            injury_profile = self._get_injury_profile(player_name, position)
            
            # Prepare ML features and get predictions
            features_df = self._prepare_ml_features(position, current_year)
            ml_predictions = {'breakout_probability': None, 'consistency_score': None, 'positional_value': None, 'model_confidence': None}
            
            if features_df is not None:
                player_features = features_df[features_df['PLAYER'].apply(self.clean_player_name_for_matching) == 
                                             self.clean_player_name_for_matching(player_name)]
                if not player_features.empty:
                    feature_vector = player_features.iloc[0].drop('PLAYER').values
                    feature_vector = np.array([self.safe_float_conversion(val) for val in feature_vector], dtype=float)
                    if not np.any(np.isnan(feature_vector)):
                        ml_predictions = self._generate_ml_predictions(player_name, position, feature_vector, debug_mode=False)
            
            # Calculate overall scores with detailed breakdown
            test_results.append("\nDETAILED SCORING BREAKDOWN:")
            test_results.append("-" * 40)
            
            # 1. Historical Performance Score - REAL DATA ONLY
            test_results.append(f"\n1. HISTORICAL PERFORMANCE (Weight: 25%):")
            test_results.append(f"   • Data Quality: REAL DATA")
            
            # Show historical scoring structure
            test_results.append("   • Historical Score Structure:")
            test_results.append("     - ADP Differential: 50% of historical score (5/10 points)")
            test_results.append("     - Fantasy Points scaling: 30% of historical score (3/10 points)")
            test_results.append("     - Years of Data reliability: 20% of historical score (2/10 points)")
            test_results.append("     - Total Historical Score: /10 points")
            
            hist_score = 0.0
            if historical_perf['years_of_data'] > 0:
                # ADP differential component with detailed breakdown
                adp_diff = historical_perf['avg_adp_differential']
                test_results.append("   • ADP Differential Assessment:")
                test_results.append(f"     Raw ADP differential: {adp_diff:+.1f} spots")
                test_results.append("     ADP scoring scale:")
                test_results.append("       <=-10 spots (beat ADP by 10+): 10.0/10")
                test_results.append("       -10 to -5 spots: 8.0/10")
                test_results.append("       -5 to 0 spots: 6.0/10")
                test_results.append("       0 to +5 spots: 4.0/10")
                test_results.append("       >+5 spots (underperformed): 2.0/10")
                
                if adp_diff <= -10:  # 10+ spots better than ADP
                    adp_component = 10.0
                elif adp_diff <= -5:
                    adp_component = 8.0
                elif adp_diff <= 0:
                    adp_component = 6.0
                elif adp_diff <= 5:
                    adp_component = 4.0
                else:
                    adp_component = 2.0
                
                # Fantasy points trend component
                fp_raw_score = historical_perf['avg_fantasy_points'] / 25
                fp_component = min(10.0, max(0.0, fp_raw_score))
                
                # Years of data reliability component
                years_component = min(3.0, historical_perf['years_of_data'])
                
                test_results.append("   • Component breakdown with step-by-step calculations:")
                test_results.append(f"     - ADP Differential (50% of historical):")
                test_results.append(f"       Performance vs draft: {'Outperformed' if adp_diff < 0 else 'Underperformed' if adp_diff > 0 else 'Met expectations'}")
                test_results.append(f"       ADP component score: {adp_component:.1f}/10")
                test_results.append(f"       Step 1: {adp_component:.1f} x 0.5 = {adp_component*0.5:.2f}")
                test_results.append(f"       Final: {adp_component*0.5:.2f}/5 points")
                
                test_results.append(f"     - Fantasy Points Scaling (30% of historical):")
                test_results.append(f"       Average fantasy points: {historical_perf['avg_fantasy_points']:.1f}")
                test_results.append(f"       Step 1: {historical_perf['avg_fantasy_points']:.1f}/25 = {fp_raw_score:.3f}")
                test_results.append(f"       Step 2: min(10, max(0, {fp_raw_score:.3f})) = {fp_component:.1f}")
                test_results.append(f"       Step 3: {fp_component:.1f} x 0.3 = {fp_component*0.3:.2f}")
                test_results.append(f"       Final: {fp_component*0.3:.2f}/3 points")
                
                test_results.append(f"     - Years of Data Reliability (20% of historical):")
                test_results.append(f"       Years available: {historical_perf['years_of_data']}")
                test_results.append(f"       Step 1: min(3, {historical_perf['years_of_data']}) = {years_component:.1f}")
                test_results.append(f"       Step 2: {years_component:.1f} x 0.2 = {years_component*0.2:.2f}")
                test_results.append(f"       Final: {years_component*0.2:.2f}/2 points")
                
                hist_score = (adp_component * 0.5 + fp_component * 0.3 + years_component * 0.2)
                test_results.append(f"   • Final Historical Score calculation:")
                test_results.append(f"     Step 1: {adp_component*0.5:.2f} + {fp_component*0.3:.2f} + {years_component*0.2:.2f}")
                test_results.append(f"     Final: {hist_score:.2f}/10 points")
            else:
                hist_score = 0.0  # No score for missing data
                test_results.append(f"   • No historical data available")
                test_results.append(f"   • Score: {hist_score:.2f}/10 (no fallback)")
            
            # 2. Advanced Metrics Score (30% weight)
            test_results.append("\n2. ADVANCED METRICS (30% weight):")
            
            # Show detailed advanced metrics calculation
            test_results.append(f"   • Available metrics: {list(advanced_metrics.keys())}")
            
            # Calculate percentile-based scoring for better differentiation
            skip_keys = {'PLAYER', 'Tm', 'POS', 'composite_score', 'player_name'}
            numeric_values = []
            metric_scores = []
            
            test_results.append("   • ENHANCED Z-SCORE NORMALIZATION (Wider Distribution):")
            test_results.append("     - Elite (z >= +1.2): 9.0-10.0 points [Steep curve]") 
            test_results.append("     - Very Good (z >= +0.5): 7.5-9.0 points")
            test_results.append("     - Average (z >= -0.5): 4.5-7.5 points [Flattened]") 
            test_results.append("     - Below Average (z >= -1.2): 3.0-4.5 points")
            test_results.append("     - Poor (z < -1.2): 1.0-3.0 points [Steep curve]")
            test_results.append("     - Very Poor (z < -1.0): 2.0-3.5 points")
            
            # Get all players at this position for percentile calculation
            current_year = max(self.years)
            position_data = None
            if current_year in self.advanced_data and position in self.advanced_data[current_year]:
                position_data = self.advanced_data[current_year][position]
            
            for key, value in advanced_metrics.items():
                if key.lower() in [k.lower() for k in skip_keys]:
                    continue
                    
                try:
                    numeric_value = None
                    if isinstance(value, (int, float)):
                        numeric_value = float(value)
                    elif isinstance(value, str):
                        clean_val = value.replace('%', '').strip()
                        if clean_val:
                            numeric_value = float(clean_val)
                    
                    if numeric_value is not None:
                        # Calculate Z-SCORE BASED scoring for true 2-10 range
                        if position_data is not None and key in position_data.columns:
                            position_values = position_data[key].dropna()
                            if len(position_values) > 3:  # Need enough data for z-scores
                                z_score, scaled_value = self._calculate_z_score_based_score(numeric_value, position_values, key)
                                
                                # Classify performance tier (updated to match new z-score mapping)
                                if z_score >= 1.2:
                                    tier = "ELITE"
                                elif z_score >= 0.5:
                                    tier = "Very Good"
                                elif z_score >= -0.5:
                                    tier = "Average"
                                elif z_score >= -1.2:
                                    tier = "Below Avg"
                                else:
                                    tier = "Poor"
                                
                                metric_scores.append((key, value, scaled_value, f"z={z_score:.2f} ({tier})"))
                                numeric_values.append(scaled_value)
                            else:
                                # Fallback to old method if not enough data
                                if numeric_value > 50:
                                    scaled_value = min(10.0, (numeric_value / 20) * 5.0)
                                else:
                                    scaled_value = min(10.0, numeric_value)
                                metric_scores.append((key, value, scaled_value, "fallback scaling"))
                                numeric_values.append(scaled_value)
                        else:
                            # Fallback to old method if no position data
                            if numeric_value > 50:
                                scaled_value = min(10.0, (numeric_value / 20) * 5.0)
                            else:
                                scaled_value = min(10.0, numeric_value)
                            metric_scores.append((key, value, scaled_value, "no position data"))
                            numeric_values.append(scaled_value)
                except (ValueError, TypeError):
                    continue
            
            # Show individual metric contributions with detailed scaling
            test_results.append("   • Individual metric scores with percentile rankings:")
            for metric, raw_val, scaled_val, stat_type in metric_scores:
                test_results.append(f"     - {metric}: {raw_val} -> {scaled_val:.2f}/10 ({stat_type})")
            
            # CRITICAL: Optimize advanced metrics averaging
            # Weight top 3 metrics higher (60%) and rest lower (40%) to reduce central tendency
            if numeric_values and len(numeric_values) >= 3:
                sorted_values = sorted(numeric_values, reverse=True)  # Sort descending
                top_3 = sorted_values[:3]
                rest = sorted_values[3:]
                
                # Weighted calculation: top 3 = 60%, rest = 40%
                top_3_weight = 0.6
                rest_weight = 0.4
                
                if rest:  # If there are metrics beyond top 3
                    weighted_score = (np.mean(top_3) * top_3_weight) + (np.mean(rest) * rest_weight)
                    test_results.append(f"   • OPTIMIZED composite calculation (top 3 weighted):")
                    test_results.append(f"     - Top 3 metrics average: {np.mean(top_3):.2f} (weight: 60%)")
                    test_results.append(f"     - Remaining {len(rest)} metrics average: {np.mean(rest):.2f} (weight: 40%)")
                    test_results.append(f"     - Weighted calculation: ({np.mean(top_3):.2f} × 0.6) + ({np.mean(rest):.2f} × 0.4) = {weighted_score:.2f}")
                else:  # Only 3 metrics available
                    weighted_score = np.mean(top_3)
                    test_results.append(f"   • Final composite calculation (only 3 metrics):")
                    test_results.append(f"     - Average of top 3: {weighted_score:.2f}/10")
                
                adv_score = min(10.0, max(0.0, weighted_score))
                test_results.append(f"   • Composite Advanced Score: {adv_score:.2f}/10 (top-weighted from {len(numeric_values)} metrics)")
            else:
                # Fallback to regular averaging if insufficient metrics
                adv_score = min(10.0, max(0.0, np.mean(numeric_values))) if numeric_values else 5.0
                test_results.append(f"   • Final composite calculation (regular average):")
                test_results.append(f"     - Sum of scaled values: {sum(numeric_values):.2f}")
                test_results.append(f"     - Average: {sum(numeric_values):.2f}/{len(numeric_values)} = {adv_score:.2f}/10")
                test_results.append(f"   • Composite Advanced Score: {adv_score:.2f}/10 (average of {len(numeric_values)} metrics)")
            
            # 3. ML Predictions Score (30% weight)
            test_results.append("\n3. ML PREDICTIONS (30% weight):")
            
            # Show ML scoring structure
            test_results.append("   • ML Score Structure:")
            test_results.append("     - Breakout Probability: 30% of ML score (3/10 points)")
            test_results.append("     - Consistency Score: 30% of ML score (3/10 points)")
            test_results.append("     - Positional Value: 40% of ML score (4/10 points)")
            test_results.append("     - Total ML Score: /10 points")
            
            # IMPROVED ML SCORING with expanded range
            test_results.append("   • IMPROVED ML SCORING:")
            test_results.append("     - Enhanced scaling to prevent compression")
            test_results.append("     - Breakout uses exponential curve for differentiation")
            test_results.append("     - Positional value uses percentile-based scaling")
            test_results.append("     - Minimum baseline scores to prevent zeros")
            
            # ENHANCED ML SCORING with expanded percentile banding
            breakout_raw = ml_predictions['breakout_probability']
            pos_value_raw = ml_predictions['positional_value']
            
            # Enhanced breakout scoring using percentile bands (0.6-3.0 range)
            if breakout_raw >= 0.9:  # 90th+ percentile
                breakout_enhanced = 2.7 + (breakout_raw - 0.9) / 0.1 * 0.3  # 2.7-3.0
            elif breakout_raw >= 0.75:  # 75-89th  
                breakout_enhanced = 2.25 + (breakout_raw - 0.75) / 0.15 * 0.45  # 2.25-2.7
            elif breakout_raw >= 0.5:  # 50-74th
                breakout_enhanced = 1.65 + (breakout_raw - 0.5) / 0.25 * 0.6  # 1.65-2.25
            elif breakout_raw >= 0.25:  # 25-49th
                breakout_enhanced = 1.05 + (breakout_raw - 0.25) / 0.25 * 0.6  # 1.05-1.65
            else:  # <25th percentile
                breakout_enhanced = 0.6 + breakout_raw / 0.25 * 0.45  # 0.6-1.05
            
            # Consistency scoring with percentile bands (0.6-3.0 range)
            consistency_raw = ml_predictions['consistency_score'] / 10  # Normalize to 0-1
            if consistency_raw >= 0.9:
                consistency_enhanced = 2.7 + (consistency_raw - 0.9) / 0.1 * 0.3  # 2.7-3.0
            elif consistency_raw >= 0.75:
                consistency_enhanced = 2.25 + (consistency_raw - 0.75) / 0.15 * 0.45  # 2.25-2.7
            elif consistency_raw >= 0.5:
                consistency_enhanced = 1.65 + (consistency_raw - 0.5) / 0.25 * 0.6  # 1.65-2.25
            elif consistency_raw >= 0.25:
                consistency_enhanced = 1.05 + (consistency_raw - 0.25) / 0.25 * 0.6  # 1.05-1.65
            else:
                consistency_enhanced = 0.6 + consistency_raw / 0.25 * 0.45  # 0.6-1.05
            
            # Enhanced positional value scoring with percentile bands (0.8-4.0 range)
            if pos_value_raw >= 0.9:  # Elite 
                positional_enhanced = 3.6 + (pos_value_raw - 0.9) / 0.1 * 0.4  # 3.6-4.0
            elif pos_value_raw >= 0.75:  # Very good
                positional_enhanced = 3.0 + (pos_value_raw - 0.75) / 0.15 * 0.6  # 3.0-3.6
            elif pos_value_raw >= 0.5:  # Above average
                positional_enhanced = 2.2 + (pos_value_raw - 0.5) / 0.25 * 0.8  # 2.2-3.0
            elif pos_value_raw >= 0.25:  # Below average
                positional_enhanced = 1.4 + (pos_value_raw - 0.25) / 0.25 * 0.8  # 1.4-2.2
            else:  # Poor
                positional_enhanced = 0.8 + pos_value_raw / 0.25 * 0.6  # 0.8-1.4
            
            breakout_component = breakout_enhanced
            consistency_component = consistency_enhanced
            positional_component = positional_enhanced
            
            test_results.append("   • Component breakdown with ENHANCED calculations:")
            test_results.append(f"     - Breakout Probability (30% of ML):")
            test_results.append(f"       Raw probability: {breakout_raw:.3f}")
            test_results.append(f"       Enhanced formula: 0.5 + ({breakout_raw:.3f}^1.5) * 2.5")
            test_results.append(f"       Enhanced score: {breakout_enhanced:.2f}")
            test_results.append(f"       Final: {breakout_component:.2f}/3 points")
            
            test_results.append(f"     - Consistency Score (30% of ML):")
            test_results.append(f"       Raw score: {ml_predictions['consistency_score']:.1f}/10")
            test_results.append(f"       Step 1: {ml_predictions['consistency_score']:.1f}/10 = {ml_predictions['consistency_score']/10:.3f}")
            test_results.append(f"       Step 2: {ml_predictions['consistency_score']/10:.3f} x 3 = {consistency_component:.2f}")
            test_results.append(f"       Final: {consistency_component:.2f}/3 points")
            
            test_results.append(f"     - Positional Value (40% of ML):")
            test_results.append(f"       Raw value: {pos_value_raw:.3f}")
            if pos_value_raw <= 0.001:
                test_results.append(f"       Enhanced: Minimum baseline = 1.0")
            elif pos_value_raw <= 0.1:
                test_results.append(f"       Enhanced: 1.0 + ({pos_value_raw:.3f}/0.1) * 1.0 = {positional_enhanced:.2f}")
            elif pos_value_raw <= 0.5:
                test_results.append(f"       Enhanced: 2.0 + (({pos_value_raw:.3f}-0.1)/0.4) * 1.0 = {positional_enhanced:.2f}")
            else:
                test_results.append(f"       Enhanced: 3.0 + min(1.0, ({pos_value_raw:.3f}-0.5)/0.5) = {positional_enhanced:.2f}")
            test_results.append(f"       Final: {positional_component:.2f}/4 points")
            
            test_results.append(f"   • Model Confidence: {ml_predictions['model_confidence']:.3f}")
            
            ml_score = breakout_component + consistency_component + positional_component
            test_results.append(f"   • Final ML Score calculation:")
            test_results.append(f"     Step 1: {breakout_component:.2f} + {consistency_component:.2f} + {positional_component:.2f}")
            test_results.append(f"     Final: {ml_score:.2f}/10 points")
            
            # 4. Injury Profile Score (15% weight)
            test_results.append("\n4. INJURY PROFILE (15% weight):")
            
            # Show injury scoring structure
            test_results.append("   • Injury Score Structure:")
            test_results.append("     - Durability Score: 60% of injury score (6/10 points)")
            test_results.append("     - Games Missed calculation: 40% of injury score (4/10 points)")
            test_results.append("     - Total Injury Score: /10 points")
            
            # Show detailed injury calculation with step-by-step breakdown
            durability_component = (injury_profile['durability_score'] / 10) * 6
            games_available = 17 - injury_profile['projected_games_missed']
            availability_ratio = games_available / 17
            games_missed_component = availability_ratio * 4
            
            test_results.append("   • Component breakdown with detailed calculations:")
            test_results.append(f"     - Durability Score (60% of injury):")
            test_results.append(f"       Raw durability: {injury_profile['durability_score']:.1f}/10")
            test_results.append(f"       Step 1: {injury_profile['durability_score']:.1f}/10 = {injury_profile['durability_score']/10:.3f}")
            test_results.append(f"       Step 2: {injury_profile['durability_score']/10:.3f} x 6 = {durability_component:.2f}")
            test_results.append(f"       Final: {durability_component:.2f}/6 points")
            
            test_results.append(f"     - Games Missed Calculation (40% of injury):")
            test_results.append(f"       Projected games missed: {injury_profile['projected_games_missed']:.1f}")
            test_results.append(f"       Step 1: Games available = 17 - {injury_profile['projected_games_missed']:.1f} = {games_available:.1f}")
            test_results.append(f"       Step 2: Availability ratio = {games_available:.1f}/17 = {availability_ratio:.3f}")
            test_results.append(f"       Step 3: {availability_ratio:.3f} x 4 = {games_missed_component:.2f}")
            test_results.append(f"       Final: {games_missed_component:.2f}/4 points")
            
            test_results.append(f"     - Risk Assessment:")
            test_results.append(f"       Risk Level: {injury_profile['injury_risk_level']}")
            test_results.append(f"       Risk Percentage: {injury_profile['injury_risk']*100:.1f}%")
            
            injury_score = durability_component + games_missed_component
            test_results.append(f"   • Final Injury Score calculation:")
            test_results.append(f"     Step 1: {durability_component:.2f} + {games_missed_component:.2f}")
            test_results.append(f"     Final: {injury_score:.2f}/10 points")
            
            # 5. Calculate Final Score with NON-LINEAR AGGREGATION
            test_results.append("\nFINAL SCORE CALCULATION WITH NON-LINEAR AGGREGATION:")
            test_results.append("-" * 60)
            
            # Redistribute weight from historical to other components
            advanced_weight = 0.30 + (weight_redistributed * 0.4)  # Gets 40% of redistributed weight
            ml_weight = 0.30 + (weight_redistributed * 0.4)        # Gets 40% of redistributed weight  
            injury_weight = 0.15 + (weight_redistributed * 0.2)    # Gets 20% of redistributed weight
            
            # Standard weighted calculation
            weighted_historical = hist_score * historical_weight
            weighted_advanced = adv_score * advanced_weight
            weighted_ml = ml_score * ml_weight
            weighted_injury = injury_score * injury_weight
            standard_score = weighted_historical + weighted_advanced + weighted_ml + weighted_injury
            
            # Apply non-linear aggregation
            final_score, aggregation_details = self._calculate_non_linear_aggregated_score(
                hist_score, adv_score, ml_score, injury_score,
                historical_weight, advanced_weight, ml_weight, injury_weight
            )
            
            test_results.append("COMPONENT SCORES:")
            test_results.append(f"Historical ({historical_weight*100:.0f}%): {hist_score:.2f} x {historical_weight:.2f} = {weighted_historical:.2f}")
            test_results.append(f"Advanced ({advanced_weight*100:.0f}%):  {adv_score:.2f} x {advanced_weight:.2f} = {weighted_advanced:.2f}")
            test_results.append(f"ML ({ml_weight*100:.0f}%):        {ml_score:.2f} x {ml_weight:.2f} = {weighted_ml:.2f}")
            test_results.append(f"Injury ({injury_weight*100:.0f}%):    {injury_score:.2f} x {injury_weight:.2f} = {weighted_injury:.2f}")
            
            test_results.append(f"\nNON-LINEAR AGGREGATION DETAILS:")
            test_results.append(f"Standard Score: {standard_score:.2f}")
            if aggregation_details['elite_components']:
                test_results.append(f"Elite Performance Bonus: +{aggregation_details['elite_bonus']:.2f}")
                test_results.append(f"Elite Components: {', '.join(aggregation_details['elite_components'])}")
            elif aggregation_details['elite_bonus'] < 0:
                test_results.append(f"Poor Performance Penalty: {aggregation_details['elite_bonus']:.2f}")
            else:
                test_results.append(f"No elite/poor adjustments applied")
            
            if weight_redistributed > 0:
                test_results.append(f"Weight redistributed from Historical: {weight_redistributed:.2f} ({weight_redistributed*100:.0f}%)")
            test_results.append(f"" + "-" * 30)
            test_results.append(f"FINAL SCORE: {final_score:.2f}/10 ({final_score*10:.0f}/100)")
            
            # Star rating
            star_rating = min(5, max(1, int((final_score / 10) * 5) + 1))
            stars = "*" * star_rating + "-" * (5 - star_rating)
            test_results.append(f"STAR RATING: {stars} ({star_rating}/5)")
        
        # Add missing data report
        test_results.append(f"\n{'='*80}")
        test_results.append("MISSING DATA REPORT")
        test_results.append(f"{'='*80}")
        
        if self.missing_data_log:
            test_results.append("Data gaps identified during testing:")
            test_results.append("-" * 40)
            for entry in self.missing_data_log:
                test_results.append(f"• Player: {entry['player']} ({entry['position']})")
                test_results.append(f"  Missing: {entry['missing_stat']}")
                test_results.append(f"  Attempted columns: {entry['attempted_columns']}")
                test_results.append(f"  Data source: {entry.get('data_source', 'unknown')}")
                test_results.append(f"  Time: {entry['timestamp']}")
                test_results.append("")
        else:
            test_results.append("OK - No missing data detected during test")
        
        # Add fallback usage report
        if self.debug_log:
            test_results.append("\nFallback/Estimation Usage:")
            test_results.append("-" * 40)
            for entry in self.debug_log:
                if entry.get('action') == 'FALLBACK_USED':
                    test_results.append(f"• Player: {entry['player']} ({entry['position']})")
                    test_results.append(f"  Stat: {entry['stat']}")
                    test_results.append(f"  Fallback value: {entry['fallback_value']}")
                    test_results.append(f"  Reason: {entry['reason']}")
                    test_results.append("")
        
        # Add summary
        test_results.append(f"\n{'='*80}")
        test_results.append("TEST SUMMARY")
        test_results.append(f"{'='*80}")
        test_results.append("Scoring system test completed successfully")
        test_results.append("Review the individual component scores above")
        test_results.append("Provide feedback on scoring weights or calculations")
        test_results.append("Run test again with: analyzer.run_scoring_test()")
        
        result_text = "\n".join(test_results)
        print(result_text)
        return result_text

# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your data directory
    # Set enable_testing_mode=True to run scoring tests instead of full analysis
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=False)
    
    # Check if testing mode is enabled
    if analyzer.enable_testing_mode:
        print("\nTESTING MODE ENABLED")
        print("="*40)
        print("Running scoring system test with 10 random players...")
        analyzer.run_scoring_test()
        print("\nTesting complete! To run full analysis, set enable_testing_mode=False")
        exit()
    
    # First, diagnose CSV files for issues
    analyzer.diagnose_csv_files()
    
    # Load all data files
    analyzer.load_data()
    
    # Generate dual output system
    print("\n" + "="*60)
    print("GENERATING DUAL OUTPUT SYSTEM")
    print("="*60)
    
    # 1. Generate league analysis (existing functionality)
    print("Generating league analysis report...")
    league_report = analyzer.save_report("league_analysis.txt")
    print(f"League analysis saved to: {league_report}")
    
    # 2. Generate ML-enhanced player profiles
    print("\nGenerating ML-enhanced player profiles...")
    player_profiles = analyzer.generate_player_profiles()
    profiles_file = analyzer.save_player_profiles(player_profiles, "player_profiles.txt")
    json_file = analyzer.save_profiles_json(player_profiles, "player_profiles.json")
    
    print(f" Player profiles saved to: {profiles_file}")
    print(f" JSON profiles for Custom GPT saved to: {json_file}")
    print(f" Generated profiles for {len(player_profiles)} players")
    
    # Summary
    print(f"\n ANALYSIS COMPLETE!")
    print("="*40)
    print("Generated Files:")
    print(f"  • {league_report} - League draft tendencies and manager profiles")
    print(f"  • {profiles_file} - ML-enhanced player profiles")
    print(f"  • {json_file} - JSON format for Custom GPT integration")
    print("\nReady for 2025 fantasy draft! ")
    print("\nTo run scoring tests: Set enable_testing_mode=True and run again")