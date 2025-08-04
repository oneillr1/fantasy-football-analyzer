"""
Machine Learning Models Module for Fantasy Football Analyzer

Handles all ML model training, predictions, and feature engineering.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import warnings

from config.constants import (
    ML_MODEL_TYPES, POSITIONS, KEY_METRICS, DEFAULT_SCORE
)

warnings.filterwarnings('ignore')


class MLModels:
    """
    Handles all machine learning model training and predictions.
    
    Model types:
    - breakout: Predicts breakout probability
    - consistency: Predicts consistency score
    - positional_value: Predicts positional value
    - injury_risk: Predicts injury risk
    """
    
    def __init__(self, data_loader):
        """
        Initialize the ML models manager.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        
        # ML models for player profiling
        self.ml_models = {
            'breakout': {},
            'consistency': {},
            'positional_value': {},
            'injury_risk': {}
        }
        self.scalers = {}
        self.model_confidence = {}
    
    def train_models(self) -> None:
        """Train ML models for player profiling."""
        positions = POSITIONS
        
        for position in positions:
            print(f"Training models for {position}...")
            print(f"  Available years: {self.data_loader.years}")
            print(f"  Training years: {self.data_loader.years[:-1]}")
            print(f"  Results data years: {list(self.data_loader.results_data.keys())}")
            print(f"  Advanced data years: {list(self.data_loader.advanced_data.keys())}")
            print(f"  ADP data years: {list(self.data_loader.adp_data.keys())}")
            
            # Collect training data across all years
            all_features = []
            all_targets = {
                'breakout': [],
                'consistency': [],
                'positional_value': [],
                'fantasy_points': []
            }
            
            for year in self.data_loader.years[:-1]:  # Exclude most recent year for prediction
                features_df = self._prepare_ml_features(position, year)
                if features_df is None:
                    continue
                
                # Get corresponding results for targets
                if year not in self.data_loader.results_data:
                    continue
                
                results_df = self.data_loader.results_data[year]
                adp_df = self.data_loader.adp_data.get(year)
                
                print(f"    Processing {year}: {len(features_df)} players in advanced data")
                for _, row in features_df.iterrows():
                    player = row['Player']
                    
                    # Skip if player name is not valid
                    if not isinstance(player, str) or pd.isna(player) or len(player.strip()) == 0:
                        continue
                    
                    # Clean player name for better matching
                    clean_player = self.data_loader.clean_player_name_for_matching(player)
                    if not clean_player:
                        continue
                    
                    # Find player in results using enhanced matching
                    results_player_col = 'Player' if 'Player' in results_df.columns else 'PLAYER'
                    
                    # First try exact match with cleaned names
                    results_df_cleaned = results_df.copy()
                    results_df_cleaned['Player_Clean'] = results_df_cleaned[results_player_col].apply(
                        self.data_loader.clean_player_name_for_matching)
                    player_results = results_df_cleaned[results_df_cleaned['Player_Clean'] == clean_player]
                    
                    # If no exact match, try fuzzy matching
                    if player_results.empty:
                        player_results = results_df[results_df[results_player_col].str.contains(
                            player.split()[0], case=False, na=False)]
                        if player_results.empty:
                            continue
                    
                    player_result = player_results.iloc[0]
                    fantasy_points = self.data_loader.safe_float_conversion(
                        player_result.get('FPTS', player_result.get('TTL', player_result.get('PTS', 0))))
                    finish_rank = self.data_loader.safe_float_conversion(
                        player_result.get('RK', player_result.get('#', 999)))
                    
                    # Get ADP data using enhanced matching
                    adp_rank = 999
                    if adp_df is not None:
                        adp_player_col = 'Player' if 'Player' in adp_df.columns else 'PLAYER'
                        
                        # First try exact match with cleaned names
                        adp_df_cleaned = adp_df.copy()
                        adp_df_cleaned['Player_Clean'] = adp_df_cleaned[adp_player_col].apply(
                            self.data_loader.clean_player_name_for_matching)
                        player_adp = adp_df_cleaned[adp_df_cleaned['Player_Clean'] == clean_player]
                        
                        # If no exact match, try fuzzy matching
                        if player_adp.empty:
                            player_adp = adp_df[adp_df[adp_player_col].str.contains(
                                player.split()[0], case=False, na=False)]
                        
                        if not player_adp.empty:
                            adp_rank = self.data_loader.safe_float_conversion(
                                player_adp.iloc[0].get('ADP', player_adp.iloc[0].get('AVG', 999)))
                    
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
                    # For now, use a simplified consistency metric
                    consistency = max(0.0, min(1.0, 1.0 - (abs(adp_diff) / 50)))  # 0-1 scale
                    
                    # 3. Positional value (0-1) based on finish rank vs position average
                    positional_value = max(0.0, min(1.0, (50 - finish_rank) / 50))  # 0-1 scale
                    
                    # 4. Fantasy points (raw value for regression)
                    fantasy_points_normalized = max(0.0, min(1.0, fantasy_points / 400))  # 0-1 scale
                    
                    # Store features and targets
                    feature_vector = self._extract_features(row, position)
                    if feature_vector is not None:
                        all_features.append(feature_vector)
                        all_targets['breakout'].append(breakout)
                        all_targets['consistency'].append(consistency)
                        all_targets['positional_value'].append(positional_value)
                        all_targets['fantasy_points'].append(fantasy_points_normalized)
            
            if not all_features:
                print(f"    No training data available for {position}")
                continue
            
            # Convert to numpy arrays
            X = np.array(all_features)
            print(f"    Training data shape: {X.shape}")
            
            # Train models for each target
            for target_name in ['breakout', 'consistency', 'positional_value']:
                y = np.array(all_targets[target_name])
                
                if len(y) < 10:  # Need minimum data for training
                    print(f"    Insufficient data for {target_name} model ({len(y)} samples)")
                    continue
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
                model.fit(X_train_scaled, y_train)
                
                # Evaluate model
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                confidence = max(0.0, min(1.0, r2))  # Use R² as confidence
                
                # Store model and metadata
                self.ml_models[target_name][position] = model
                self.scalers[f"{target_name}_{position}"] = scaler
                self.model_confidence[f"{target_name}_{position}"] = confidence
                
                print(f"    {target_name} model trained - Confidence: {confidence:.3f}")
    
    def _prepare_ml_features(self, position: str, year: int) -> Optional[pd.DataFrame]:
        """
        Prepare features for ML training.
        
        Args:
            position: Player position
            year: Year of data
            
        Returns:
            DataFrame with features, or None if no data
        """
        if position not in self.data_loader.advanced_data or year not in self.data_loader.advanced_data[position]:
            return None
        
        df = self.data_loader.advanced_data[position][year]
        
        # Get key metrics for this position
        key_metrics = KEY_METRICS.get(position, [])
        
        # Filter to only players with sufficient data
        valid_players = []
        for _, row in df.iterrows():
            player = row.get('PLAYER', row.get('Player', ''))
            if pd.isna(player) or not isinstance(player, str):
                continue
            
            # Check if player has sufficient metrics
            metric_count = 0
            for metric in key_metrics:
                if metric in row.index and not pd.isna(row[metric]):
                    metric_count += 1
            
            if metric_count >= len(key_metrics) * 0.5:  # At least 50% of metrics
                valid_players.append(row)
        
        if not valid_players:
            return None
        
        return pd.DataFrame(valid_players)
    
    def _extract_features(self, row: pd.Series, position: str) -> Optional[np.ndarray]:
        """
        Extract features from a player row for ML training.
        
        Args:
            row: Player's data row
            position: Player position
            
        Returns:
            Feature array, or None if insufficient data
        """
        key_metrics = KEY_METRICS.get(position, [])
        features = []
        
        for metric in key_metrics:
            if metric in row.index and not pd.isna(row[metric]):
                value = self.data_loader.safe_float_conversion(row[metric])
                features.append(value)
            else:
                features.append(0.0)  # Default value for missing metrics
        
        if len(features) != len(key_metrics):
            return None
        
        return np.array(features)
    
    def generate_predictions(self, player: str, position: str, features: np.ndarray, 
                           debug_mode: bool = False) -> Dict[str, float]:
        """
        Generate ML predictions for a player.
        
        Args:
            player: Player name
            position: Player position
            features: Feature array
            debug_mode: Enable debug output
            
        Returns:
            Dictionary of predictions
        """
        predictions = {}
        
        for model_type in ML_MODEL_TYPES:
            if model_type not in self.ml_models or position not in self.ml_models[model_type]:
                predictions[model_type] = DEFAULT_SCORE
                continue
            
            model = self.ml_models[model_type][position]
            scaler_key = f"{model_type}_{position}"
            
            if scaler_key not in self.scalers:
                predictions[model_type] = DEFAULT_SCORE
                continue
            
            scaler = self.scalers[scaler_key]
            confidence = self.model_confidence.get(scaler_key, 0.5)
            
            try:
                # Scale features
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                
                # Apply confidence weighting
                weighted_prediction = prediction * confidence
                
                predictions[model_type] = max(0.0, min(1.0, weighted_prediction))
                
                if debug_mode:
                    print(f"  {model_type}: {prediction:.3f} (confidence: {confidence:.3f})")
                
            except Exception as e:
                if debug_mode:
                    print(f"  {model_type}: Error - {e}")
                predictions[model_type] = DEFAULT_SCORE
        
        return predictions
    
    def create_advanced_features(self, df: pd.DataFrame, position: str) -> pd.DataFrame:
        """
        Create advanced features for ML training.
        
        Args:
            df: DataFrame with player data
            position: Player position
            
        Returns:
            DataFrame with additional features
        """
        df_copy = df.copy()
        
        # Add position-specific features
        if position == 'QB':
            # QB-specific features
            if 'CMP' in df_copy.columns and 'ATT' in df_copy.columns:
                df_copy['COMP_PCT'] = df_copy['CMP'] / df_copy['ATT'].replace(0, 1)
            
            if 'YDS' in df_copy.columns and 'ATT' in df_copy.columns:
                df_copy['YPA'] = df_copy['YDS'] / df_copy['ATT'].replace(0, 1)
        
        elif position in ['RB', 'WR', 'TE']:
            # RB/WR/TE-specific features
            if 'REC' in df_copy.columns and 'TGT' in df_copy.columns:
                df_copy['CATCH_RATE'] = df_copy['REC'] / df_copy['TGT'].replace(0, 1)
            
            if 'YDS' in df_copy.columns and 'REC' in df_copy.columns:
                df_copy['YPR'] = df_copy['YDS'] / df_copy['REC'].replace(0, 1)
        
        # Fill NaN values with 0
        df_copy = df_copy.fillna(0)
        
        return df_copy
    
    def calculate_advanced_performance_score(self, player_row: pd.Series, position: str) -> float:
        """
        Calculate advanced performance score using ML models.
        
        Args:
            player_row: Player's advanced metrics row
            position: Player position
            
        Returns:
            Advanced performance score (0-10)
        """
        # Extract features
        features = self._extract_features(player_row, position)
        if features is None:
            return DEFAULT_SCORE
        
        # Generate predictions
        predictions = self.generate_predictions("unknown", position, features)
        
        # Calculate weighted average of predictions
        total_score = 0.0
        total_weight = 0.0
        
        for prediction_type, score in predictions.items():
            if score is not None and not pd.isna(score):
                # Normalize to 0-10 scale
                normalized_score = score * 10.0
                weight = 1.0  # Equal weight for all predictions
                
                total_score += normalized_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return DEFAULT_SCORE
        
        return total_score / total_weight
    
    def find_similar_players(self, target_player: str, position: str, n_similar: int = 10) -> List[Dict[str, Any]]:
        """
        Find similar players using ML-based similarity.
        
        Args:
            target_player: Name of target player
            position: Player position
            n_similar: Number of similar players to find
            
        Returns:
            List of similar player dictionaries
        """
        if position not in self.data_loader.advanced_data:
            return []
        
        # Get target player's features
        target_features = None
        target_row = None
        
        # Search in most recent year's data
        for year in reversed(self.data_loader.years):
            if position in self.data_loader.advanced_data and year in self.data_loader.advanced_data[position]:
                df = self.data_loader.advanced_data[position][year]
                
                # Find target player
                player_col = self.data_loader.get_player_column(df)
                if player_col:
                    target_mask = df[player_col].str.contains(target_player, case=False, na=False)
                    if target_mask.any():
                        target_row = df[target_mask].iloc[0]
                        target_features = self._extract_features(target_row, position)
                        break
        
        if target_features is None:
            return []
        
        # Find similar players
        similarities = []
        
        for year in self.data_loader.years:
            if position in self.data_loader.advanced_data and year in self.data_loader.advanced_data[position]:
                df = self.data_loader.advanced_data[position][year]
                
                for _, row in df.iterrows():
                    player_name = row.get('PLAYER', row.get('Player', ''))
                    if pd.isna(player_name) or not isinstance(player_name, str):
                        continue
                    
                    # Skip target player
                    if target_player.lower() in player_name.lower():
                        continue
                    
                    # Extract features
                    features = self._extract_features(row, position)
                    if features is None:
                        continue
                    
                    # Calculate similarity (cosine similarity)
                    similarity = np.dot(target_features, features) / (
                        np.linalg.norm(target_features) * np.linalg.norm(features))
                    
                    similarities.append({
                        'player': player_name,
                        'year': year,
                        'similarity': similarity,
                        'features': features
                    })
        
        # Sort by similarity and return top N
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:n_similar] 