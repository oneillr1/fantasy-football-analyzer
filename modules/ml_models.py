"""
Enhanced Machine Learning Models Module for Fantasy Football Analyzer

Handles all ML model training, predictions, and feature engineering with advanced data integration.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings

from config.constants import (
    ML_MODEL_TYPES, POSITIONS, KEY_METRICS, DEFAULT_SCORE
)

warnings.filterwarnings('ignore')

class MLModels:
    """
    Enhanced ML models manager focused on year-to-year prediction (2024→2025) and age/injury integration.
    """
    def __init__(self, data_loader):
        self.data_loader = data_loader
        self.ml_models = {'breakout': {}, 'consistency': {}, 'positional_value': {}, 'injury_risk': {}}
        self.scalers = {}
        self.model_confidence = {}
        self.best_models = {}
        self.model_configs = {
            'random_forest': {
                'model': RandomForestRegressor,
                'params': {'n_estimators': [50, 100], 'max_depth': [5, 10, None], 'random_state': [42]}
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'params': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5], 'random_state': [42]}
            },
            'ridge': {
                'model': Ridge,
                'params': {'alpha': [0.1, 1.0, 10.0]}
            },
            'lasso': {
                'model': Lasso,
                'params': {'alpha': [0.1, 1.0, 10.0], 'random_state': [42]}
            }
        }
        # Load player age data
        self.player_age_data = {}
        try:
            age_df = pd.read_csv('fantasy_data/player_age.csv')
            for _, row in age_df.iterrows():
                # Skip DST entries and entries with non-numeric age
                if row['POS'].startswith('DST') or row['AGE'] == '-' or pd.isna(row['AGE']):
                    continue
                try:
                    age = float(row['AGE'])
                    self.player_age_data[row['PLAYER NAME']] = age
                except (ValueError, TypeError):
                    continue  # Skip entries that can't be converted to float
        except Exception as e:
            print(f"Warning: Could not load player age data: {e}")
        # Load injury predictor data
        self.injury_data = {}
        for pos in ['qb', 'rb', 'wr', 'te']:
            try:
                df = pd.read_csv(f'fantasy_data/injury_predictor_{pos}.csv')
                self.injury_data[pos.upper()] = df
            except Exception as e:
                print(f"Warning: Could not load injury data for {pos}: {e}")
    
    def train_models(self) -> None:
        positions = POSITIONS
        for position in positions:
            print(f"Training year-to-year models for {position}...")
            all_features = []
            all_targets = {'breakout': [], 'consistency': [], 'positional_value': [], 'injury_risk': []}
            # Train on consecutive year pairs
            for i in range(len(self.data_loader.years) - 1):
                year = self.data_loader.years[i]
                next_year = self.data_loader.years[i + 1]
                features_df = self._prepare_enhanced_ml_features(position, year)
                if features_df is None or next_year not in self.data_loader.results_data:
                    continue
                next_results = self.data_loader.results_data[next_year]
                next_adp = self.data_loader.adp_data.get(next_year)
                for _, row in features_df.iterrows():
                    player = row['Player']
                    clean_player = self.data_loader.clean_player_name_for_matching(player)
                    # Find player in next year's results
                    next_results_cleaned = next_results.copy()
                    results_player_col = 'Player' if 'Player' in next_results_cleaned.columns else 'PLAYER'
                    next_results_cleaned['Player_Clean'] = next_results_cleaned[results_player_col].apply(self.data_loader.clean_player_name_for_matching)
                    next_row = next_results_cleaned[next_results_cleaned['Player_Clean'] == clean_player]
                    if next_row.empty:
                        continue
                    next_row = next_row.iloc[0]
                    # Get next year's ADP
                    next_adp_rank = 999
                    if next_adp is not None:
                        adp_player_col = 'Player' if 'Player' in next_adp.columns else 'PLAYER'
                        next_adp_cleaned = next_adp.copy()
                        next_adp_cleaned['Player_Clean'] = next_adp_cleaned[adp_player_col].apply(self.data_loader.clean_player_name_for_matching)
                        player_next_adp = next_adp_cleaned[next_adp_cleaned['Player_Clean'] == clean_player]
                        if not player_next_adp.empty:
                            next_adp_rank = self.data_loader.safe_float_conversion(player_next_adp.iloc[0].get('ADP', player_next_adp.iloc[0].get('AVG', 999)))
                    # Extract features from year, targets from next_year
                    feature_vector = self._extract_enhanced_features_with_age_injury(row, position, player, year)
                    targets = self._calculate_year_to_year_targets(next_adp_rank, next_row, position, player, next_year)
                    if feature_vector is not None:
                        all_features.append(feature_vector)
                        for target_name, target_value in targets.items():
                            all_targets[target_name].append(target_value)
            if not all_features:
                print(f"    No year-to-year training data available for {position}")
                continue
            X = np.array(all_features)
            for target_name in ['breakout', 'consistency', 'positional_value', 'injury_risk']:
                y = np.array(all_targets[target_name])
                if len(y) < 10 or np.std(y) == 0:
                    print(f"    Insufficient/No variance for {target_name} model ({len(y)} samples)")
                    continue
                best_model, best_score = self._train_multiple_models(X, y, target_name, position)
                if best_model is not None:
                    self.best_models[f"{target_name}_{position}"] = best_model
                    self.model_confidence[f"{target_name}_{position}"] = best_score
                    print(f"    Best {target_name} model: {best_score:.3f}")

    def _prepare_enhanced_ml_features(self, position: str, year: int) -> Optional[pd.DataFrame]:
        if position not in self.data_loader.advanced_data or year not in self.data_loader.advanced_data[position]:
            return None
        df = self.data_loader.advanced_data[position][year]
        key_metrics = KEY_METRICS.get(position, [])
        valid_players = []
        for _, row in df.iterrows():
            player = row.get('PLAYER', row.get('Player', ''))
            if pd.isna(player) or not isinstance(player, str):
                continue
            metric_count = 0
            for metric in key_metrics:
                if metric in row.index and not pd.isna(row[metric]):
                    metric_count += 1
            if metric_count >= len(key_metrics) * 0.5:
                valid_players.append(row)
        if not valid_players:
            return None
        return pd.DataFrame(valid_players)
    
    def _extract_enhanced_features_with_age_injury(self, row, position, player, year):
        features = []
        key_metrics = KEY_METRICS.get(position, [])
        for metric in key_metrics:
            value = self.data_loader.safe_float_conversion(row[metric]) if metric in row.index and not pd.isna(row[metric]) else 0.0
            features.append(value)
        
        # Games played and missed (accounting for season length)
        games_played = self.data_loader.safe_float_conversion(row.get('G', 0))
        features.append(games_played)
        # Season length: 16 games before 2021, 17 games from 2021 onwards
        season_length = 17 if year >= 2021 else 16
        games_missed = max(0, season_length - games_played)
        features.append(games_missed)
        durability_ratio = games_played / season_length if games_played > 0 else 0.0
        features.append(durability_ratio)
        
        # NEW: Enhanced PPG-related features
        total_fantasy_points = self.data_loader.safe_float_conversion(row.get('FPTS', row.get('TTL', row.get('PTS', 0))))
        ppg = total_fantasy_points / games_played if games_played > 0 else 0.0
        features.append(ppg)  # Points per game
        features.append(ppg / 25.0)  # Normalized PPG (assuming 25 PPG is elite)
        
        # NEW: Weekly consistency features (if weekly data available)
        weekly_points = []
        for week in range(1, 19):  # Weeks 1-18
            week_points = self.data_loader.safe_float_conversion(row.get(str(week), 0))
            if week_points > 0:  # Only count played weeks
                weekly_points.append(week_points)
        
        if weekly_points:
            ppg_std = np.std(weekly_points)
            ppg_mean = np.mean(weekly_points)
            ppg_cv = ppg_std / ppg_mean if ppg_mean > 0 else 1.0
            ppg_consistency = max(0, min(1, 1 - ppg_cv))
            features.append(ppg_consistency)  # PPG consistency (0-1)
            features.append(ppg_std)  # PPG standard deviation
            features.append(max(weekly_points))  # Best single game
            features.append(min(weekly_points))  # Worst single game
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])  # Default values if no weekly data
        
        # Age features
        age = self.player_age_data.get(player, 25.0)
        features.append(age)
        features.append(age / 40.0)
        features.append(max(0, (age - 25) / 10))
        
        # Injury features
        injury_row = self.injury_data.get(position.upper())
        if injury_row is not None:
            features.append(float(injury_row.get('Projected Games Missed', 0)))
            features.append(float(injury_row.get('Career Injuries', 0)))
            features.append(float(injury_row.get('Injuries Per Season', 0)))
        else:
            features.extend([0, 0, 0])
        
        # Position-specific engineered features (as before)
        if position == 'QB':
            att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
            comp = self.data_loader.safe_float_conversion(row.get('COMP', 0))
            yds = self.data_loader.safe_float_conversion(row.get('YDS', 0))
            air = self.data_loader.safe_float_conversion(row.get('AIR', 0))
            rtg = self.data_loader.safe_float_conversion(row.get('RTG', 0))
            sack = self.data_loader.safe_float_conversion(row.get('SACK', 0))
            knck = self.data_loader.safe_float_conversion(row.get('KNCK', 0))
            hrry = self.data_loader.safe_float_conversion(row.get('HRRY', 0))
            rz_att = self.data_loader.safe_float_conversion(row.get('RZ ATT', 0))
            features.extend([
                comp / att if att else 0.0,
                yds / att if att else 0.0,
                air / att if att else 0.0,
                rtg / 100 if rtg else 0.0,
                (sack + knck + hrry) / att if att else 0.0,
                rz_att / att if att else 0.0,
                (sack + knck + hrry) / games_played if games_played else 0.0,
            ])
        elif position == 'RB':
            att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
            rec = self.data_loader.safe_float_conversion(row.get('REC', 0))
            tgt = self.data_loader.safe_float_conversion(row.get('TGT', 0)) or 1
            yds = self.data_loader.safe_float_conversion(row.get('YDS', 0))
            yacon = self.data_loader.safe_float_conversion(row.get('YACON', 0))
            brktkl = self.data_loader.safe_float_conversion(row.get('BRKTKL', 0))
            rz_tgt = self.data_loader.safe_float_conversion(row.get('RZ TGT', 0))
            features.extend([
                yds / att if att else 0.0,
                yacon / att if att else 0.0,
                brktkl / att if att else 0.0,
                rec / tgt if tgt else 0.0,
                rz_tgt / tgt if tgt else 0.0,
                att / games_played if games_played else 0.0,
                rec / games_played if games_played else 0.0,
            ])
        elif position in ['WR', 'TE']:
            rec = self.data_loader.safe_float_conversion(row.get('REC', 0)) or 1
            tgt = self.data_loader.safe_float_conversion(row.get('TGT', 0)) or 1
            yds = self.data_loader.safe_float_conversion(row.get('YDS', 0))
            yac = self.data_loader.safe_float_conversion(row.get('YAC', 0))
            air = self.data_loader.safe_float_conversion(row.get('AIR', 0))
            rz_tgt = self.data_loader.safe_float_conversion(row.get('RZ TGT', 0))
            pct_tm = self.data_loader.safe_float_conversion(row.get('% TM', 0))
            features.extend([
                yds / rec if rec else 0.0,
                rec / tgt if tgt else 0.0,
                yac / rec if rec else 0.0,
                air / rec if rec else 0.0,
                rz_tgt / tgt if tgt else 0.0,
                pct_tm,
                rec / games_played if games_played else 0.0,
                tgt / games_played if games_played else 0.0,
            ])
        return np.array(features) if features else None

    def _calculate_year_to_year_targets(self, adp_rank, next_row, position, player, year):
        """
        Calculate year-to-year targets using real ADP and fantasy points data.
        Improved target calculation for better model training.
        """
        finish_rank = self.data_loader.safe_float_conversion(next_row.get('RK', next_row.get('#', 999)))
        fantasy_points = self.data_loader.safe_float_conversion(next_row.get('FPTS', next_row.get('TTL', next_row.get('PTS', 0))))
        
        # Use real ADP rank if available, otherwise use a reasonable default
        if adp_rank == 999:  # Default ADP rank
            adp_rank = 50.0  # Middle ADP as fallback
        
        adp_diff = adp_rank - finish_rank
        
        # Improved breakout calculation with better scaling
        if adp_diff >= 50:
            breakout = 0.95 + min(0.05, (adp_diff - 50) / 50)
        elif adp_diff >= 30:
            breakout = 0.85 + (adp_diff - 30) / 20 * 0.1
        elif adp_diff >= 15:
            breakout = 0.70 + (adp_diff - 15) / 15 * 0.15
        elif adp_diff >= 5:
            breakout = 0.55 + (adp_diff - 5) / 10 * 0.15
        elif adp_diff >= -5:
            breakout = 0.45 + (adp_diff + 5) / 10 * 0.1
        elif adp_diff >= -20:
            breakout = 0.25 + (adp_diff + 20) / 15 * 0.2
        else:
            breakout = max(0.0, 0.25 + (adp_diff + 20) / 30 * 0.25)
        
        # Improved consistency calculation
        consistency = max(0.0, min(1.0, 1.0 - (abs(adp_diff) / 80)))
        
        # Improved positional value calculation
        positional_value = max(0.0, min(1.0, (80 - finish_rank) / 80))
        
        # Get games played and calculate injury risk
        games_played = self._get_games_played(player, position, year)
        season_length = 17 if year >= 2021 else 16
        games_missed = max(0, season_length - games_played)
        
        # Improved injury risk calculation
        if games_missed == 0:
            injury_risk = 0.1  # Very low risk
        elif games_missed <= 2:
            injury_risk = 0.2  # Low risk
        elif games_missed <= 5:
            injury_risk = 0.4  # Moderate risk
        elif games_missed <= 8:
            injury_risk = 0.6  # High risk
        else:
            injury_risk = 0.8  # Very high risk
        
        return {
            'breakout': breakout,
            'consistency': consistency,
            'positional_value': positional_value,
            'injury_risk': injury_risk
        }

    def _get_games_played(self, player: str, position: str, year: int) -> float:
        """Get games played for a player in a specific year."""
        if position in self.data_loader.advanced_data and year in self.data_loader.advanced_data[position]:
            df = self.data_loader.advanced_data[position][year]
            player_col = 'Player' if 'Player' in df.columns else 'PLAYER'
            player_mask = df[player_col].str.contains(player.split()[0], case=False, na=False)
            if player_mask.any():
                player_row = df[player_mask].iloc[0]
                return self.data_loader.safe_float_conversion(player_row.get('G', 0))
        return 0.0

    def _get_player_age(self, player: str, position: str) -> Optional[float]:
        """Get player age from the age data."""
        # Try to find the player in the age data
        for player_name, age in self.player_age_data.items():
            if player.lower() in player_name.lower() or player_name.lower() in player.lower():
                return age
        return None

    def _train_multiple_models(self, X: np.ndarray, y: np.ndarray, target_name: str, position: str) -> Tuple[Any, float]:
        best_model = None
        best_score = -1.0
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[f"{target_name}_{position}"] = scaler
        for model_name, config in self.model_configs.items():
            try:
                print(f"      Testing {model_name}...")
                model = config['model']()
                grid_search = GridSearchCV(model, config['params'], cv=3, scoring='r2', n_jobs=-1, verbose=0)
                grid_search.fit(X_train_scaled, y_train)
                y_pred = grid_search.predict(X_test_scaled)
                r2 = r2_score(y_test, y_pred)
                print(f"        {model_name} R²: {r2:.3f}")
                if r2 > best_score:
                    best_score = r2
                    best_model = grid_search.best_estimator_
            except Exception as e:
                print(f"        {model_name} failed: {e}")
                continue
        return best_model, best_score

    def generate_predictions(self, player: str, position: str, features: np.ndarray, debug_mode: bool = False) -> Dict[str, float]:
        predictions = {}
        for model_type in ML_MODEL_TYPES:
            model_key = f"{model_type}_{position}"
            if model_key not in self.best_models:
                predictions[model_type] = DEFAULT_SCORE
                continue
            model = self.best_models[model_key]
            scaler_key = f"{model_type}_{position}"
            if scaler_key not in self.scalers:
                predictions[model_type] = DEFAULT_SCORE
                continue
            scaler = self.scalers[scaler_key]
            confidence = self.model_confidence.get(model_key, 0.5)
            try:
                features_scaled = scaler.transform(features.reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
                min_confidence = 0.3
                effective_confidence = max(confidence, min_confidence)
                weighted_prediction = prediction * effective_confidence
                # Remove minimum threshold to allow lower predictions
                final_prediction = weighted_prediction
                predictions[model_type] = max(0.0, min(1.0, final_prediction))
                if debug_mode:
                    print(f"  {model_type}: {prediction:.3f} (confidence: {confidence:.3f})")
            except Exception as e:
                if debug_mode:
                    print(f"  {model_type}: Error - {e}")
                predictions[model_type] = DEFAULT_SCORE
        return predictions