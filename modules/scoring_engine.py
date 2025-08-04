"""
Scoring Engine Module for Fantasy Football Analyzer

Handles all scoring calculations and the Universal Scoring System.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from config.constants import (
    SCORING_WEIGHTS, PROFILE_WEIGHTS, METRIC_CATEGORIES,
    KEY_METRICS, DEFAULT_SCORE, POSITIONS
)


class ScoringEngine:
    """
    Handles all scoring calculations using the Universal Scoring System.
    
    The Universal Scoring System categorizes metrics into:
    - Volume Metrics (20% weight): Raw production numbers
    - Efficiency Metrics (35% weight): Per-attempt and per-game efficiency
    - Explosiveness Metrics (25% weight): Big-play potential and ceiling
    - Opportunity Metrics (15% weight): Usage and role indicators
    - Negative Metrics (-5% weight): Penalties for turnovers and inefficiency
    """
    
    def __init__(self, data_loader):
        """
        Initialize the scoring engine.
        
        Args:
            data_loader: DataLoader instance for accessing data
        """
        self.data_loader = data_loader
        
        # Metric descriptions for better analysis
        self.metric_descriptions = {
            # QB Metrics
            'PCT': 'Completion Percentage - QB accuracy',
            'Y/A': 'Yards per Attempt - QB efficiency',
            'AIR': 'Air Yards - Downfield passing volume',
            'AIR/A': 'Air Yards per Attempt - QB downfield aggression',
            '10+ YDS': '10+ Yard Completions - Big play ability',
            '20+ YDS': '20+ Yard Completions - Explosive plays',
            'PKT TIME': 'Pocket Time - Protection quality',
            'RTG': 'Passer Rating - Overall QB efficiency',
            'SACK': 'Sacks Taken - Protection issues',
            
            # RB Metrics
            'Y/ATT': 'Yards per Attempt - RB efficiency',
            'YACON/ATT': 'Yards After Contact per Attempt - Power/elusiveness',
            'BRKTKL': 'Break Tackles - Elusiveness',
            'REC': 'Receptions - Receiving volume',
            'TGT': 'Targets - Receiving opportunity',
            'RZ TGT': 'Red Zone Targets - Goal line opportunity',
            
            # WR/TE Metrics
            'Y/R': 'Yards per Reception - Efficiency',
            'YAC/R': 'Yards After Catch per Reception - YAC ability',
            'YACON/R': 'Yards After Contact per Reception - Power',
            '% TM': 'Team Target Share - Usage percentage',
            'CATCHABLE': 'Catchable Targets - Quality of targets'
        }
    
    def calculate_advanced_score_universal(self, player_row, position: str) -> Tuple[float, Dict]:
        """
        Calculate advanced metrics score using the Universal Scoring System.
        
        Args:
            player_row: Player's advanced metrics data
            position: Player position (QB, RB, WR, TE)
            
        Returns:
            Tuple of (score, breakdown_dict)
        """
        if position not in KEY_METRICS:
            return DEFAULT_SCORE, {}
        
        key_metrics = KEY_METRICS[position]
        category_scores = {}
        category_breakdowns = {}
        
        # Calculate scores for each category
        for category, weight in SCORING_WEIGHTS.items():
            category_metrics = self._get_category_metrics(position, category)
            category_score = 0.0
            category_breakdown = {}
            
            for metric in category_metrics:
                if metric in player_row.index and not pd.isna(player_row[metric]):
                    raw_value = self.clean_numeric_value_universal(player_row[metric])
                    normalized_score = self.normalize_metric_universal(raw_value, metric, category)
                    
                    category_score += normalized_score
                    category_breakdown[metric] = {
                        'raw_value': raw_value,
                        'normalized_score': normalized_score
                    }
            
            # Average the category score
            if category_breakdown:
                category_score /= len(category_breakdown)
            
            category_scores[category] = category_score
            category_breakdowns[category] = category_breakdown
        
        # Calculate weighted final score
        final_score = 0.0
        for category, score in category_scores.items():
            final_score += score * SCORING_WEIGHTS[category]
        
        # Ensure score is within 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        breakdown = {
            'category_scores': category_scores,
            'category_breakdowns': category_breakdowns,
            'final_score': final_score
        }
        
        return final_score, breakdown
    
    def _get_category_metrics(self, position: str, category: str) -> List[str]:
        """Get metrics that belong to a specific category for a position."""
        if category not in METRIC_CATEGORIES:
            return []
        
        category_metrics = METRIC_CATEGORIES[category]
        position_metrics = KEY_METRICS.get(position, [])
        
        # Return intersection of category metrics and position metrics
        return [metric for metric in category_metrics if metric in position_metrics]
    
    def clean_numeric_value_universal(self, value) -> float:
        """
        Clean and convert a value to float for universal scoring.
        
        Args:
            value: Raw value to convert
            
        Returns:
            Cleaned float value
        """
        if pd.isna(value):
            return 0.0
        
        try:
            # Handle percentage strings
            if isinstance(value, str) and '%' in value:
                return float(value.replace('%', '')) / 100.0
            
            # Handle other string formats
            if isinstance(value, str):
                # Remove common non-numeric characters
                cleaned = value.replace(',', '').replace('$', '').strip()
                return float(cleaned)
            
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def normalize_metric_universal(self, raw_value: float, metric_name: str, category: str) -> float:
        """
        Normalize a metric value to 0-10 scale based on category-specific ranges.
        
        Args:
            raw_value: Raw metric value
            metric_name: Name of the metric
            category: Category of the metric
            
        Returns:
            Normalized score (0-10)
        """
        if raw_value == 0.0:
            return 0.0
        
        # Define normalization ranges for each category
        normalization_ranges = {
            'volume': {
                'YDS': (0, 5000),      # Passing yards
                'ATT': (0, 400),        # Rushing attempts
                'REC': (0, 150),        # Receptions
                'TGT': (0, 200),        # Targets
                'TD': (0, 50),          # Touchdowns
                'CMP': (0, 500),        # Completions
            },
            'efficiency': {
                'Y/A': (5.0, 10.0),    # Yards per attempt
                'Y/R': (8.0, 20.0),    # Yards per reception
                'PCT': (0.50, 0.75),   # Completion percentage
                'RTG': (70.0, 120.0),  # Passer rating
                'YAC/R': (2.0, 8.0),   # Yards after catch per reception
                'YACON/R': (1.0, 5.0), # Yards after contact per reception
                'YACON/ATT': (1.0, 4.0), # Yards after contact per attempt
            },
            'explosiveness': {
                '10+ YDS': (0, 100),   # 10+ yard plays
                '20+ YDS': (0, 50),    # 20+ yard plays
                '40+ YDS': (0, 20),    # 40+ yard plays
                'BRKTKL': (0, 50),     # Broken tackles
                'AIR/A': (5.0, 12.0),  # Air yards per attempt
            },
            'opportunity': {
                'TGT': (0, 200),       # Targets
                'RZ TGT': (0, 30),     # Red zone targets
                '% TM': (0.05, 0.30),  # Team target share
                'CATCHABLE': (0, 200), # Catchable targets
                'PKT TIME': (2.0, 3.5), # Pocket time
            },
            'negative': {
                'SACK': (0, 50),       # Sacks (lower is better)
                'INT': (0, 20),        # Interceptions (lower is better)
                'FUM': (0, 10),        # Fumbles (lower is better)
                'DROP': (0, 15),       # Drops (lower is better)
            }
        }
        
        # Get range for this metric and category
        ranges = normalization_ranges.get(category, {})
        metric_range = ranges.get(metric_name, (0, 100))  # Default range
        
        min_val, max_val = metric_range
        
        # Handle negative metrics (lower is better)
        if category == 'negative':
            # Invert the scale for negative metrics
            if raw_value <= min_val:
                normalized = 10.0
            elif raw_value >= max_val:
                normalized = 0.0
            else:
                normalized = 10.0 - ((raw_value - min_val) / (max_val - min_val)) * 10.0
        else:
            # Normal positive metrics
            if raw_value <= min_val:
                normalized = 0.0
            elif raw_value >= max_val:
                normalized = 10.0
            else:
                normalized = ((raw_value - min_val) / (max_val - min_val)) * 10.0
        
        return max(0.0, min(10.0, normalized))
    
    def calculate_overall_profile_score(self, historical_perf: Dict[str, float], 
                                      advanced_metrics: Dict[str, Any], 
                                      ml_predictions: Dict[str, float], 
                                      injury_profile: Dict[str, Any], 
                                      debug_mode: bool = False,
                                      position: str = None, 
                                      player_row: Any = None, 
                                      player_name: str = "unknown") -> Dict[str, float]:
        """
        Calculate overall player profile score using weighted components.
        
        Args:
            historical_perf: Historical performance data
            advanced_metrics: Advanced metrics data
            ml_predictions: ML model predictions
            injury_profile: Injury profile data
            debug_mode: Enable debug output
            position: Player position
            player_row: Player's advanced metrics row
            player_name: Player name for logging
            
        Returns:
            Dictionary with component scores and final score
        """
        # Calculate individual component scores
        historical_score = self._calculate_historical_score(historical_perf)
        advanced_score = self._calculate_advanced_metrics_score(advanced_metrics, position, player_row)
        ml_score = self._calculate_ml_predictions_score(ml_predictions, player_name, position)
        injury_score = self._calculate_injury_profile_score(injury_profile, player_name, position)
        
        # Apply weights
        final_score = (
            historical_score * PROFILE_WEIGHTS['historical'] +
            advanced_score * PROFILE_WEIGHTS['advanced'] +
            ml_score * PROFILE_WEIGHTS['ml_predictions'] +
            injury_score * PROFILE_WEIGHTS['injury']
        )
        
        # Ensure final score is within 0-10 range
        final_score = max(0.0, min(10.0, final_score))
        
        scores = {
            'historical_score': historical_score,
            'advanced_score': advanced_score,
            'ml_score': ml_score,
            'injury_score': injury_score,
            'final_score': final_score
        }
        
        if debug_mode:
            print(f"\nDEBUG - {player_name} ({position}) Scoring:")
            print(f"  Historical: {historical_score:.2f} (weight: {PROFILE_WEIGHTS['historical']:.2f})")
            print(f"  Advanced: {advanced_score:.2f} (weight: {PROFILE_WEIGHTS['advanced']:.2f})")
            print(f"  ML: {ml_score:.2f} (weight: {PROFILE_WEIGHTS['ml_predictions']:.2f})")
            print(f"  Injury: {injury_score:.2f} (weight: {PROFILE_WEIGHTS['injury']:.2f})")
            print(f"  Final: {final_score:.2f}")
        
        return scores
    
    def _calculate_historical_score(self, historical_perf: Dict[str, float]) -> float:
        """
        Calculate historical performance component score (0-10) - ONLY REAL DATA.
        
        Args:
            historical_perf: Historical performance data
            
        Returns:
            Historical score (0-10)
        """
        # REQUIRE real historical data - no fallbacks
        if not historical_perf or historical_perf.get('years_of_data', 0) == 0:
            self.data_loader._log_missing_data("unknown", "unknown", "no_historical_data", 
                                             [f"historical_perf: {historical_perf}"])
            return DEFAULT_SCORE  # Return 0 if no historical data, no fallback
        
        # Position-specific ADP differential scoring (negative is better - outperforming ADP)
        adp_differential = historical_perf.get('adp_differential', 0)
        peak_season = historical_perf.get('peak_season', 0)
        consistency = historical_perf.get('consistency', 0)
        experience = historical_perf.get('years_of_data', 0)
        
        # Calculate component scores
        adp_score = max(0, min(10, (25 + adp_differential) / 5))  # -25 to +25 range -> 0-10
        peak_score = max(0, min(10, peak_season / 10))  # 0-100 range -> 0-10
        consistency_score = max(0, min(10, consistency / 10))  # 0-100 range -> 0-10
        experience_score = max(0, min(10, experience / 3))  # 0-3+ years -> 0-10
        
        # Weighted average
        historical_score = (
            adp_score * 0.4 +
            peak_score * 0.3 +
            consistency_score * 0.2 +
            experience_score * 0.1
        )
        
        return max(0.0, min(10.0, historical_score))
    
    def _calculate_advanced_metrics_score(self, advanced_metrics: Dict[str, Any], 
                                        position: str = None, 
                                        player_row: Any = None) -> float:
        """
        Calculate advanced metrics component score using universal system ONLY - NO FALLBACKS.
        
        Args:
            advanced_metrics: Advanced metrics data
            position: Player position
            player_row: Player's advanced metrics row
            
        Returns:
            Advanced metrics score (0-10)
        """
        # REQUIRE player_row and position for universal system
        if player_row is None or position is None:
            self.data_loader._log_missing_data("unknown", position or "unknown", "player_row_or_position_missing", 
                                             [f"player_row: {player_row is not None}", f"position: {position}"])
            return DEFAULT_SCORE  # Return 0 if missing required data, no fallback
        
        try:
            universal_score, breakdown = self.calculate_advanced_score_universal(player_row, position)
            return universal_score
        except Exception as e:
            self.data_loader._log_missing_data("unknown", position, "universal_scoring_error", [str(e)])
            return DEFAULT_SCORE  # Return 0 on error, no fallback
    
    def _calculate_ml_predictions_score(self, ml_predictions: Dict[str, float], 
                                      player_name: str = "unknown", 
                                      position: str = "unknown") -> float:
        """
        Calculate ML predictions component score (0-10).
        
        Args:
            ml_predictions: ML model predictions
            player_name: Player name for logging
            position: Player position for logging
            
        Returns:
            ML predictions score (0-10)
        """
        if not ml_predictions:
            self.data_loader._log_missing_data(player_name, position, "no_ml_predictions", 
                                             ["ml_predictions"])
            return DEFAULT_SCORE
        
        # Calculate weighted average of ML predictions with improved handling
        total_score = 0.0
        total_weight = 0.0
        valid_predictions = 0
        
        for prediction_type, score in ml_predictions.items():
            if score is not None and not pd.isna(score):
                # Apply minimum threshold to prevent extremely low scores
                min_threshold = 0.1  # Minimum 10% score
                boosted_score = max(score, min_threshold)
                
                # Normalize score to 0-10 range with boost for low scores
                if boosted_score < 0.3:
                    # Boost very low scores to make them more meaningful
                    boosted_score = boosted_score * 2.0
                
                normalized_score = max(0.0, min(10.0, boosted_score * 10.0))
                weight = 1.0  # Equal weight for all predictions
                
                total_score += normalized_score * weight
                total_weight += weight
                valid_predictions += 1
        
        if total_weight == 0:
            return DEFAULT_SCORE
        
        # Apply additional boost if we have multiple valid predictions
        base_score = total_score / total_weight
        
        # Boost score if we have good prediction coverage
        if valid_predictions >= 2:
            base_score = min(10.0, base_score * 1.2)  # 20% boost
        
        return max(0.0, min(10.0, base_score))
    
    def _calculate_injury_profile_score(self, injury_profile: Dict[str, Any], 
                                      player_name: str = "unknown", 
                                      position: str = "unknown") -> float:
        """
        Calculate injury profile component score (0-10).
        
        Args:
            injury_profile: Injury profile data
            player_name: Player name for logging
            position: Player position for logging
            
        Returns:
            Injury profile score (0-10)
        """
        if not injury_profile:
            self.data_loader._log_missing_data(player_name, position, "no_injury_profile", 
                                             ["injury_profile"])
            return DEFAULT_SCORE
        
        # Extract injury risk factors
        injury_risk = injury_profile.get('injury_risk', 0.5)
        games_missed = injury_profile.get('games_missed', 0)
        injury_history = injury_profile.get('injury_history', [])
        
        # Calculate injury score (lower risk = higher score)
        risk_score = max(0.0, min(10.0, (1.0 - injury_risk) * 10.0))
        
        # Penalize for games missed
        games_penalty = min(5.0, games_missed * 0.5)  # Max 5 point penalty
        games_score = max(0.0, 10.0 - games_penalty)
        
        # Penalize for injury history
        history_penalty = min(3.0, len(injury_history) * 0.5)  # Max 3 point penalty
        history_score = max(0.0, 10.0 - history_penalty)
        
        # Weighted average
        injury_score = (
            risk_score * 0.5 +
            games_score * 0.3 +
            history_score * 0.2
        )
        
        return max(0.0, min(10.0, injury_score))
    
    def _calculate_non_linear_aggregated_score(self, hist_score: float, adv_score: float, 
                                             ml_score: float, injury_score: float,
                                             historical_weight: float, advanced_weight: float,
                                             ml_weight: float, injury_weight: float) -> Tuple[float, Dict]:
        """
        Calculate non-linear aggregated score with component breakdown.
        
        Args:
            hist_score: Historical performance score
            adv_score: Advanced metrics score
            ml_score: ML predictions score
            injury_score: Injury profile score
            historical_weight: Weight for historical score
            advanced_weight: Weight for advanced score
            ml_weight: Weight for ML score
            injury_weight: Weight for injury score
            
        Returns:
            Tuple of (final_score, breakdown_dict)
        """
        # Apply non-linear transformations
        hist_transformed = np.tanh(hist_score / 5.0) * 5.0  # Compress to 0-5 range
        adv_transformed = np.tanh(adv_score / 5.0) * 5.0
        ml_transformed = np.tanh(ml_score / 5.0) * 5.0
        injury_transformed = np.tanh(injury_score / 5.0) * 5.0
        
        # Calculate weighted sum
        final_score = (
            hist_transformed * historical_weight +
            adv_transformed * advanced_weight +
            ml_transformed * ml_weight +
            injury_transformed * injury_weight
        )
        
        # Normalize to 0-10 range
        final_score = max(0.0, min(10.0, final_score * 2.0))
        
        breakdown = {
            'historical': {
                'raw_score': hist_score,
                'transformed_score': hist_transformed,
                'weight': historical_weight
            },
            'advanced': {
                'raw_score': adv_score,
                'transformed_score': adv_transformed,
                'weight': advanced_weight
            },
            'ml_predictions': {
                'raw_score': ml_score,
                'transformed_score': ml_transformed,
                'weight': ml_weight
            },
            'injury': {
                'raw_score': injury_score,
                'transformed_score': injury_transformed,
                'weight': injury_weight
            },
            'final_score': final_score
        }
        
        return final_score, breakdown
    
    def _calculate_z_score_based_score(self, value: float, position_values: pd.Series, 
                                     metric_name: str = "") -> Tuple[float, float]:
        """
        Calculate z-score based score for a metric.
        
        Args:
            value: Raw metric value
            position_values: Series of values for the position
            metric_name: Name of the metric for logging
            
        Returns:
            Tuple of (z_score, normalized_score)
        """
        if position_values.empty or pd.isna(value):
            return 0.0, 0.0
        
        # Calculate z-score
        mean_val = position_values.mean()
        std_val = position_values.std()
        
        if std_val == 0:
            return 0.0, 5.0  # Neutral score if no variance
        
        z_score = (value - mean_val) / std_val
        
        # Convert z-score to 0-10 scale
        # Z-score of 0 = 5.0, Z-score of ±2 = 0.0/10.0
        normalized_score = 5.0 + (z_score * 2.5)
        normalized_score = max(0.0, min(10.0, normalized_score))
        
        return z_score, normalized_score 