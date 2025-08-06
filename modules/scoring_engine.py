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
                    normalized_score = self.normalize_metric_universal(raw_value, metric, category, position, player_row)
                    
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
    
    def normalize_metric_universal(self, raw_value: float, metric_name: str, category: str, position: str = None, player_row: Any = None) -> float:
        """
        Normalize a metric value to 0-10 scale using position-specific min-max scaling.
        
        Args:
            raw_value: Raw metric value
            metric_name: Name of the metric
            category: Category of the metric
            position: Player position for position-specific normalization
            player_row: Player's data row for context
            
        Returns:
            Normalized score (0-10)
        """
        if raw_value == 0.0:
            return 0.0
        
        # If we have position and data, use position-specific normalization
        if position and self.data_loader.advanced_data and position in self.data_loader.advanced_data:
            return self._normalize_metric_position_specific(raw_value, metric_name, category, position)
        else:
            # Fallback to universal ranges
            return self._normalize_metric_universal_fallback(raw_value, metric_name, category)

    def _normalize_metric_position_specific(self, raw_value: float, metric_name: str, category: str, position: str) -> float:
        """
        Normalize metric using position-specific min-max scaling.
        
        Args:
            raw_value: Raw metric value
            metric_name: Name of the metric
            category: Category of the metric
            position: Player position
            
        Returns:
            Normalized score (0-10)
        """
        # Get all players for this position in the current year
        current_year = max(self.data_loader.years)
        if position not in self.data_loader.advanced_data or current_year not in self.data_loader.advanced_data[position]:
            return self._normalize_metric_universal_fallback(raw_value, metric_name, category)
        
        df = self.data_loader.advanced_data[position][current_year]
        
        # Get all values for this metric in this position
        if metric_name not in df.columns:
            return self._normalize_metric_universal_fallback(raw_value, metric_name, category)
        
        # Filter out NaN values and convert to float
        position_values = df[metric_name].dropna()
        # Convert all values to float, filtering out any that can't be converted
        float_values = []
        for val in position_values:
            try:
                float_val = self.data_loader.safe_float_conversion(val)
                if float_val is not None:
                    float_values.append(float_val)
            except:
                continue
        
        if len(float_values) < 5:  # Need minimum sample size
            return self._normalize_metric_universal_fallback(raw_value, metric_name, category)
        
        # Calculate position-specific min and max
        min_val = min(float_values)
        max_val = max(float_values)
        
        # Handle edge cases
        if min_val == max_val:
            return 5.0  # Neutral score if no variance
        
        # Apply metric-specific weighting
        weight = self._get_metric_weight(metric_name, category, position)
        
        # Normalize with position-specific range
        if category == 'negative':
            # Invert the scale for negative metrics (lower is better)
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
        
        # Apply metric weighting
        weighted_score = normalized * weight
        
        return max(0.0, min(10.0, weighted_score))

    def _get_metric_weight(self, metric_name: str, category: str, position: str) -> float:
        """
        Get weight for a specific metric based on its importance.
        
        Args:
            metric_name: Name of the metric
            category: Category of the metric
            position: Player position
            
        Returns:
            Weight multiplier (1.0 = normal, >1.0 = more important, <1.0 = less important)
        """
        # Define metric importance weights by position - including ALL metrics from constants
        metric_weights = {
            'QB': {
                # High importance metrics (1.3x weight)
                'PCT': 1.3,      # Completion percentage
                'Y/A': 1.3,      # Yards per attempt
                'RTG': 1.3,      # Passer rating
                'COMP': 1.2,     # Completions
                'YDS': 1.2,      # Passing yards
                'ATT': 1.2,      # Attempts
                'AIR/A': 1.2,    # Air yards per attempt
                'AIR': 1.1,      # Air yards
                'RZ ATT': 1.1,   # Red zone attempts
                
                # Explosive play metrics (1.2x weight)
                '10+ YDS': 1.2,  # 10+ yard plays
                '20+ YDS': 1.2,  # 20+ yard plays
                '30+ YDS': 1.2,  # 30+ yard plays
                '40+ YDS': 1.2,  # 40+ yard plays
                '50+ YDS': 1.2,  # 50+ yard plays
                
                # Pressure metrics (1.0x weight)
                'PKT TIME': 1.0, # Pocket time
                'SACK': 0.9,     # Sacks (negative)
                'KNCK': 0.9,     # Knockdowns (negative)
                'HRRY': 0.9,     # Hurries (negative)
                'BLITZ': 1.0,    # Times blitzed
                
                # Accuracy metrics (1.1x weight)
                'POOR': 0.8,     # Poor passes (negative)
                'DROP': 0.8,     # Drops by receivers (negative)
            },
            'RB': {
                # High importance metrics (1.3x weight)
                'Y/ATT': 1.3,    # Yards per attempt
                'YDS': 1.3,      # Rushing yards
                'ATT': 1.2,      # Rushing attempts
                'YACON/ATT': 1.2, # Yards after contact per attempt
                'YBCON/ATT': 1.2, # Yards before contact per attempt
                'BRKTKL': 1.2,   # Broken tackles
                'REC': 1.1,      # Receptions
                'TGT': 1.1,      # Targets
                'RZ TGT': 1.1,   # Red zone targets
                
                # Explosive play metrics (1.2x weight)
                '10+ YDS': 1.2,  # 10+ yard runs
                '20+ YDS': 1.2,  # 20+ yard runs
                '30+ YDS': 1.2,  # 30+ yard runs
                '40+ YDS': 1.2,  # 40+ yard runs
                '50+ YDS': 1.2,  # 50+ yard runs
                'LNG TD': 1.1,   # Longest touchdown
                'LNG': 1.1,      # Longest run
                
                # Power/elusiveness metrics (1.1x weight)
                'YACON': 1.1,    # Yards after contact
                'YBCON': 1.0,    # Yards before contact
                
                # Negative metrics (0.8x weight)
                'TK LOSS': 0.8,  # Tackled for loss (negative)
                'TK LOSS YDS': 0.8, # Tackled for loss yards (negative)
                'FUM': 0.8,      # Fumbles (negative)
            },
            'WR': {
                # High importance metrics (1.3x weight)
                'Y/R': 1.3,      # Yards per reception
                'YDS': 1.3,      # Receiving yards
                'REC': 1.2,      # Receptions
                'TGT': 1.2,      # Targets
                'YAC/R': 1.2,    # Yards after catch per reception
                'YACON/R': 1.2,  # Yards after contact per reception
                'YBC/R': 1.1,    # Yards before catch per reception
                '% TM': 1.2,     # Team target share
                'CATCHABLE': 1.1, # Catchable targets
                'RZ TGT': 1.1,   # Red zone targets
                
                # Explosive play metrics (1.2x weight)
                '10+ YDS': 1.2,  # 10+ yard receptions
                '20+ YDS': 1.2,  # 20+ yard receptions
                '30+ YDS': 1.2,  # 30+ yard receptions
                '40+ YDS': 1.2,  # 40+ yard receptions
                '50+ YDS': 1.2,  # 50+ yard receptions
                'LNG': 1.1,      # Longest reception
                
                # Playmaking metrics (1.1x weight)
                'YAC': 1.1,      # Yards after catch
                'YACON': 1.1,    # Yards after contact
                'YBC': 1.0,      # Yards before catch
                'AIR': 1.0,      # Air yards
                'AIR/R': 1.0,    # Air yards per reception
                'BRKTKL': 1.1,   # Broken tackles
                
                # Negative metrics (0.8x weight)
                'DROP': 0.8,     # Drops (negative)
                'FUM': 0.8,      # Fumbles (negative)
            },
            'TE': {
                # High importance metrics (1.3x weight)
                'Y/R': 1.3,      # Yards per reception
                'YDS': 1.3,      # Receiving yards
                'REC': 1.2,      # Receptions
                'TGT': 1.2,      # Targets
                'YAC/R': 1.2,    # Yards after catch per reception
                'YACON/R': 1.2,  # Yards after contact per reception
                'YBC/R': 1.1,    # Yards before catch per reception
                '% TM': 1.2,     # Team target share
                'CATCHABLE': 1.1, # Catchable targets
                'RZ TGT': 1.1,   # Red zone targets
                
                # Explosive play metrics (1.2x weight)
                '10+ YDS': 1.2,  # 10+ yard receptions
                '20+ YDS': 1.2,  # 20+ yard receptions
                '30+ YDS': 1.2,  # 30+ yard receptions
                '40+ YDS': 1.2,  # 40+ yard receptions
                '50+ YDS': 1.2,  # 50+ yard receptions
                'LNG': 1.1,      # Longest reception
                
                # Playmaking metrics (1.1x weight)
                'YAC': 1.1,      # Yards after catch
                'YACON': 1.1,    # Yards after contact
                'YBC': 1.0,      # Yards before catch
                'AIR': 1.0,      # Air yards
                'AIR/R': 1.0,    # Air yards per reception
                'BRKTKL': 1.1,   # Broken tackles
                
                # Negative metrics (0.8x weight)
                'DROP': 0.8,     # Drops (negative)
                'FUM': 0.8,      # Fumbles (negative)
            }
        }
        
        # Get weights for this position
        position_weights = metric_weights.get(position, {})
        
        # Return weight for this metric, or 1.0 if not specified
        return position_weights.get(metric_name, 1.0)

    def _normalize_metric_universal_fallback(self, raw_value: float, metric_name: str, category: str) -> float:
        """
        Fallback normalization using universal ranges when position-specific data is not available.
        
        Args:
            raw_value: Raw metric value
            metric_name: Name of the metric
            category: Category of the metric
            
        Returns:
            Normalized score (0-10)
        """
        if raw_value == 0.0:
            return 0.0
        
        # Define normalization ranges for each category (fallback) - including ALL metrics
        normalization_ranges = {
            'volume': {
                'YDS': (0, 5000),      # Passing/receiving yards
                'ATT': (0, 400),        # Rushing attempts
                'REC': (0, 150),        # Receptions
                'TGT': (0, 200),        # Targets
                'TD': (0, 50),          # Touchdowns
                'COMP': (0, 500),       # Completions
                'AIR': (0, 3000),       # Air yards
            },
            'efficiency': {
                'Y/A': (5.0, 10.0),    # Yards per attempt
                'Y/R': (8.0, 20.0),    # Yards per reception
                'PCT': (0.50, 0.75),   # Completion percentage
                'RTG': (70.0, 120.0),  # Passer rating
                'YAC/R': (2.0, 8.0),   # Yards after catch per reception
                'YACON/R': (1.0, 5.0), # Yards after contact per reception
                'YACON/ATT': (1.0, 4.0), # Yards after contact per attempt
                'YBCON/ATT': (1.0, 4.0), # Yards before contact per attempt
                'YBC/R': (1.0, 8.0),   # Yards before catch per reception
                'AIR/A': (5.0, 12.0),  # Air yards per attempt
                'AIR/R': (5.0, 15.0),  # Air yards per reception
            },
            'explosiveness': {
                '10+ YDS': (0, 100),   # 10+ yard plays
                '20+ YDS': (0, 50),    # 20+ yard plays
                '30+ YDS': (0, 30),    # 30+ yard plays
                '40+ YDS': (0, 20),    # 40+ yard plays
                '50+ YDS': (0, 10),    # 50+ yard plays
                'BRKTKL': (0, 50),     # Broken tackles
                'LNG TD': (0, 10),     # Longest touchdown
                'LNG': (0, 100),       # Longest play
            },
            'opportunity': {
                'TGT': (0, 200),       # Targets
                'RZ TGT': (0, 30),     # Red zone targets
                'RZ ATT': (0, 50),     # Red zone attempts
                '% TM': (0.05, 0.30),  # Team target share
                'CATCHABLE': (0, 200), # Catchable targets
                'PKT TIME': (2.0, 3.5), # Pocket time
            },
            'negative': {
                'SACK': (0, 50),       # Sacks (lower is better)
                'INT': (0, 20),        # Interceptions (lower is better)
                'FUM': (0, 10),        # Fumbles (lower is better)
                'DROP': (0, 15),       # Drops (lower is better)
                'KNCK': (0, 20),       # Knockdowns (lower is better)
                'HRRY': (0, 30),       # Hurries (lower is better)
                'POOR': (0, 50),       # Poor passes (lower is better)
                'TK LOSS': (0, 20),    # Tackled for loss (lower is better)
                'TK LOSS YDS': (0, 100), # Tackled for loss yards (lower is better)
                'BLITZ': (0, 100),     # Times blitzed (context dependent)
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
        historical_score = self._calculate_enhanced_historical_score(historical_perf, position)
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
    
    def _calculate_position_specific_peak_score(self, peak_ppg: float, position: str) -> float:
        """
        Calculate position-specific peak score based on peak PPG with proper scaling.
        USED FOR ML MODELS ONLY - to help predict breakouts.
        
        Args:
            peak_ppg: Peak points per game
            position: Player position
            
        Returns:
            Position-specific peak score (0-10)
        """
        # Position-specific max PPG thresholds (based on elite performance)
        position_max_ppg = {
            'QB': 30.0,  # Elite QBs can average 30+ PPG
            'RB': 25.0,  # Elite RBs can average 25+ PPG
            'WR': 22.0,  # Elite WRs can average 22+ PPG
            'TE': 15.0,  # Elite TEs can average 15+ PPG
        }
        
        max_ppg = position_max_ppg.get(position, 20.0)  # Default to 20
        
        # Calculate peak score using the formula: min(10, (peak_ppg / position_max_ppg) * 10)
        peak_score = min(10.0, (peak_ppg / max_ppg) * 10.0)
        
        return max(0.0, peak_score)

    def _calculate_position_specific_avg_ppg_score(self, avg_ppg: float, position: str) -> float:
        """
        Calculate position-specific average PPG score for historical scoring.
        
        Args:
            avg_ppg: Average points per game
            position: Player position
            
        Returns:
            Position-specific average PPG score (0-10)
        """
        # Position-specific average PPG thresholds (based on good performance)
        position_avg_ppg = {
            'QB': 20.0,  # Good QBs average 20+ PPG
            'RB': 15.0,  # Good RBs average 15+ PPG
            'WR': 12.0,  # Good WRs average 12+ PPG
            'TE': 8.0,   # Good TEs average 8+ PPG
        }
        
        avg_threshold = position_avg_ppg.get(position, 12.0)  # Default to 12
        
        # Calculate average PPG score using the formula: min(10, (avg_ppg / position_avg_ppg) * 10)
        avg_score = min(10.0, (avg_ppg / avg_threshold) * 10.0)
        
        return max(0.0, avg_score)

    def _calculate_enhanced_historical_score(self, historical_perf: Dict[str, float], position: str = None) -> float:
        """
        Calculate enhanced historical performance component score (0-10) with position-specific scaling.
        
        Args:
            historical_perf: Enhanced historical performance data
            position: Player position for position-specific scaling
            
        Returns:
            Enhanced historical score (0-10)
        """
        # REQUIRE real historical data - no fallbacks
        if not historical_perf or historical_perf.get('years_of_data', 0) == 0:
            self.data_loader._log_missing_data("unknown", "unknown", "no_historical_data", 
                                             [f"historical_perf: {historical_perf}"])
            return DEFAULT_SCORE  # Return 0 if no historical data, no fallback
        
        # Enhanced component extraction - REQUIRE all data, no defaults
        adp_differential = historical_perf.get('adp_differential')
        avg_ppg = historical_perf.get('avg_ppg')  # NEW: Use average PPG instead of peak
        ppg_consistency = historical_perf.get('ppg_consistency')
        injury_adjustment = historical_perf.get('injury_adjustment')
        experience = historical_perf.get('years_of_data')
        
        # Validate all required data is present
        if adp_differential is None:
            self.data_loader._log_missing_data("unknown", "unknown", "adp_differential", 
                                             ["adp_differential"], "historical_perf")
            return DEFAULT_SCORE
        
        if avg_ppg is None:
            self.data_loader._log_missing_data("unknown", "unknown", "avg_ppg", 
                                             ["avg_ppg"], "historical_perf")
            return DEFAULT_SCORE
        
        if ppg_consistency is None:
            self.data_loader._log_missing_data("unknown", "unknown", "ppg_consistency", 
                                             ["ppg_consistency"], "historical_perf")
            return DEFAULT_SCORE
        
        if injury_adjustment is None:
            self.data_loader._log_missing_data("unknown", "unknown", "injury_adjustment", 
                                             ["injury_adjustment"], "historical_perf")
            return DEFAULT_SCORE
        
        if experience is None:
            self.data_loader._log_missing_data("unknown", "unknown", "years_of_data", 
                                             ["years_of_data"], "historical_perf")
            return DEFAULT_SCORE
        
        # Calculate enhanced component scores
        adp_score = max(0, min(10, (25 + adp_differential) / 5))  # -25 to +25 range -> 0-10
        avg_ppg_score = self._calculate_position_specific_avg_ppg_score(avg_ppg, position)  # NEW: Position-specific average PPG scaling
        ppg_consistency_score = max(0, min(10, ppg_consistency / 10))  # 0-100 range -> 0-10
        injury_score = injury_adjustment  # Already 0-10 scale
        experience_score = max(0, min(10, experience / 3))  # 0-3+ years -> 0-10
        
        # Enhanced weighted average - REMOVED peak score, added average PPG score
        historical_score = (
            adp_score * 0.30 +           # Same weight
            avg_ppg_score * 0.35 +       # NEW: Average PPG score (replaced peak score)
            ppg_consistency_score * 0.20 + # Same weight
            injury_score * 0.05 +        # Same weight
            experience_score * 0.10      # Same weight
        )
        
        return max(0.0, min(10.0, historical_score))

    def _calculate_historical_score(self, historical_perf: Dict[str, float]) -> float:
        """
        Calculate historical performance component score (0-10) - ONLY REAL DATA.
        DEPRECATED: Use _calculate_enhanced_historical_score instead.
        
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
        Calculate ML predictions component score (0-10) with improved normalization.
        REDUCED breakout weight in favor of PPG-focused metrics.
        
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
        
        # Calculate weighted average of ML predictions with REDUCED breakout weight
        total_score = 0.0
        total_weight = 0.0
        valid_predictions = 0
        
        # NEW: Adjusted weights - reduce breakout, increase PPG-focused metrics
        prediction_weights = {
            'breakout': 0.15,      # REDUCED from equal weight (was ~0.25)
            'consistency': 0.35,   # INCREASED - more PPG-focused
            'positional_value': 0.35,  # INCREASED - more PPG-focused
            'injury_risk': 0.15    # Same weight
        }
        
        for prediction_type, score in ml_predictions.items():
            if score is not None and not pd.isna(score):
                # Apply more aggressive non-linear normalization to spread scores
                # Formula: (raw_score^0.5) * 15 - 1
                # This maps: 0.1→3.7, 0.2→5.7, 0.3→7.2, 0.4→8.5, 0.5→9.6
                improved_score = (score ** 0.5) * 15 - 1
                improved_score = max(0.0, min(10.0, improved_score))
                
                # Apply confidence-based weighting with new weights
                weight = prediction_weights.get(prediction_type, 0.25)  # Default weight
                
                total_score += improved_score * weight
                total_weight += weight
                valid_predictions += 1
        
        if total_weight == 0:
            return DEFAULT_SCORE
        
        # Calculate base score
        base_score = total_score / total_weight
        
        # Apply additional boost if we have multiple valid predictions
        if valid_predictions >= 2:
            base_score = min(10.0, base_score * 1.1)  # 10% boost for multiple predictions
        
        return max(0.0, min(10.0, base_score))
    
    def _calculate_injury_profile_score(self, injury_profile: Dict[str, Any], 
                                      player_name: str = "unknown", 
                                      position: str = "unknown") -> float:
        """
        Calculate injury profile score using percentile-based normalization.
        
        NEW: Fallback to old system if no injury data is available.
        
        Component 1: 2025 Season Risk (50% weight)
        - Projected Games Missed (30% of total) - Percentile-based normalized
        - Risk Score (20% of total) - Direct mapping with improved spacing
        
        Component 2: Historical Pattern Context (30% weight)  
        - Injuries Per Season (20% of total) - Percentile-based normalized
        - Per Game Risk (10% of total) - Percentile-based normalized
        
        Component 3: Current Physical State (20% weight)
        - Durability Score (20% of total) - Percentile-based normalized
        """
        if not injury_profile:
            self.data_loader._log_missing_data(player_name, position, "no_injury_profile", 
                                             ["injury_profile"])
            return DEFAULT_SCORE
        
        # NEW: Check if we have real injury data or if this is a fallback
        has_real_injury_data = injury_profile.get('has_injury_data', False)
        
        # If no real injury data, use the old system based on games played
        if not has_real_injury_data:
            return self._calculate_legacy_injury_score(injury_profile, player_name, position)
        
        # Extract metrics for new system
        projected_games_missed = self._safe_float(injury_profile.get('projected_games_missed', 0))
        injury_risk_text = str(injury_profile.get('injury_risk', 'Medium Risk'))
        injuries_per_season = self._safe_float(injury_profile.get('injuries_per_season', 0))
        injury_risk_per_game = self._safe_float(injury_profile.get('injury_risk_per_game', 0))
        durability = self._safe_float(injury_profile.get('durability', 5))
        
        # Get position-specific data for percentile calculations
        position_data = self._get_position_injury_data(position)
        
        # Component 1: 2025 Season Risk (50% weight)
        # Projected Games Missed (30% of total) - Percentile-based normalized (lower is better)
        games_missed_score = self._percentile_normalize_metric(
            projected_games_missed, position_data['projected_games_missed'], inverse=True
        )
        
        # Risk Score (20% of total) - Direct mapping with improved spacing
        risk_score_raw = self._parse_injury_risk_text_improved(injury_risk_text)
        risk_score_normalized = (1.0 - risk_score_raw) * 10.0  # Convert to 0-10 scale
        
        season_risk = (games_missed_score * 0.6) + (risk_score_normalized * 0.4)  # 30% + 20% of total
        
        # Component 2: Historical Pattern Context (30% weight)
        # Injuries Per Season (20% of total) - Percentile-based normalized (lower is better)
        season_frequency_score = self._percentile_normalize_metric(
            injuries_per_season, position_data['injuries_per_season'], inverse=True
        )
        
        # Per Game Risk (10% of total) - Percentile-based normalized (lower is better)
        game_risk_score = self._percentile_normalize_metric(
            injury_risk_per_game, position_data['injury_risk_per_game'], inverse=True
        )
        
        historical_context = (season_frequency_score * 0.667) + (game_risk_score * 0.333)  # 20% + 10% of total
        
        # Component 3: Current Physical State (20% weight)
        # Durability Score (20% of total) - Percentile-based normalized (higher is better)
        durability_score = self._percentile_normalize_metric(
            durability, position_data['durability'], inverse=False
        )
        
        # Calculate final weighted score
        final_score = (
            season_risk * 0.50 +      # Component 1: 50%
            historical_context * 0.30 + # Component 2: 30%
            durability_score * 0.20    # Component 3: 20%
        )
        
        # Determine risk level based on final score
        if final_score >= 8.0:
            risk_level = "VERY LOW"
        elif final_score >= 6.0:
            risk_level = "LOW"
        elif final_score >= 4.0:
            risk_level = "MEDIUM"
        elif final_score >= 2.0:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        # DEBUG OUTPUT for validation
        print(f"DEBUG - {player_name} Injury Score: {final_score:.2f}/10, Risk Level: {risk_level}")
        print(f"  Season Risk: {season_risk:.2f} (Games: {games_missed_score:.2f}, Risk: {risk_score_normalized:.2f})")
        print(f"  Historical: {historical_context:.2f} (Freq: {season_frequency_score:.2f}, Game: {game_risk_score:.2f})")
        print(f"  Durability: {durability_score:.2f}")
        
        return max(0.0, min(10.0, final_score))
    
    def _calculate_legacy_injury_score(self, injury_profile: Dict[str, Any], 
                                     player_name: str, position: str) -> float:
        """
        Calculate injury score using the old system based on games played.
        Used as fallback when no injury data is available.
        
        Args:
            injury_profile: Injury profile data
            player_name: Player name for logging
            position: Player position
            
        Returns:
            Legacy injury score (0-10)
        """
        # Get games missed from the profile
        games_missed = injury_profile.get('games_missed', 0)
        
        # Calculate injury risk based on games missed (old system)
        if games_missed == 0:
            injury_score = 8.0  # Very low risk
        elif games_missed <= 2:
            injury_score = 6.0  # Low risk
        elif games_missed <= 5:
            injury_score = 4.0  # Moderate risk
        elif games_missed <= 8:
            injury_score = 2.0  # High risk
        else:
            injury_score = 0.5  # Very high risk
        
        # Apply age factor if available
        age = injury_profile.get('age')
        if age is not None:
            age_factor = self._calculate_extreme_age_factor(age)
            # Blend age factor with games missed (70% games, 30% age)
            injury_score = (injury_score * 0.7) + (age_factor * 0.3)
        
        # Determine risk level for debugging
        if injury_score >= 7.0:
            risk_level = "VERY LOW"
        elif injury_score >= 5.0:
            risk_level = "LOW"
        elif injury_score >= 3.0:
            risk_level = "MEDIUM"
        elif injury_score >= 1.0:
            risk_level = "HIGH"
        else:
            risk_level = "VERY HIGH"
        
        # DEBUG OUTPUT for legacy system
        print(f"DEBUG - {player_name} Legacy Injury Score: {injury_score:.2f}/10, Risk Level: {risk_level}")
        print(f"  Games Missed: {games_missed}, Age: {age}")
        
        return max(0.0, min(10.0, injury_score))
    
    def _safe_float(self, value, default=0.0):
        """Safely convert value to float, handling special cases."""
        if pd.isna(value) or value == '':
            return default
        
        # Handle "Injuries Per Season" with /yr suffix
        if isinstance(value, str) and '/yr' in value:
            try:
                return float(value.replace('/yr', ''))
            except:
                return default
        
        # Handle percentage strings
        if isinstance(value, str) and '%' in value:
            try:
                return float(value.replace('%', ''))
            except:
                return default
        
        try:
            return float(value)
        except:
            return default
    
    def _parse_injury_risk_text_improved(self, risk_text):
        """Convert injury risk text to numeric value using improved mapping."""
        risk_text = str(risk_text).strip()
        
        # Improved mapping with even spacing (lower numbers = lower risk)
        risk_score_mapping = {
            'Very Low Risk': 0.15,    # Very Low Risk
            'Low Risk': 0.30,         # Low Risk  
            'Medium Risk': 0.50,       # Medium Risk (unchanged)
            'High Risk': 0.70,         # High Risk
            'Very High Risk': 0.85     # Very High Risk
        }
        
        # Handle variations in text
        risk_text_lower = risk_text.lower()
        if 'very low' in risk_text_lower:
            return 0.15
        elif 'low' in risk_text_lower:
            return 0.30
        elif 'medium' in risk_text_lower:
            return 0.50
        elif 'high' in risk_text_lower:
            return 0.70
        elif 'very high' in risk_text_lower:
            return 0.85
        else:
            return 0.50  # Default to medium
    
    def _get_injury_normalization_ranges(self, position):
        """Get position-specific normalization ranges for injury metrics."""
        # Base ranges (can be adjusted based on data analysis)
        base_ranges = {
            'projected_games_missed': (0, 8),      # 0-8 games missed
            'injuries_per_season': (0, 3),          # 0-3 injuries per season
            'injury_risk_per_game': (0, 15),        # 0-15% risk per game
            'durability': (0, 5)                    # 0-5 durability scale
        }
        
        # Position-specific adjustments (minimal since we're removing position multipliers)
        position_adjustments = {
            'QB': {'projected_games_missed': (0, 6)},   # QBs miss fewer games
            'RB': {'projected_games_missed': (0, 10)},  # RBs miss more games
            'WR': {'projected_games_missed': (0, 8)},   # WRs miss moderate games
            'TE': {'projected_games_missed': (0, 8)},   # TEs miss moderate games
        }
        
        ranges = base_ranges.copy()
        if position in position_adjustments:
            ranges.update(position_adjustments[position])
        
        return ranges
    
    def _normalize_injury_metric(self, value, range_tuple, inverse=False):
        """
        Normalize injury metric using min-max scaling.
        
        Args:
            value: Raw metric value
            range_tuple: (min_val, max_val) for normalization
            inverse: If True, invert the scale (lower values = higher scores)
        
        Returns:
            Normalized score (0-10)
        """
        min_val, max_val = range_tuple
        
        # Handle edge cases
        if min_val == max_val:
            return 5.0  # Neutral score if no variance
        
        # Clamp value to range
        clamped_value = max(min_val, min(max_val, value))
        
        # Normalize to 0-1 range
        normalized = (clamped_value - min_val) / (max_val - min_val)
        
        # Apply inverse if needed
        if inverse:
            normalized = 1.0 - normalized
        
        # Convert to 0-10 scale
        return normalized * 10.0

    def _get_position_injury_data(self, position: str) -> Dict[str, List[float]]:
        """Get all injury data for a position to calculate percentiles."""
        try:
            if position not in self.data_loader.injury_data:
                return self._get_default_position_data()
            
            df = self.data_loader.injury_data[position]
            
            # Extract all available data for each metric
            position_data = {
                'projected_games_missed': df['Projected\nGames\nMissed'].dropna().tolist(),
                'injuries_per_season': df['Injuries\nPer Season'].dropna().apply(self._safe_float).tolist(),
                'injury_risk_per_game': df['Injury Risk\nPer Game'].dropna().apply(self._safe_float).tolist(),
                'durability': df['Durability'].dropna().apply(self._safe_float).tolist()
            }
            
            return position_data
            
        except Exception as e:
            print(f"Warning: Could not get position data for {position}: {str(e)}")
            return self._get_default_position_data()
    
    def _get_default_position_data(self) -> Dict[str, List[float]]:
        """Get default data ranges when position data is not available."""
        return {
            'projected_games_missed': list(range(0, 9)),  # 0-8 games
            'injuries_per_season': [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],  # 0-3 injuries
            'injury_risk_per_game': list(range(0, 16)),  # 0-15% risk
            'durability': [1.0, 2.0, 3.0, 4.0, 5.0]  # 1-5 durability
        }
    
    def _percentile_normalize_metric(self, value: float, data_list: List[float], inverse: bool = False) -> float:
        """
        Normalize metric using percentile-based scoring with specified distribution.
        
        Args:
            value: Raw metric value
            data_list: List of all values for this metric in the position
            inverse: If True, invert the scale (lower values = higher scores)
        
        Returns:
            Normalized score (1-10) based on percentile distribution
        """
        if not data_list or len(data_list) < 2:
            return 5.0  # Neutral score if no data
        
        # Calculate percentile of the value
        sorted_data = sorted(data_list)
        percentile = self._calculate_percentile(value, sorted_data)
        
        # Apply inverse if needed
        if inverse:
            percentile = 1.0 - percentile
        
        # Map percentile to score using specified distribution
        score = self._map_percentile_to_score(percentile)
        
        return score
    
    def _calculate_percentile(self, value: float, sorted_data: List[float]) -> float:
        """Calculate percentile of value in sorted data."""
        if value <= sorted_data[0]:
            return 0.0
        if value >= sorted_data[-1]:
            return 1.0
        
        # Find position in sorted data
        for i, data_point in enumerate(sorted_data):
            if value <= data_point:
                # Linear interpolation
                if i == 0:
                    return 0.0
                prev_value = sorted_data[i-1]
                return (i - 1 + (value - prev_value) / (data_point - prev_value)) / len(sorted_data)
        
        return 1.0
    
    def _map_percentile_to_score(self, percentile: float) -> float:
        """
        Map percentile to score using specified distribution:
        - Bottom 5% → 1.0, Bottom 15% → 2.5, Bottom 35% → 4.0
        - Middle 30% → 6.0, Top 20% → 8.0, Top 10% → 9.0, Top 5% → 10.0
        """
        if percentile <= 0.05:
            return 1.0
        elif percentile <= 0.15:
            # Linear interpolation between 1.0 and 2.5
            return 1.0 + (percentile - 0.05) * (1.5 / 0.10)
        elif percentile <= 0.35:
            # Linear interpolation between 2.5 and 4.0
            return 2.5 + (percentile - 0.15) * (1.5 / 0.20)
        elif percentile <= 0.65:
            # Linear interpolation between 4.0 and 6.0
            return 4.0 + (percentile - 0.35) * (2.0 / 0.30)
        elif percentile <= 0.80:
            # Linear interpolation between 6.0 and 8.0
            return 6.0 + (percentile - 0.65) * (2.0 / 0.15)
        elif percentile <= 0.90:
            # Linear interpolation between 8.0 and 9.0
            return 8.0 + (percentile - 0.80) * (1.0 / 0.10)
        elif percentile <= 0.95:
            # Linear interpolation between 9.0 and 10.0
            return 9.0 + (percentile - 0.90) * (1.0 / 0.05)
        else:
            return 10.0

    def _calculate_extreme_age_factor(self, age):
        """Calculate age-based injury risk factor with extreme thresholds."""
        if age is None or pd.isna(age):
            return 5.0  # Default middle score
        
        age = float(age)
        
        # Extreme age-based risk calculation
        if age <= 22:
            return 9.0  # Very low risk (young players like Bijan)
        elif age <= 24:
            return 8.0   # Low risk
        elif age <= 26:
            return 7.0   # Slightly low risk
        elif age <= 28:
            return 6.0   # Moderate risk
        elif age <= 30:
            return 5.0   # Average risk
        elif age <= 32:
            return 4.0   # Slightly high risk
        elif age <= 34:
            return 3.0   # High risk
        elif age <= 36:
            return 2.0   # Very high risk
        elif age <= 38:
            return 1.0   # Extremely high risk
        else:
            return 0.5   # Maximum risk
    
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