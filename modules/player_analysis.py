"""
Player Analysis Module for Fantasy Football Analyzer

Handles individual player analysis and profiling.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

from config.constants import (
    POSITIONS, KEY_METRICS, DEFAULT_SCORE, LEAGUE_SCORING
)


class PlayerAnalyzer:
    """
    Handles individual player analysis and profiling.
    """
    
    def __init__(self, data_loader, scoring_engine, ml_models):
        """
        Initialize the player analyzer.
        
        Args:
            data_loader: DataLoader instance
            scoring_engine: ScoringEngine instance
            ml_models: MLModels instance
        """
        self.data_loader = data_loader
        self.scoring_engine = scoring_engine
        self.ml_models = ml_models
        
        # Ensure ML models have access to scoring engine for injury calculations
        if hasattr(self.ml_models, 'set_scoring_engine'):
            self.ml_models.set_scoring_engine(self.scoring_engine)
    
    def calculate_historical_performance(self, player: str, position: str) -> Dict[str, float]:
        """
        Calculate enhanced historical performance metrics for a player.
        
        Args:
            player: Player name
            position: Player position
            
        Returns:
            Dictionary with enhanced historical performance metrics
        """
        historical_data = {
            'years_of_data': 0,
            'adp_differential': 0,
            'peak_season': 0,
            'peak_ppg': 0,  # NEW: Peak points per game
            'avg_ppg': 0,   # NEW: Average points per game
            'ppg_consistency': 0,
            'injury_adjustment': 0,
            'games_played_data': [],
            'ppg_data': [],
            'adp_differentials': [],
            'has_historical_data': False
        }
        
        # Collect enhanced data across all years
        adp_differentials = []
        fantasy_points = []
        finish_ranks = []
        peak_ppg = 0  # Track peak PPG across all years
        
        for year in self.data_loader.years:
            # Get ADP data
            if year in self.data_loader.adp_data:
                adp_df = self.data_loader.adp_data[year]
                adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {year}")
                adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {year}")
                pos_col = self.data_loader.get_position_column(adp_df, f"ADP {year}")
                
                if adp_player_col and adp_col and pos_col:
                    # Filter by position first (ADP data has "WR1", "RB2" format)
                    adp_df_filtered = adp_df[adp_df[pos_col].notna() & adp_df[pos_col].str.upper().str.startswith(position.upper())]
                    
                    if not adp_df_filtered.empty:
                        # Find player in filtered ADP data
                        clean_player = self.data_loader.clean_player_name_for_matching(player)
                        adp_df_cleaned = adp_df_filtered.copy()
                        adp_df_cleaned['Player_Clean'] = adp_df_cleaned[adp_player_col].apply(
                            self.data_loader.clean_player_name_for_matching)
                        player_adp = adp_df_cleaned[adp_df_cleaned['Player_Clean'] == clean_player]
                        
                        if not player_adp.empty:
                            adp_rank = self.data_loader.safe_float_conversion(
                                player_adp.iloc[0].get(adp_col))
                            
                            # Get results data
                            if year in self.data_loader.results_data:
                                results_df = self.data_loader.results_data[year]
                                results_player_col = self.data_loader.get_player_column(results_df, f"Results {year}")
                                results_pos_col = self.data_loader.get_position_column(results_df, f"Results {year}")
                                
                                if results_player_col and results_pos_col:
                                    # Filter results by position
                                    results_df_filtered = results_df[results_df[results_pos_col].str.upper() == position.upper()]
                                    
                                    if not results_df_filtered.empty:
                                        # Find player in filtered results data
                                        results_df_cleaned = results_df_filtered.copy()
                                        results_df_cleaned['Player_Clean'] = results_df_cleaned[results_player_col].apply(
                                            self.data_loader.clean_player_name_for_matching)
                                        player_results = results_df_cleaned[results_df_cleaned['Player_Clean'] == clean_player]
                                        
                                        if not player_results.empty:
                                            player_result = player_results.iloc[0]
                                            finish_rank = self.data_loader.safe_float_conversion(
                                                player_result.get('RK', player_result.get('#', 999)))
                                            fantasy_pts = self.data_loader.safe_float_conversion(
                                                player_result.get('FPTS', player_result.get('TTL', player_result.get('PTS', 0))))
                                            
                                            # NEW: Collect weekly points for PPG calculation
                                            weekly_points = []
                                            for week in range(1, 19):  # Weeks 1-18
                                                week_points = self.data_loader.safe_float_conversion(
                                                    player_result.get(str(week), 0))
                                                if week_points > 0:  # Only count played weeks
                                                    weekly_points.append(week_points)
                                            
                                            if adp_rank > 0 and finish_rank > 0:
                                                adp_diff = adp_rank - finish_rank  # Positive = outperformed ADP
                                                adp_differentials.append(adp_diff)
                                                fantasy_points.append(fantasy_pts)
                                                finish_ranks.append(finish_rank)
                                                
                                                # NEW: Store enhanced data and track peak PPG
                                                if weekly_points:
                                                    games_played = len(weekly_points)
                                                    ppg = np.mean(weekly_points)
                                                    
                                                    # Track peak PPG across all years
                                                    if ppg > peak_ppg:
                                                        peak_ppg = ppg
                                                    
                                                    historical_data['games_played_data'].append(games_played)
                                                    historical_data['ppg_data'].append(ppg)
        
        # Calculate enhanced historical metrics
        if adp_differentials:
            historical_data['years_of_data'] = len(adp_differentials)
            historical_data['adp_differential'] = np.mean(adp_differentials)
            historical_data['peak_season'] = max(fantasy_points) if fantasy_points else 0
            historical_data['peak_ppg'] = peak_ppg  # NEW: Store peak PPG
            historical_data['avg_ppg'] = np.mean(historical_data['ppg_data']) if historical_data['ppg_data'] else 0  # NEW: Store average PPG
            historical_data['adp_differentials'] = adp_differentials
            
            # Calculate PPG consistency using mean-to-std ratio - REQUIRE sufficient data
            if len(historical_data['ppg_data']) >= 2:
                ppg_std = np.std(historical_data['ppg_data'])
                ppg_mean = np.mean(historical_data['ppg_data'])
                epsilon = 0.1  # Small constant to avoid division by zero
                
                # Use mean-to-std ratio: mean_ppg / (std_ppg + ε)
                # This rewards consistent positive performance, penalizes consistent negative performance
                if ppg_mean > 0:
                    consistency_ratio = ppg_mean / (ppg_std + epsilon)
                    # Normalize to 0-100 scale (typical good ratio is 2-5, excellent is 5+)
                    historical_data['ppg_consistency'] = max(0, min(100, consistency_ratio * 10))
                else:
                    # If mean PPG is 0 or negative, consistency is 0
                    historical_data['ppg_consistency'] = 0
            else:
                # Log missing PPG consistency data
                self.data_loader._log_missing_data(player, position, "ppg_consistency", 
                                                 ["ppg_data"], f"insufficient_data: {len(historical_data['ppg_data'])} years")
                historical_data['ppg_consistency'] = None
            
            # Calculate injury adjustment - REQUIRE games played data
            if historical_data['games_played_data']:
                avg_games_played = np.mean(historical_data['games_played_data'])
                expected_games = 17  # 2021+ seasons
                durability_ratio = avg_games_played / expected_games
                historical_data['injury_adjustment'] = max(0, min(10, durability_ratio * 10))
            else:
                # Log missing injury adjustment data
                self.data_loader._log_missing_data(player, position, "injury_adjustment", 
                                                 ["games_played_data"], "no_games_played_data")
                historical_data['injury_adjustment'] = None
            
            historical_data['has_historical_data'] = True
        else:
            # Log missing historical data entirely
            self.data_loader._log_missing_data(player, position, "historical_data", 
                                             ["adp_differentials"], "no_adp_data_found")
        
        return historical_data
    
    def get_injury_profile(self, player: str, position: str, year: int = 2024) -> Dict[str, Any]:
        """
        Get injury profile for a player with improved parsing and age factor.
        
        Args:
            player: Player name
            position: Player position
            year: Year for season length calculation
            
        Returns:
            Dictionary with injury profile data
        """
        injury_profile = {
            'injury_risk': 0.5,  # Default moderate risk
            'games_missed': 0,
            'injury_history': [],
            'has_injury_data': False,
            'age': None
        }
        
        # Get player age
        age = self.ml_models._get_player_age(player, position)
        injury_profile['age'] = age
        
        # Get injury data
        if position in self.ml_models.injury_data:
            df = self.ml_models.injury_data[position]
            
            # Try to find player in injury data
            player_found = False
            for _, row in df.iterrows():
                player_name_col = None
                for col in df.columns:
                    if 'player' in col.lower() or 'name' in col.lower():
                        player_name_col = col
                        break
                
                if player_name_col is None:
                    continue
                
                injury_player = str(row[player_name_col]).strip()
                if player in injury_player or injury_player in player:
                    player_found = True
                    
                    # Extract injury risk - handle different column formats
                    injury_risk = 0.5  # Default
                    games_missed = 0.0
                    
                    # Try to find injury risk column
                    for col in df.columns:
                        if 'risk' in col.lower() and 'injury' in col.lower():
                            risk_value = row[col]
                            if isinstance(risk_value, str):
                                if 'high' in risk_value.lower():
                                    injury_risk = 0.8
                                elif 'very high' in risk_value.lower():
                                    injury_risk = 0.9
                                elif 'medium' in risk_value.lower():
                                    injury_risk = 0.5
                                elif 'low' in risk_value.lower():
                                    injury_risk = 0.3
                                elif 'very low' in risk_value.lower():
                                    injury_risk = 0.2
                            break
                    
                    # Try to find games missed column
                    for col in df.columns:
                        if 'missed' in col.lower() or 'games' in col.lower():
                            try:
                                games_missed = float(row[col])
                            except (ValueError, TypeError):
                                games_missed = 0.0
                            break
                    
                    # Try to find injury risk percentage
                    for col in df.columns:
                        if 'risk' in col.lower() and '%' in str(row[col]):
                            try:
                                risk_pct = str(row[col]).replace('%', '').strip()
                                injury_risk = float(risk_pct) / 100.0
                            except (ValueError, TypeError):
                                pass
                            break
                    
                    injury_profile.update({
                        'injury_risk': injury_risk,
                        'games_missed': games_missed,
                        'has_injury_data': True,
                        'injury_history': []  # Could be enhanced with actual history
                    })
                    break
            
            if not player_found:
                # Calculate games missed from advanced data
                season_length = self.ml_models._get_games_played(player, position, year)
                games_played = 0.0
                
                if position in self.ml_models.data_loader.advanced_data and year in self.ml_models.data_loader.advanced_data[position]:
                    df = self.ml_models.data_loader.advanced_data[position][year]
                    player_col = self.ml_models.data_loader.get_player_column(df, f"Advanced {position}")
                    
                    for _, row in df.iterrows():
                        if player in str(row.get(player_col, '')):
                            games_played = self.ml_models.data_loader.safe_float_conversion(row.get('G', 0))
                            break
                
                games_missed = max(0, season_length - games_played)
                injury_profile['games_missed'] = games_missed
        
        return injury_profile

    def _convert_injury_risk_text_to_numeric(self, risk_text: str) -> float:
        """Convert injury risk text to numeric value."""
        if not risk_text or pd.isna(risk_text):
            return 0.5
        
        risk_text = str(risk_text).strip().lower()
        
        if 'very low' in risk_text:
            return 0.1
        elif 'low' in risk_text:
            return 0.3
        elif 'medium' in risk_text:
            return 0.5
        elif 'high' in risk_text:
            return 0.7
        elif 'very high' in risk_text:
            return 0.9
        else:
            return 0.5  # Default to medium risk
    
    def perform_basic_trend_analysis(self, player_name: str, position: str) -> Dict[str, Any]:
        """
        Perform basic trend analysis for a player.
        
        Args:
            player_name: Player name
            position: Player position
            
        Returns:
            Dictionary with trend analysis results
        """
        trend_analysis = {
            'adp_trend': [],
            'performance_trend': [],
            'consistency_score': 0,
            'breakout_potential': 0,
            'risk_factors': []
        }
        
        # Analyze ADP trends
        adp_trend = []
        performance_trend = []
        
        for year in sorted(self.data_loader.years):
            if year in self.data_loader.adp_data:
                adp_df = self.data_loader.adp_data[year]
                adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {year}")
                adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {year}")
                
                if adp_player_col and adp_col:
                    clean_player = self.data_loader.clean_player_name_for_matching(player_name)
                    adp_df_cleaned = adp_df.copy()
                    adp_df_cleaned['Player_Clean'] = adp_df_cleaned[adp_player_col].apply(
                        self.data_loader.clean_player_name_for_matching)
                    player_adp = adp_df_cleaned[adp_df_cleaned['Player_Clean'] == clean_player]
                    
                    if not player_adp.empty:
                        adp_rank = self.data_loader.safe_float_conversion(
                            player_adp.iloc[0].get(adp_col))
                        adp_trend.append((year, adp_rank))
                        
                        # Get performance data
                        if year in self.data_loader.results_data:
                            results_df = self.data_loader.results_data[year]
                            results_player_col = self.data_loader.get_player_column(results_df, f"Results {year}")
                            
                            if results_player_col:
                                results_df_cleaned = results_df.copy()
                                results_df_cleaned['Player_Clean'] = results_df_cleaned[results_player_col].apply(
                                    self.data_loader.clean_player_name_for_matching)
                                player_results = results_df_cleaned[results_df_cleaned['Player_Clean'] == clean_player]
                                
                                if not player_results.empty:
                                    finish_rank = self.data_loader.safe_float_conversion(
                                        player_results.iloc[0].get('RK', player_results.iloc[0].get('#', 999)))
                                    performance_trend.append((year, finish_rank))
        
        # Calculate trends
        if len(adp_trend) >= 2:
            # ADP trend (improving = lower ADP)
            adp_values = [rank for _, rank in adp_trend]
            adp_slope = np.polyfit(range(len(adp_values)), adp_values, 1)[0]
            trend_analysis['adp_trend'] = adp_trend
            
            # Performance trend (improving = lower finish rank)
            if len(performance_trend) >= 2:
                perf_values = [rank for _, rank in performance_trend]
                perf_slope = np.polyfit(range(len(perf_values)), perf_values, 1)[0]
                trend_analysis['performance_trend'] = performance_trend
                
                # Consistency score (lower variance = higher consistency)
                if len(perf_values) > 1:
                    consistency = max(0, 100 - np.std(perf_values) * 10)
                    trend_analysis['consistency_score'] = consistency
                
                # Breakout potential (improving performance + declining ADP)
                breakout_score = 0
                if perf_slope < 0 and adp_slope > 0:  # Improving performance, declining ADP
                    breakout_score = min(100, abs(perf_slope) * 50 + abs(adp_slope) * 30)
                trend_analysis['breakout_potential'] = breakout_score
        
        # Identify risk factors
        risk_factors = []
        if len(adp_trend) > 0:
            latest_adp = adp_trend[-1][1]
            if latest_adp > 100:  # Late round ADP
                risk_factors.append("Late round ADP")
        
        if len(performance_trend) > 0:
            latest_perf = performance_trend[-1][1]
            if latest_perf > 50:  # Poor recent performance
                risk_factors.append("Poor recent performance")
        
        trend_analysis['risk_factors'] = risk_factors
        
        return trend_analysis
    
    def get_actual_fantasy_points(self, player_name: str, position: str, year: int = 2024) -> Optional[float]:
        """
        Get actual fantasy points for a player in a specific year.
        
        Args:
            player_name: Player name
            position: Player position
            year: Year to get data for
            
        Returns:
            Fantasy points if found, None otherwise
        """
        if year not in self.data_loader.results_data:
            return None
        
        results_df = self.data_loader.results_data[year]
        results_player_col = self.data_loader.get_player_column(results_df, f"Results {year}")
        
        if not results_player_col:
            return None
        
        # Find player in results
        clean_player = self.data_loader.clean_player_name_for_matching(player_name)
        results_df_cleaned = results_df.copy()
        results_df_cleaned['Player_Clean'] = results_df_cleaned[results_player_col].apply(
            self.data_loader.clean_player_name_for_matching)
        player_results = results_df_cleaned[results_df_cleaned['Player_Clean'] == clean_player]
        
        if not player_results.empty:
            player_result = player_results.iloc[0]
            fantasy_points = self.data_loader.safe_float_conversion(
                player_result.get('FPTS', player_result.get('TTL', player_result.get('PTS', 0))))
            return fantasy_points if fantasy_points > 0 else None
        
        return None
    
    def generate_player_profile(self, player_name: str, position: str, 
                              current_year: int = 2024) -> Dict[str, Any]:
        """
        Generate comprehensive player profile.
        
        Args:
            player_name: Player name
            position: Player position
            current_year: Current year for analysis
            
        Returns:
            Dictionary with complete player profile
        """
        profile = {
            'player_name': player_name,
            'position': position,
            'current_year': current_year,
            'historical_performance': {},
            'advanced_metrics': {},
            'ml_predictions': {},
            'injury_profile': {},
            'overall_scores': {},
            'trend_analysis': {},
            'similar_players': []
        }
        
        # Get historical performance
        historical_perf = self.calculate_historical_performance(player_name, position)
        profile['historical_performance'] = historical_perf
        
        # Get advanced metrics
        advanced_metrics = {}
        player_row = None
        
        if position in self.data_loader.advanced_data and current_year in self.data_loader.advanced_data[position]:
            df = self.data_loader.advanced_data[position][current_year]
            player_col = self.data_loader.get_player_column(df, f"Advanced {position} {current_year}")
            
            if player_col:
                clean_player = self.data_loader.clean_player_name_for_matching(player_name)
                df_cleaned = df.copy()
                df_cleaned['Player_Clean'] = df_cleaned[player_col].apply(
                    self.data_loader.clean_player_name_for_matching)
                player_data = df_cleaned[df_cleaned['Player_Clean'] == clean_player]
                
                if not player_data.empty:
                    player_row = player_data.iloc[0]
                    
                    # Extract key metrics
                    key_metrics = KEY_METRICS.get(position, [])
                    for metric in key_metrics:
                        if metric in player_row.index and not pd.isna(player_row[metric]):
                            advanced_metrics[metric] = self.data_loader.safe_float_conversion(player_row[metric])
        
        profile['advanced_metrics'] = advanced_metrics
        
        # Get ML predictions
        features = self.ml_models._extract_enhanced_features_with_age_injury(player_row, position, player_name, current_year)
        if features is not None:
            ml_predictions = self.ml_models.generate_predictions(player_name, position, features)
            profile['ml_predictions'] = ml_predictions
        
        # Get injury profile
        injury_profile = self.get_injury_profile(player_name, position)
        profile['injury_profile'] = injury_profile
        
        # Calculate overall scores
        overall_scores = self.scoring_engine.calculate_overall_profile_score(
            historical_perf, advanced_metrics, profile.get('ml_predictions', {}), 
            injury_profile, position=position, player_row=player_row, player_name=player_name)
        profile['overall_scores'] = overall_scores
        
        # Perform trend analysis
        trend_analysis = self.perform_basic_trend_analysis(player_name, position)
        profile['trend_analysis'] = trend_analysis
        
        # Find similar players
        if player_row is not None:
            similar_players = self.ml_models.find_similar_players(player_name, position, n_similar=5)
            profile['similar_players'] = similar_players
        
        return profile 