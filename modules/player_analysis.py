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
    
    def calculate_historical_performance(self, player: str, position: str) -> Dict[str, float]:
        """
        Calculate historical performance metrics for a player.
        
        Args:
            player: Player name
            position: Player position
            
        Returns:
            Dictionary with historical performance metrics
        """
        historical_data = {
            'years_of_data': 0,
            'adp_differential': 0,
            'peak_season': 0,
            'consistency': 0,
            'has_historical_data': False
        }
        
        # Collect data across all years
        adp_differentials = []
        fantasy_points = []
        finish_ranks = []
        
        for year in self.data_loader.years:
            # Get ADP data
            if year in self.data_loader.adp_data:
                adp_df = self.data_loader.adp_data[year]
                adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {year}")
                adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {year}")
                
                if adp_player_col and adp_col:
                    # Find player in ADP data
                    clean_player = self.data_loader.clean_player_name_for_matching(player)
                    adp_df_cleaned = adp_df.copy()
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
                            
                            if results_player_col:
                                # Find player in results data
                                results_df_cleaned = results_df.copy()
                                results_df_cleaned['Player_Clean'] = results_df_cleaned[results_player_col].apply(
                                    self.data_loader.clean_player_name_for_matching)
                                player_results = results_df_cleaned[results_df_cleaned['Player_Clean'] == clean_player]
                                
                                if not player_results.empty:
                                    player_result = player_results.iloc[0]
                                    finish_rank = self.data_loader.safe_float_conversion(
                                        player_result.get('RK', player_result.get('#', 999)))
                                    fantasy_pts = self.data_loader.safe_float_conversion(
                                        player_result.get('FPTS', player_result.get('TTL', player_result.get('PTS', 0))))
                                    
                                    if adp_rank > 0 and finish_rank > 0:
                                        adp_diff = adp_rank - finish_rank  # Positive = outperformed ADP
                                        adp_differentials.append(adp_diff)
                                        fantasy_points.append(fantasy_pts)
                                        finish_ranks.append(finish_rank)
        
        # Calculate historical metrics
        if adp_differentials:
            historical_data['years_of_data'] = len(adp_differentials)
            historical_data['adp_differential'] = np.mean(adp_differentials)
            historical_data['peak_season'] = max(fantasy_points) if fantasy_points else 0
            historical_data['consistency'] = 100 - np.std(adp_differentials) if len(adp_differentials) > 1 else 0
            historical_data['has_historical_data'] = True
        
        return historical_data
    
    def get_injury_profile(self, player: str, position: str) -> Dict[str, Any]:
        """
        Get injury profile for a player.
        
        Args:
            player: Player name
            position: Player position
            
        Returns:
            Dictionary with injury profile data
        """
        injury_profile = {
            'injury_risk': 0.5,  # Default moderate risk
            'games_missed': 0,
            'injury_history': [],
            'has_injury_data': False
        }
        
        # Check if injury data exists for this position
        if position in self.data_loader.injury_data:
            injury_df = self.data_loader.injury_data[position]
            player_col = self.data_loader.get_player_column(injury_df, f"Injury {position}")
            
            if player_col:
                # Find player in injury data
                clean_player = self.data_loader.clean_player_name_for_matching(player)
                injury_df_cleaned = injury_df.copy()
                injury_df_cleaned['Player_Clean'] = injury_df_cleaned[player_col].apply(
                    self.data_loader.clean_player_name_for_matching)
                player_injury = injury_df_cleaned[injury_df_cleaned['Player_Clean'] == clean_player]
                
                if not player_injury.empty:
                    injury_row = player_injury.iloc[0]
                    
                    # Extract injury metrics
                    injury_risk = self.data_loader.safe_float_conversion(
                        injury_row.get('INJURY_RISK', injury_row.get('RISK', 0.5)))
                    games_missed = self.data_loader.safe_float_conversion(
                        injury_row.get('GAMES_MISSED', injury_row.get('MISSED', 0)))
                    
                    injury_profile.update({
                        'injury_risk': injury_risk,
                        'games_missed': games_missed,
                        'has_injury_data': True
                    })
        
        return injury_profile
    
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
        if player_row is not None:
            features = self.ml_models._extract_features(player_row, position)
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