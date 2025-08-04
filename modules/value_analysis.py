"""
Value Analysis Module for Fantasy Football Analyzer

Handles ADP analysis and value identification.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from config.constants import (
    ADP_OUTPERFORM_THRESHOLD, POSITIONS, KEY_METRICS
)


class ValueAnalyzer:
    """
    Handles ADP analysis and value identification.
    """
    
    def __init__(self, data_loader, scoring_engine):
        """
        Initialize the value analyzer.
        
        Args:
            data_loader: DataLoader instance
            scoring_engine: ScoringEngine instance
        """
        self.data_loader = data_loader
        self.scoring_engine = scoring_engine
    
    def analyze_player_value_predictions(self) -> str:
        """
        Analyze player value predictions based on advanced metrics and ADP.
        
        Returns:
            Analysis report as string
        """
        print("\n" + "="*60)
        print("PLAYER VALUE PREDICTIONS ANALYSIS")
        print("="*60)
        
        analysis_text = []
        
        analysis_text.append("PLAYER VALUE PREDICTIONS")
        analysis_text.append("="*60)
        
        # Analyze each position
        for position in POSITIONS:
            analysis_text.append(f"\n{position} VALUE ANALYSIS")
            analysis_text.append("-" * 40)
            
            # Get current year data
            current_year = max(self.data_loader.years)
            
            if position not in self.data_loader.advanced_data or current_year not in self.data_loader.advanced_data[position]:
                analysis_text.append(f"No advanced data available for {position} in {current_year}")
                continue
            
            advanced_df = self.data_loader.advanced_data[position][current_year]
            adp_df = self.data_loader.adp_data.get(current_year)
            
            if adp_df is None:
                analysis_text.append(f"No ADP data available for {current_year}")
                continue
            
            # Get column names
            advanced_player_col = self.data_loader.get_player_column(advanced_df, f"Advanced {position}")
            adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {current_year}")
            adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {current_year}")
            
            if not all([advanced_player_col, adp_player_col, adp_col]):
                analysis_text.append(f"Missing required columns for {position} analysis")
                continue
            
            # Clean player names
            advanced_df['Player_Clean'] = advanced_df[advanced_player_col].apply(self.data_loader.clean_player_name)
            adp_df['Player_Clean'] = adp_df[adp_player_col].apply(self.data_loader.clean_player_name)
            
            # Merge datasets
            merged = pd.merge(advanced_df, adp_df, on='Player_Clean', how='inner')
            
            value_players = []
            overvalued_players = []
            
            for _, row in merged.iterrows():
                player_name = row['Player_Clean']
                adp_rank = self.data_loader.safe_float_conversion(row.get(adp_col))
                
                if adp_rank <= 0:
                    continue
                
                # Calculate advanced score
                try:
                    advanced_score, breakdown = self.scoring_engine.calculate_advanced_score_universal(row, position)
                    
                    # Get expected ADP range based on advanced score
                    expected_range = self._get_expected_adp_range(advanced_score, position)
                    expected_adp = np.mean(expected_range)
                    
                    # Calculate value
                    adp_diff = expected_adp - adp_rank  # Positive = undervalued
                    
                    player_data = {
                        'player': player_name,
                        'adp': adp_rank,
                        'advanced_score': advanced_score,
                        'expected_adp': expected_adp,
                        'adp_diff': adp_diff,
                        'value_type': self._classify_value_type(adp_rank, expected_range, advanced_score)
                    }
                    
                    if adp_diff >= 12:  # Significant value
                        value_players.append(player_data)
                    elif adp_diff <= -12:  # Overvalued
                        overvalued_players.append(player_data)
                    
                except Exception as e:
                    continue
            
            # Sort by value
            value_players.sort(key=lambda x: x['adp_diff'], reverse=True)
            overvalued_players.sort(key=lambda x: x['adp_diff'])
            
            # Report findings
            analysis_text.append(f"Top Value Picks ({len(value_players)} found):")
            for player in value_players[:10]:  # Top 10
                analysis_text.append(f"  {player['player']}: ADP {player['adp']:.0f}, Expected {player['expected_adp']:.0f}, +{player['adp_diff']:.0f} picks")
            
            analysis_text.append(f"\nOvervalued Players ({len(overvalued_players)} found):")
            for player in overvalued_players[:10]:  # Top 10
                analysis_text.append(f"  {player['player']}: ADP {player['adp']:.0f}, Expected {player['expected_adp']:.0f}, {player['adp_diff']:.0f} picks")
        
        return "\n".join(analysis_text)
    
    def analyze_adp_value_mismatches(self) -> str:
        """
        Analyze ADP value mismatches across all positions.
        
        Returns:
            Analysis report as string
        """
        print("\n" + "="*60)
        print("ADP VALUE MISMATCHES ANALYSIS")
        print("="*60)
        
        analysis_text = []
        
        analysis_text.append("ADP VALUE MISMATCHES")
        analysis_text.append("="*60)
        
        # Get current year data
        current_year = max(self.data_loader.years)
        
        all_value_players = []
        all_overvalued_players = []
        
        for position in POSITIONS:
            if position not in self.data_loader.advanced_data or current_year not in self.data_loader.advanced_data[position]:
                continue
            
            advanced_df = self.data_loader.advanced_data[position][current_year]
            adp_df = self.data_loader.adp_data.get(current_year)
            
            if adp_df is None:
                continue
            
            # Get column names
            advanced_player_col = self.data_loader.get_player_column(advanced_df, f"Advanced {position}")
            adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {current_year}")
            adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {current_year}")
            
            if not all([advanced_player_col, adp_player_col, adp_col]):
                continue
            
            # Clean player names
            advanced_df['Player_Clean'] = advanced_df[advanced_player_col].apply(self.data_loader.clean_player_name)
            adp_df['Player_Clean'] = adp_df[adp_player_col].apply(self.data_loader.clean_player_name)
            
            # Merge datasets
            merged = pd.merge(advanced_df, adp_df, on='Player_Clean', how='inner')
            
            for _, row in merged.iterrows():
                player_name = row['Player_Clean']
                adp_rank = self.data_loader.safe_float_conversion(row.get(adp_col))
                
                if adp_rank <= 0:
                    continue
                
                # Calculate advanced score
                try:
                    advanced_score, breakdown = self.scoring_engine.calculate_advanced_score_universal(row, position)
                    
                    # Get expected ADP range
                    expected_range = self._get_expected_adp_range(advanced_score, position)
                    expected_adp = np.mean(expected_range)
                    
                    # Calculate value
                    adp_diff = expected_adp - adp_rank
                    
                    player_data = {
                        'player': player_name,
                        'position': position,
                        'adp': adp_rank,
                        'advanced_score': advanced_score,
                        'expected_adp': expected_adp,
                        'adp_diff': adp_diff,
                        'value_type': self._classify_value_type(adp_rank, expected_range, advanced_score)
                    }
                    
                    if adp_diff >= 12:  # Significant value
                        all_value_players.append(player_data)
                    elif adp_diff <= -12:  # Overvalued
                        all_overvalued_players.append(player_data)
                    
                except Exception as e:
                    continue
        
        # Sort by value
        all_value_players.sort(key=lambda x: x['adp_diff'], reverse=True)
        all_overvalued_players.sort(key=lambda x: x['adp_diff'])
        
        # Report findings
        analysis_text.append(f"TOP VALUE PICKS (All Positions)")
        analysis_text.append("-" * 40)
        for player in all_value_players[:20]:  # Top 20
            analysis_text.append(f"{player['player']} ({player['position']}): ADP {player['adp']:.0f}, Expected {player['expected_adp']:.0f}, +{player['adp_diff']:.0f} picks")
        
        analysis_text.append(f"\nMOST OVERVALUED PLAYERS (All Positions)")
        analysis_text.append("-" * 40)
        for player in all_overvalued_players[:20]:  # Top 20
            analysis_text.append(f"{player['player']} ({player['position']}): ADP {player['adp']:.0f}, Expected {player['expected_adp']:.0f}, {player['adp_diff']:.0f} picks")
        
        # Position breakdown
        analysis_text.append(f"\nVALUE BREAKDOWN BY POSITION")
        analysis_text.append("-" * 40)
        
        for position in POSITIONS:
            pos_values = [p for p in all_value_players if p['position'] == position]
            pos_overvalued = [p for p in all_overvalued_players if p['position'] == position]
            
            analysis_text.append(f"{position}: {len(pos_values)} values, {len(pos_overvalued)} overvalued")
        
        return "\n".join(analysis_text)
    
    def _get_expected_adp_range(self, advanced_score: float, position: str) -> Tuple[float, float]:
        """
        Get expected ADP range based on advanced score.
        
        Args:
            advanced_score: Advanced metrics score (0-10)
            position: Player position
            
        Returns:
            Tuple of (min_adp, max_adp)
        """
        # Position-specific ADP ranges based on advanced score
        # Higher advanced score = lower (better) ADP
        
        if position == 'QB':
            if advanced_score >= 8.0:
                return (1, 24)  # QB1 range
            elif advanced_score >= 6.0:
                return (25, 60)  # QB2 range
            elif advanced_score >= 4.0:
                return (61, 120)  # QB3 range
            else:
                return (121, 200)  # Late round
                
        elif position == 'RB':
            if advanced_score >= 8.0:
                return (1, 12)  # RB1 range
            elif advanced_score >= 6.0:
                return (13, 36)  # RB2 range
            elif advanced_score >= 4.0:
                return (37, 84)  # RB3 range
            else:
                return (85, 200)  # Late round
                
        elif position == 'WR':
            if advanced_score >= 8.0:
                return (1, 24)  # WR1 range
            elif advanced_score >= 6.0:
                return (25, 60)  # WR2 range
            elif advanced_score >= 4.0:
                return (61, 120)  # WR3 range
            else:
                return (121, 200)  # Late round
                
        elif position == 'TE':
            if advanced_score >= 8.0:
                return (1, 36)  # TE1 range
            elif advanced_score >= 6.0:
                return (37, 84)  # TE2 range
            elif advanced_score >= 4.0:
                return (85, 144)  # TE3 range
            else:
                return (145, 200)  # Late round
        
        # Default range
        return (100, 150)
    
    def _classify_value_type(self, actual_adp: float, expected_range: Tuple[float, float], score: float) -> str:
        """
        Classify the type of value for a player.
        
        Args:
            actual_adp: Actual ADP rank
            expected_range: Expected ADP range
            score: Advanced score
            
        Returns:
            Value classification string
        """
        expected_min, expected_max = expected_range
        expected_adp = (expected_min + expected_max) / 2
        
        adp_diff = expected_adp - actual_adp
        
        if adp_diff >= 24:
            return "Massive Value"
        elif adp_diff >= 12:
            return "Good Value"
        elif adp_diff >= 6:
            return "Slight Value"
        elif adp_diff >= -6:
            return "Fair Value"
        elif adp_diff >= -12:
            return "Slight Reach"
        elif adp_diff >= -24:
            return "Significant Reach"
        else:
            return "Major Reach"
    
    def find_position_values(self, position: str, min_adp: int = 1, max_adp: int = 200) -> List[Dict[str, Any]]:
        """
        Find value players for a specific position within ADP range.
        
        Args:
            position: Player position
            min_adp: Minimum ADP to consider
            max_adp: Maximum ADP to consider
            
        Returns:
            List of value player dictionaries
        """
        current_year = max(self.data_loader.years)
        
        if position not in self.data_loader.advanced_data or current_year not in self.data_loader.advanced_data[position]:
            return []
        
        advanced_df = self.data_loader.advanced_data[position][current_year]
        adp_df = self.data_loader.adp_data.get(current_year)
        
        if adp_df is None:
            return []
        
        # Get column names
        advanced_player_col = self.data_loader.get_player_column(advanced_df, f"Advanced {position}")
        adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {current_year}")
        adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {current_year}")
        
        if not all([advanced_player_col, adp_player_col, adp_col]):
            return []
        
        # Clean player names
        advanced_df['Player_Clean'] = advanced_df[advanced_player_col].apply(self.data_loader.clean_player_name)
        adp_df['Player_Clean'] = adp_df[adp_player_col].apply(self.data_loader.clean_player_name)
        
        # Merge datasets
        merged = pd.merge(advanced_df, adp_df, on='Player_Clean', how='inner')
        
        value_players = []
        
        for _, row in merged.iterrows():
            player_name = row['Player_Clean']
            adp_rank = self.data_loader.safe_float_conversion(row.get(adp_col))
            
            if adp_rank < min_adp or adp_rank > max_adp:
                continue
            
            # Calculate advanced score
            try:
                advanced_score, breakdown = self.scoring_engine.calculate_advanced_score_universal(row, position)
                
                # Get expected ADP range
                expected_range = self._get_expected_adp_range(advanced_score, position)
                expected_adp = np.mean(expected_range)
                
                # Calculate value
                adp_diff = expected_adp - adp_rank
                
                player_data = {
                    'player': player_name,
                    'position': position,
                    'adp': adp_rank,
                    'advanced_score': advanced_score,
                    'expected_adp': expected_adp,
                    'adp_diff': adp_diff,
                    'value_type': self._classify_value_type(adp_rank, expected_range, advanced_score),
                    'breakdown': breakdown
                }
                
                if adp_diff >= 6:  # Any value
                    value_players.append(player_data)
                
            except Exception as e:
                continue
        
        # Sort by value
        value_players.sort(key=lambda x: x['adp_diff'], reverse=True)
        
        return value_players 