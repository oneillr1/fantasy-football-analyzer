"""
Advanced Feature Engineering Module for Fantasy Football Analyzer

Handles calculation of advanced efficiency, opportunity, and contact metrics
that provide deeper insights into player performance and predictive value.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from config.constants import POSITIONS, DEFAULT_SCORE


class AdvancedFeatureEngine:
    """
    Handles calculation of advanced efficiency and opportunity metrics.
    
    Features include:
    - Breakaway percentage (separates big-play players from grinders)
    - TD rate (role in scoring plays)
    - Weighted opportunities (values receiving targets more than carries)
    - Contact efficiency metrics (separates OL quality from RB skill)
    - Explosive play rates (big-play potential)
    - Red zone opportunity share (normalized opportunity metrics)
    """
    
    def __init__(self, data_loader):
        """
        Initialize the advanced feature engine.
        
        Args:
            data_loader: Reference to the data loader for safe conversions
        """
        self.data_loader = data_loader
        # Build lookup dictionaries for efficient TD data access
        self._build_td_lookup()
        
    def _build_td_lookup(self):
        """Build lookup dictionaries for TD data from results tables."""
        self.td_lookup = {}
        for year in self.data_loader.years:
            if year in self.data_loader.results_data:
                self.td_lookup[year] = {}
                df = self.data_loader.results_data[year]
                for _, row in df.iterrows():
                    player_name = self.data_loader.clean_player_name_for_matching(row.get('Player', ''))
                    if player_name:
                        self.td_lookup[year][player_name] = {
                            'RushTD': self.data_loader.safe_float_conversion(row.get('RushTD', 0)),
                            'RecTD': self.data_loader.safe_float_conversion(row.get('RecTD', 0)),
                            'PassTD': self.data_loader.safe_float_conversion(row.get('PassTD', 0))
                        }
    
    def _get_total_tds(self, player_name: str, year: int, position: str) -> float:
        """
        Get total TDs for a player from results data.
        
        Args:
            player_name: Player name
            year: Season year
            position: Player position
            
        Returns:
            Total TDs (rushing + receiving for RB/WR/TE, passing for QB)
        """
        if year not in self.td_lookup:
            return 0.0
            
        clean_name = self.data_loader.clean_player_name_for_matching(player_name)
        if clean_name not in self.td_lookup[year]:
            return 0.0
            
        td_data = self.td_lookup[year][clean_name]
        
        if position.lower() == 'qb':
            return td_data.get('PassTD', 0.0)
        else:
            # RB/WR/TE: rushing + receiving TDs
            return td_data.get('RushTD', 0.0) + td_data.get('RecTD', 0.0)
        
    def calculate_all_features(self, row: pd.Series, position: str) -> List[float]:
        """
        Calculate all advanced features for a player.
        
        Args:
            row: Player's advanced metrics data
            position: Player position (QB, RB, WR, TE)
            
        Returns:
            List of calculated feature values
        """
        # Normalize position casing once at the top
        position = position.lower()
        
        features = []
        
        # Efficiency/Rate Features
        features.append(self.calculate_breakaway_percentage(row, position))
        features.append(self.calculate_td_rate(row, position))
        
        # Opportunity Share Features
        features.append(self.calculate_weighted_opportunities(row, position))
        features.append(self.calculate_redzone_opportunity_share(row, position))
        
        # Contact Metrics (RB-specific, 0.0 for other positions)
        if position == 'rb':
            features.extend(self.calculate_contact_efficiency_metrics(row))
        else:
            features.extend([0.0, 0.0, 0.0])  # Placeholder for non-RB positions
        
        # Universal Efficiency Metrics
        features.append(self.calculate_explosive_play_rate(row, position))
        # Removed long_td_rate due to unreliable estimation
        
        return features
    
    def calculate_breakaway_percentage(self, row: pd.Series, position: str) -> float:
        """
        Calculate breakaway percentage based on position.
        
        Args:
            row: Player's advanced metrics data
            position: Player position (normalized to lowercase)
            
        Returns:
            Breakaway percentage (0.0-1.0)
        """
        try:
            if position == 'rb':
                # RB: (10+ YDS) / ATT
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                ten_plus_yds = self.data_loader.safe_float_conversion(row.get('10+ YDS', 0))
                result = ten_plus_yds / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'breakaway_percentage')
                
            elif position in ['wr', 'te']:
                # WR/TE: (20+ YDS) / REC
                rec = self.data_loader.safe_float_conversion(row.get('REC', 0)) or 1
                twenty_plus_yds = self.data_loader.safe_float_conversion(row.get('20+ YDS', 0))
                result = twenty_plus_yds / rec if rec > 0 else 0.0
                return self._validate_rate_feature(result, 'breakaway_percentage')
                
            elif position == 'qb':
                # QB: (20+ YDS) / ATT (using 20+ for QBs as breakaway equivalent)
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                twenty_plus_yds = self.data_loader.safe_float_conversion(row.get('20+ YDS', 0))
                result = twenty_plus_yds / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'breakaway_percentage')
                
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
        
        return 0.0
    
    def calculate_td_rate(self, row: pd.Series, position: str) -> float:
        """
        Calculate TD rate per opportunity using proper TD data from results.
        
        Args:
            row: Player's advanced metrics data
            position: Player position (normalized to lowercase)
            
        Returns:
            TD rate (0.0-1.0)
        """
        try:
            # Get player name and year for TD lookup
            player_name = row.get('Player', '')
            year = self._extract_year_from_row(row)
            
            if position == 'rb':
                # RB: Total TDs / (ATT + REC)
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0))
                rec = self.data_loader.safe_float_conversion(row.get('REC', 0))
                total_opportunities = att + rec
                
                # Get total TDs from results data
                total_tds = self._get_total_tds(player_name, year, position)
                
                result = total_tds / total_opportunities if total_opportunities > 0 else 0.0
                return self._validate_rate_feature(result, 'td_rate')
                
            elif position in ['wr', 'te']:
                # WR/TE: Total TDs / TGT
                tgt = self.data_loader.safe_float_conversion(row.get('TGT', 0)) or 1
                
                # Get total TDs from results data
                total_tds = self._get_total_tds(player_name, year, position)
                
                result = total_tds / tgt if tgt > 0 else 0.0
                return self._validate_rate_feature(result, 'td_rate')
                
            elif position == 'qb':
                # QB: Passing TDs / ATT
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                
                # Get passing TDs from results data
                total_tds = self._get_total_tds(player_name, year, position)
                
                result = total_tds / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'td_rate')
                
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
        
        return 0.0
    
    def calculate_weighted_opportunities(self, row: pd.Series, position: str) -> float:
        """
        Calculate weighted opportunities with per-game normalization.
        
        Args:
            row: Player's advanced metrics data
            position: Player position (normalized to lowercase)
            
        Returns:
            Weighted opportunities per game
        """
        try:
            games = self.data_loader.safe_float_conversion(row.get('G', 0)) or 1
            
            if position == 'rb':
                # RB: (ATT + 1.5*TGT) / games
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0))
                tgt = self.data_loader.safe_float_conversion(row.get('TGT', 0))
                wo_total = att + (1.5 * tgt)
                result = wo_total / games
                return self._validate_per_game_volume(result, 'weighted_opportunities')
                
            elif position in ['wr', 'te']:
                # WR/TE: TGT / games
                tgt = self.data_loader.safe_float_conversion(row.get('TGT', 0))
                result = tgt / games
                return self._validate_per_game_volume(result, 'weighted_opportunities')
                
            elif position == 'qb':
                # QB: ATT / games
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0))
                result = att / games
                return self._validate_per_game_volume(result, 'weighted_opportunities')
                
        except (ValueError, TypeError):
            return 0.0
        
        return 0.0
    
    def calculate_redzone_opportunity_share(self, row: pd.Series, position: str) -> float:
        """
        Calculate red zone opportunity share.
        
        Args:
            row: Player's advanced metrics data
            position: Player position (normalized to lowercase)
            
        Returns:
            Red zone opportunity share (0.0-1.0)
        """
        try:
            if position == 'rb':
                # RB: RZ_ATT / ATT
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                rz_att = self.data_loader.safe_float_conversion(row.get('RZ ATT', 0))
                result = rz_att / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'redzone_opportunity_share')
                
            elif position in ['wr', 'te']:
                # WR/TE: RZ_TGT / TGT
                tgt = self.data_loader.safe_float_conversion(row.get('TGT', 0)) or 1
                rz_tgt = self.data_loader.safe_float_conversion(row.get('RZ TGT', 0))
                result = rz_tgt / tgt if tgt > 0 else 0.0
                return self._validate_rate_feature(result, 'redzone_opportunity_share')
                
            elif position == 'qb':
                # QB: RZ_ATT / ATT
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                rz_att = self.data_loader.safe_float_conversion(row.get('RZ ATT', 0))
                result = rz_att / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'redzone_opportunity_share')
                
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
        
        return 0.0
    
    def calculate_contact_efficiency_metrics(self, row: pd.Series) -> List[float]:
        """
        Calculate contact efficiency metrics for RBs.
        
        Args:
            row: Player's advanced metrics data
            
        Returns:
            List of [ybc_per_att, yac_per_att, contact_efficiency_ratio]
        """
        try:
            att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
            ybcon = self.data_loader.safe_float_conversion(row.get('YBCON', 0))
            yacon = self.data_loader.safe_float_conversion(row.get('YACON', 0))
            
            # Yards Before Contact per Attempt (OL quality)
            ybc_per_att = ybcon / att if att > 0 else 0.0
            ybc_per_att = self._validate_per_game_volume(ybc_per_att, 'ybc_per_att')
            
            # Yards After Contact per Attempt (RB elusiveness)
            yac_per_att = yacon / att if att > 0 else 0.0
            yac_per_att = self._validate_per_game_volume(yac_per_att, 'yac_per_att')
            
            # Contact Efficiency Ratio (YACON / YBCON)
            contact_efficiency_ratio = yacon / ybcon if ybcon > 0 else 0.0
            contact_efficiency_ratio = self._validate_rate_feature(contact_efficiency_ratio, 'contact_efficiency_ratio')
            
            return [ybc_per_att, yac_per_att, contact_efficiency_ratio]
            
        except (ValueError, TypeError, ZeroDivisionError):
            return [0.0, 0.0, 0.0]
    
    def calculate_explosive_play_rate(self, row: pd.Series, position: str) -> float:
        """
        Calculate explosive play rate.
        
        Args:
            row: Player's advanced metrics data
            position: Player position (normalized to lowercase)
            
        Returns:
            Explosive play rate (0.0-1.0)
        """
        try:
            if position == 'rb':
                # RB: (20+ YDS) / ATT
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                twenty_plus_yds = self.data_loader.safe_float_conversion(row.get('20+ YDS', 0))
                result = twenty_plus_yds / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'explosive_play_rate')
                
            elif position in ['wr', 'te']:
                # WR/TE: (20+ YDS) / REC
                rec = self.data_loader.safe_float_conversion(row.get('REC', 0)) or 1
                twenty_plus_yds = self.data_loader.safe_float_conversion(row.get('20+ YDS', 0))
                result = twenty_plus_yds / rec if rec > 0 else 0.0
                return self._validate_rate_feature(result, 'explosive_play_rate')
                
            elif position == 'qb':
                # QB: (20+ YDS) / ATT
                att = self.data_loader.safe_float_conversion(row.get('ATT', 0)) or 1
                twenty_plus_yds = self.data_loader.safe_float_conversion(row.get('20+ YDS', 0))
                result = twenty_plus_yds / att if att > 0 else 0.0
                return self._validate_rate_feature(result, 'explosive_play_rate')
                
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
        
        return 0.0
    
    def _extract_year_from_row(self, row: pd.Series) -> int:
        """Extract year from row data or use a default."""
        # Try to extract year from various possible sources
        for col in row.index:
            if 'year' in col.lower() or 'season' in col.lower():
                year_val = self.data_loader.safe_float_conversion(row.get(col, 0))
                if year_val and 2019 <= year_val <= 2025:
                    return int(year_val)
        
        # Default to most recent year if not found
        return max(self.data_loader.years) if self.data_loader.years else 2024
    
    def _validate_rate_feature(self, value: float, feature_name: str) -> float:
        """
        Validate rate features (probabilities/rates ∈ [0,1.0]).
        
        Args:
            value: Raw feature value
            feature_name: Name of the feature for debugging
            
        Returns:
            Validated and clipped value
        """
        if np.isnan(value) or np.isinf(value):
            return 0.0
        
        # Clip to [0, 1.0] for rate features
        return max(0.0, min(1.0, value))
    
    def _validate_per_game_volume(self, value: float, feature_name: str) -> float:
        """
        Validate per-game volume features ∈ [0, 30].
        
        Args:
            value: Raw feature value
            feature_name: Name of the feature for debugging
            
        Returns:
            Validated and clipped value
        """
        if np.isnan(value) or np.isinf(value):
            return 0.0
        
        # Clip to [0, 30] for per-game volumes
        return max(0.0, min(30.0, value))
    
    def _validate_total_volume(self, value: float, feature_name: str) -> float:
        """
        Validate total volume features ∈ [0, 600].
        
        Args:
            value: Raw feature value
            feature_name: Name of the feature for debugging
            
        Returns:
            Validated and clipped value
        """
        if np.isnan(value) or np.isinf(value):
            return 0.0
        
        # Clip to [0, 600] for total volumes
        return max(0.0, min(600.0, value))
    
    def get_feature_names(self, position: str) -> List[str]:
        """
        Get the names of all advanced features for a position.
        
        Args:
            position: Player position
            
        Returns:
            List of feature names
        """
        # Normalize position casing
        position = position.lower()
        
        base_features = [
            'breakaway_percentage',
            'td_rate',
            'weighted_opportunities',
            'redzone_opportunity_share',
            'explosive_play_rate'
            # Removed 'long_td_rate' due to unreliable estimation
        ]
        
        if position == 'rb':
            contact_features = [
                'ybc_per_att',
                'yac_per_att', 
                'contact_efficiency_ratio'
            ]
            return base_features + contact_features
        else:
            # Add placeholder names for non-RB positions
            contact_features = [
                'ybc_per_att_placeholder',
                'yac_per_att_placeholder',
                'contact_efficiency_ratio_placeholder'
            ]
            return base_features + contact_features
    
    def validate_features(self, features: List[float], position: str) -> bool:
        """
        Validate that calculated features are within expected ranges.
        
        Args:
            features: List of calculated feature values
            position: Player position
            
        Returns:
            True if features are valid, False otherwise
        """
        if not features:
            return False
            
        expected_length = len(self.get_feature_names(position))
        if len(features) != expected_length:
            print(f"Feature length mismatch: expected {expected_length}, got {len(features)}")
            return False
        
        for i, feature in enumerate(features):
            if not isinstance(feature, (int, float)) or np.isnan(feature) or np.isinf(feature):
                print(f"Invalid feature at index {i}: {feature}")
                return False
        
        return True
