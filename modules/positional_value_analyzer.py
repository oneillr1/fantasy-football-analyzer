"""
Positional Value Analyzer Module for Fantasy Football Analyzer

Analyzes historical positional drop-offs and point differentials to inform draft strategy.
Focus: Understanding the cost of waiting to draft each position by round.
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict

from config.constants import POSITIONS


class PositionalValueAnalyzer:
    """
    Analyzes positional value and drop-offs to inform draft strategy.
    
    Main Focus:
    - Point differential analysis by position and round
    - Historical drop-off patterns (e.g., RB1 vs RB10 vs WR1 vs WR10)
    - Cost of waiting analysis - how many points do you lose by waiting a round?
    - Positional replacement value calculations
    """
    
    def __init__(self, data_loader):
        """
        Initialize the positional value analyzer.
        
        Args:
            data_loader: DataLoader instance with ADP and results data
        """
        self.data_loader = data_loader
        self.position_rankings = {}  # Historical position rankings by year
        self.drop_off_analysis = {}  # Drop-off analysis by position
        
    def analyze_historical_positional_dropoffs(self) -> str:
        """
        Main analysis: Historical positional drop-offs and point differentials.
        
        Returns:
            Comprehensive analysis report as string
        """
        print("\n" + "="*80)
        print("POSITIONAL VALUE ANALYZER - HISTORICAL DROP-OFF ANALYSIS")
        print("="*80)
        
        analysis_text = []
        analysis_text.append("POSITIONAL VALUE ANALYZER")
        analysis_text.append("Historical Drop-off and Point Differential Analysis")
        analysis_text.append("="*80)
        analysis_text.append("")
        analysis_text.append("PURPOSE: Understand the cost of waiting to draft each position")
        analysis_text.append("METHODOLOGY: Compare point differentials between position ranks across years")
        analysis_text.append("")
        
        # Analyze each year's data and build historical patterns
        historical_data = self._build_historical_position_rankings()
        
        if not historical_data:
            analysis_text.append("No sufficient data available for analysis")
            return "\n".join(analysis_text)
        
        # Generate the main analysis sections
        analysis_text.extend(self._analyze_position_dropoffs(historical_data))
        analysis_text.extend(self._analyze_replacement_value(historical_data))
        analysis_text.extend(self._analyze_opportunity_cost_matrix(historical_data))
        analysis_text.extend(self._analyze_risk_adjusted_value(historical_data))
        analysis_text.extend(self._analyze_positional_scarcity(historical_data))
        analysis_text.extend(self._analyze_value_over_replacement(historical_data))
        analysis_text.extend(self._generate_draft_recommendations(historical_data))
        
        return "\n".join(analysis_text)
    
    def _build_historical_position_rankings(self) -> Dict[int, Dict[str, List[Dict]]]:
        """
        Build historical position rankings combining ADP and fantasy points.
        
        Returns:
            Dictionary: {year: {position: [ranked_players]}}
        """
        print("Building historical position rankings...")
        historical_data = {}
        
        for year in self.data_loader.years:
            if year not in self.data_loader.adp_data or year not in self.data_loader.results_data:
                continue
                
            print(f"  Processing {year}...")
            
            adp_df = self.data_loader.adp_data[year]
            results_df = self.data_loader.results_data[year]
            
            # Get column names
            adp_player_col = self._get_player_column(adp_df, "ADP")
            adp_pos_col = self._get_position_column(adp_df, "ADP") 
            adp_rank_col = self._get_adp_column(adp_df, "ADP")
            
            results_player_col = self._get_player_column(results_df, "Results")
            results_points_col = self._get_points_column(results_df, "Results")
            results_pos_col = self._get_position_column(results_df, "Results")
            
            if not all([adp_player_col, adp_pos_col, adp_rank_col, results_player_col, results_points_col]):
                print(f"    Missing required columns for {year}:")
                print(f"      ADP columns: {list(adp_df.columns)}")
                print(f"      Results columns: {list(results_df.columns)}")
                print(f"      Detected: Player='{adp_player_col}', Pos='{adp_pos_col}', ADP='{adp_rank_col}', Results_Player='{results_player_col}', Points='{results_points_col}'")
                continue
            
            # Clean player names for matching
            adp_df['Player_Clean'] = adp_df[adp_player_col].apply(self.data_loader.clean_player_name)
            results_df['Player_Clean'] = results_df[results_player_col].apply(self.data_loader.clean_player_name)
            
            # Merge ADP and results data
            print(f"    ADP data shape: {adp_df.shape}")
            print(f"    Results data shape: {results_df.shape}")
            merged_df = pd.merge(adp_df, results_df, on='Player_Clean', how='inner')
            print(f"    Merged data shape: {merged_df.shape}")
            
            if merged_df.empty:
                print(f"    No players matched between ADP and results data for {year}")
                continue
            
            year_data = {}
            
            for position in POSITIONS:
                # Filter by position - handle merged column names that get suffixed
                available_cols = list(merged_df.columns)
                
                # Find position column after merge (might be suffixed with _x or _y)
                pos_col_to_use = None
                
                # Try results position column first (usually suffixed with _y)
                if results_pos_col:
                    for col in available_cols:
                        if results_pos_col in col:
                            pos_col_to_use = col
                            break
                
                # Fallback to ADP position column (usually suffixed with _x)
                if not pos_col_to_use and adp_pos_col:
                    for col in available_cols:
                        if adp_pos_col in col:
                            pos_col_to_use = col
                            break
                
                if not pos_col_to_use:
                    print(f"    Warning: No position column found for {position} filtering")
                    continue
                
                # Filter by position using exact match first, then contains
                try:
                    pos_data = merged_df[merged_df[pos_col_to_use] == position].copy()
                    if pos_data.empty:
                        # Try contains match for positions like "RB1", "WR2", etc.
                        pos_data = merged_df[merged_df[pos_col_to_use].str.contains(position, case=False, na=False)].copy()
                except Exception as e:
                    print(f"    Error filtering position {position}: {e}")
                    continue
                
                if pos_data.empty:
                    continue
                
                # After merge, column names get suffixed - fix them
                available_cols = list(pos_data.columns)
                
                # Find the actual ADP column after merge (might be suffixed)
                actual_adp_col = None
                for col in available_cols:
                    if adp_rank_col in col:
                        actual_adp_col = col
                        break
                
                # Find the actual points column after merge
                actual_points_col = None
                for col in available_cols:
                    if results_points_col in col:
                        actual_points_col = col
                        break
                
                if not actual_adp_col:
                    print(f"    Error: ADP column '{adp_rank_col}' not found after merge. Available: {available_cols}")
                    continue
                    
                if not actual_points_col:
                    print(f"    Error: Points column '{results_points_col}' not found after merge. Available: {available_cols}")
                    continue
                
                # Convert ADP and points to numeric with the correct column names
                try:
                    pos_data['ADP_Numeric'] = pd.to_numeric(pos_data[actual_adp_col], errors='coerce')
                    pos_data['Points_Numeric'] = pd.to_numeric(pos_data[actual_points_col], errors='coerce')
                except Exception as e:
                    print(f"    Error converting data to numeric: {e}")
                    continue
                
                # Remove invalid data
                pos_data = pos_data.dropna(subset=['ADP_Numeric', 'Points_Numeric'])
                
                if len(pos_data) < 5:  # Need at least 5 players for meaningful analysis
                    continue
                
                # Sort by fantasy points (descending) to get position rankings
                pos_data = pos_data.sort_values('Points_Numeric', ascending=False).reset_index(drop=True)
                
                # Add position rank
                pos_data['Position_Rank'] = range(1, len(pos_data) + 1)
                
                # Store player data
                position_players = []
                for _, row in pos_data.iterrows():
                    # Apply league-specific scoring adjustment
                    original_points = row['Points_Numeric']
                    adjusted_points = self.adjust_points_for_league_scoring(
                        original_points, 
                        row['Player_Clean'], 
                        position, 
                        year
                    )
                    
                    player_data = {
                        'player': row['Player_Clean'],
                        'adp': row['ADP_Numeric'],
                        'points': adjusted_points,  # Use league-adjusted points
                        'position_rank': row['Position_Rank'],
                        'adp_round': self._adp_to_round(row['ADP_Numeric'])
                    }
                    position_players.append(player_data)
                
                year_data[position] = position_players
            
            if year_data:
                historical_data[year] = year_data
                print(f"    ✓ Processed {len(year_data)} positions")
        
        print(f"Historical data built for {len(historical_data)} years")
        return historical_data
    
    def _analyze_position_dropoffs(self, historical_data: Dict) -> List[str]:
        """
        Analyze point drop-offs between position ranks.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Analysis text lines
        """
        analysis = []
        analysis.append("POSITION DROP-OFF ANALYSIS")
        analysis.append("="*50)
        analysis.append("")
        
        # Calculate average drop-offs for each position
        for position in POSITIONS:
            analysis.append(f"{position} POSITION ANALYSIS")
            analysis.append("-" * 30)
            
            # Collect all years' data for this position
            position_data = []
            for year, year_data in historical_data.items():
                if position in year_data:
                    position_data.extend(year_data[position])
            
            if len(position_data) < 10:  # Need sufficient data
                analysis.append(f"Insufficient data for {position} analysis")
                analysis.append("")
                continue
            
            # Calculate key benchmarks
            benchmarks = self._calculate_position_benchmarks(position_data, position)
            
            analysis.append(f"Average Points by Position Rank:")
            for rank_range, avg_points in benchmarks['rank_averages']:
                analysis.append(f"  {position}{rank_range}: {avg_points:.1f} points")
            
            analysis.append(f"\nDrop-off Analysis:")
            for comparison in benchmarks['dropoffs']:
                analysis.append(f"  {comparison['comparison']}: {comparison['dropoff']:.1f} points")
            
            analysis.append(f"\nRound Analysis:")
            for round_info in benchmarks['round_analysis']:
                analysis.append(f"  {round_info}")
            
            analysis.append("")
        
        return analysis
    
    def _analyze_replacement_value(self, historical_data: Dict) -> List[str]:
        """
        Analyze positional replacement value - cost of waiting.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Analysis text lines
        """
        analysis = []
        analysis.append("POSITIONAL REPLACEMENT VALUE ANALYSIS")
        analysis.append("="*50)
        analysis.append("")
        analysis.append("How many points do you lose by waiting to draft each position?")
        analysis.append("")
        
        # Calculate replacement values by round
        replacement_analysis = {}
        
        for position in POSITIONS:
            # Collect all position data
            all_position_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_position_data.extend(year_data[position])
            
            if len(all_position_data) < 10:
                continue
            
            # Group by ADP round
            round_groups = defaultdict(list)
            for player in all_position_data:
                round_groups[player['adp_round']].append(player['points'])
            
            # Calculate average points by round
            round_averages = {}
            for round_num, points_list in round_groups.items():
                if round_num <= 15 and round_num != 99 and len(points_list) >= 3:  # Cap at 15 rounds, exclude invalid data, need at least 3 players per round
                    round_averages[round_num] = np.mean(points_list)
            
            replacement_analysis[position] = round_averages
        
        # Generate replacement value comparisons
        analysis.append("COST OF WAITING BY POSITION")
        analysis.append("-" * 40)
        
        for position in POSITIONS:
            if position not in replacement_analysis:
                continue
                
            analysis.append(f"\n{position} Replacement Cost:")
            
            round_data = replacement_analysis[position]
            sorted_rounds = sorted(round_data.keys())
            
            for i, current_round in enumerate(sorted_rounds[:-1]):
                next_round = sorted_rounds[i + 1]
                current_avg = round_data[current_round]
                next_avg = round_data[next_round]
                cost = current_avg - next_avg
                
                analysis.append(f"  Round {current_round} vs Round {next_round}: {cost:.1f} points lost")
        
        return analysis
    
    def _analyze_opportunity_cost_matrix(self, historical_data: Dict) -> List[str]:
        """
        Create cross-position comparison matrix by round.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Analysis text lines
        """
        analysis = []
        analysis.append("OPPORTUNITY COST MATRIX")
        analysis.append("="*50)
        analysis.append("")
        analysis.append("Cross-position comparisons by draft round")
        analysis.append("")
        
        # Calculate average points by position and round
        position_round_averages = {}
        
        for position in POSITIONS:
            all_position_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_position_data.extend(year_data[position])
            
            if len(all_position_data) < 10:
                continue
            
            # Group by round
            round_groups = defaultdict(list)
            for player in all_position_data:
                if player['adp_round'] <= 15 and player['adp_round'] != 99:  # Exclude invalid data
                    round_groups[player['adp_round']].append(player['points'])
            
            position_round_avg = {}
            for round_num, points in round_groups.items():
                if len(points) >= 3:
                    position_round_avg[round_num] = {
                        'avg': np.mean(points),
                        'count': len(points)
                    }
            
            position_round_averages[position] = position_round_avg
        
        # Generate round-by-round comparisons for early rounds
        for round_num in range(1, 6):  # Rounds 1-5
            analysis.append(f"ROUND {round_num} COMPARISON:")
            analysis.append("-" * 25)
            
            round_values = []
            for position in POSITIONS:
                if position in position_round_averages and round_num in position_round_averages[position]:
                    data = position_round_averages[position][round_num]
                    round_values.append((position, data['avg'], data['count']))
            
            if round_values:
                round_values.sort(key=lambda x: x[1], reverse=True)
                best_pos, best_avg, _ = round_values[0]
                
                for pos, avg, count in round_values:
                    confidence = "HIGH" if count >= 10 else "MED" if count >= 5 else "LOW"
                    advantage = avg - round_values[-1][1] if len(round_values) > 1 else 0
                    analysis.append(f"  {pos}: {avg:.1f} avg points ({count} players, {confidence} confidence)")
                    if pos != round_values[-1][0]:
                        analysis.append(f"       Advantage over worst: +{advantage:.1f} points")
                
                # Show biggest opportunity cost
                if len(round_values) > 1:
                    worst_pos, worst_avg, _ = round_values[-1]
                    opportunity_cost = best_avg - worst_avg
                    analysis.append(f"  → Best choice: {best_pos} (+{opportunity_cost:.1f} points over {worst_pos})")
            
            analysis.append("")
        
        return analysis
    
    def _analyze_risk_adjusted_value(self, historical_data: Dict) -> List[str]:
        """
        Analyze hit rates, bust rates, and risk-adjusted value.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Analysis text lines
        """
        analysis = []
        analysis.append("RISK-ADJUSTED VALUE ANALYSIS")
        analysis.append("="*50)
        analysis.append("")
        
        # Define position tiers for hit rate calculation
        tier_definitions = {
            'QB': 12, 'RB': 24, 'WR': 24, 'TE': 12
        }
        
        for position in POSITIONS:
            analysis.append(f"{position} RISK ANALYSIS")
            analysis.append("-" * 20)
            
            all_position_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_position_data.extend(year_data[position])
            
            if len(all_position_data) < 20:
                analysis.append(f"Insufficient data for {position} risk analysis")
                analysis.append("")
                continue
            
            # Calculate position average for bust rate
            position_avg = np.mean([p['points'] for p in all_position_data])
            tier_cutoff = tier_definitions[position]
            
            # Analyze by early rounds
            for round_num in range(1, 6):
                round_players = [p for p in all_position_data if p['adp_round'] == round_num and p['adp_round'] != 99]
                
                if len(round_players) < 5:
                    continue
                
                # Hit rate (finish in top tier)
                hits = len([p for p in round_players if p['position_rank'] <= tier_cutoff])
                hit_rate = (hits / len(round_players)) * 100
                
                # Bust rates
                bust_50pct = len([p for p in round_players if p['points'] < (position_avg * 0.5)])
                bust_rate_50 = (bust_50pct / len(round_players)) * 100
                
                bust_100pts = len([p for p in round_players if p['points'] < 100])
                bust_rate_100 = (bust_100pts / len(round_players)) * 100
                
                avg_points = np.mean([p['points'] for p in round_players])
                
                analysis.append(f"  Round {round_num} ({len(round_players)} players):")
                analysis.append(f"    Hit Rate (finish top {tier_cutoff}): {hit_rate:.1f}%")
                analysis.append(f"    Bust Rate (<50% of avg): {bust_rate_50:.1f}%")
                analysis.append(f"    Bust Rate (<100 pts): {bust_rate_100:.1f}%")
                analysis.append(f"    Average Points: {avg_points:.1f}")
                analysis.append("")
            
            analysis.append("")
        
        return analysis
    
    def _analyze_positional_scarcity(self, historical_data: Dict) -> List[str]:
        """
        Calculate positional scarcity index.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Analysis text lines
        """
        analysis = []
        analysis.append("POSITIONAL SCARCITY INDEX")
        analysis.append("="*50)
        analysis.append("")
        analysis.append("Scarcity Score = (Elite Tier Avg - Replacement Avg) / Replacement Avg")
        analysis.append("")
        
        scarcity_scores = {}
        
        for position in POSITIONS:
            all_position_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_position_data.extend(year_data[position])
            
            if len(all_position_data) < 50:  # Need more data for deeper replacement levels
                continue
            
            # Sort by points
            sorted_data = sorted(all_position_data, key=lambda x: x['points'], reverse=True)
            
            # Define elite tier (top 12) and TRUE replacement level (waiver wire)
            if position == 'QB':
                elite_tier = sorted_data[:12]
                replacement_tier = sorted_data[15:20]  # QBs 16-20 (true waiver wire)
            elif position in ['RB', 'WR']:
                elite_tier = sorted_data[:12]
                replacement_tier = sorted_data[36:48]  # RBs/WRs 37-48 (true waiver wire)
            else:  # TE
                elite_tier = sorted_data[:12]
                replacement_tier = sorted_data[15:20]  # TEs 16-20 (true waiver wire)
            
            # Use replacement tier if we have enough data, otherwise use fallback
            if replacement_tier and len(replacement_tier) >= 3:
                replacement_avg = np.mean([p['points'] for p in replacement_tier])
            else:
                # Fallback: Use a larger but shallower replacement tier
                if position == 'QB':
                    fallback_tier = sorted_data[12:20]  # QBs 13-20
                elif position in ['RB', 'WR']:
                    fallback_tier = sorted_data[24:36]  # RBs/WRs 25-36 
                else:  # TE
                    fallback_tier = sorted_data[12:20]  # TEs 13-20
                
                if fallback_tier and len(fallback_tier) >= 3:
                    replacement_avg = np.mean([p['points'] for p in fallback_tier])
                else:
                    continue  # Skip if we don't have enough data
            
            if elite_tier:
                elite_avg = np.mean([p['points'] for p in elite_tier])
                
                scarcity_score = (elite_avg - replacement_avg) / replacement_avg
                scarcity_scores[position] = {
                    'score': scarcity_score,
                    'elite_avg': elite_avg,
                    'replacement_avg': replacement_avg
                }
        
        # Sort by scarcity score
        sorted_scarcity = sorted(scarcity_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        analysis.append("SCARCITY RANKINGS (Most to Least Scarce):")
        analysis.append("-" * 45)
        
        for i, (position, data) in enumerate(sorted_scarcity):
            score = data['score']
            elite_avg = data['elite_avg']
            replacement_avg = data['replacement_avg']
            
            analysis.append(f"{i+1}. {position}")
            analysis.append(f"   Scarcity Score: {score:.2f} ({score*100:.0f}% above replacement)")
            analysis.append(f"   Elite Tier (Top 12): {elite_avg:.1f} avg points")
            analysis.append(f"   Replacement Level: {replacement_avg:.1f} avg points")
            analysis.append("")
        
        return analysis
    
    def _analyze_value_over_replacement(self, historical_data: Dict) -> List[str]:
        """
        Calculate value over replacement by round.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Analysis text lines
        """
        analysis = []
        analysis.append("VALUE OVER REPLACEMENT BY ROUND")
        analysis.append("="*50)
        analysis.append("")
        
        # Calculate replacement level for each position
        replacement_levels = {}
        
        for position in POSITIONS:
            all_position_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_position_data.extend(year_data[position])
            
            if len(all_position_data) < 50:  # Need more data for deeper replacement levels
                continue
            
            # Sort by points and define TRUE replacement level (waiver wire players)
            sorted_data = sorted(all_position_data, key=lambda x: x['points'], reverse=True)
            
            if position == 'QB':
                replacement_tier = sorted_data[15:20]  # QBs 16-20 (true waiver wire)
            elif position in ['RB', 'WR']:
                replacement_tier = sorted_data[36:48]  # RBs/WRs 37-48 (true waiver wire)
            else:  # TE
                replacement_tier = sorted_data[15:20]  # TEs 16-20 (true waiver wire)
            
            if replacement_tier and len(replacement_tier) >= 3:  # Need at least 3 players for stable average
                replacement_levels[position] = np.mean([p['points'] for p in replacement_tier])
            else:
                # Fallback: Use a larger but shallower replacement tier
                if position == 'QB':
                    fallback_tier = sorted_data[12:20]  # QBs 13-20
                elif position in ['RB', 'WR']:
                    fallback_tier = sorted_data[24:36]  # RBs/WRs 25-36 
                else:  # TE
                    fallback_tier = sorted_data[12:20]  # TEs 13-20
                
                if fallback_tier and len(fallback_tier) >= 3:
                    replacement_levels[position] = np.mean([p['points'] for p in fallback_tier])
        
        # Calculate VOR by round for early rounds
        for round_num in range(1, 6):
            analysis.append(f"ROUND {round_num} VALUE OVER REPLACEMENT:")
            analysis.append("-" * 35)
            
            vor_values = []
            
            for position in POSITIONS:
                if position not in replacement_levels:
                    continue
                
                all_position_data = []
                for year_data in historical_data.values():
                    if position in year_data:
                        all_position_data.extend(year_data[position])
                
                round_players = [p for p in all_position_data if p['adp_round'] == round_num and p['adp_round'] != 99]
                
                if len(round_players) >= 3:
                    round_avg = np.mean([p['points'] for p in round_players])
                    replacement = replacement_levels[position]
                    vor_percentage = ((round_avg - replacement) / replacement) * 100
                    vor_points = round_avg - replacement
                    
                    vor_values.append((position, vor_percentage, vor_points, round_avg, len(round_players)))
            
            # Sort by VOR percentage
            vor_values.sort(key=lambda x: x[1], reverse=True)
            
            for pos, vor_pct, vor_pts, avg_pts, count in vor_values:
                confidence = "HIGH" if count >= 10 else "MED" if count >= 5 else "LOW"
                analysis.append(f"  {pos}: {vor_pct:.0f}% above replacement (+{vor_pts:.1f} pts)")
                analysis.append(f"       {avg_pts:.1f} avg points vs {replacement_levels[pos]:.1f} replacement")
                analysis.append(f"       ({count} players, {confidence} confidence)")
                analysis.append("")
            
            analysis.append("")
        
        return analysis
    
    def _generate_draft_recommendations(self, historical_data: Dict) -> List[str]:
        """
        Generate draft strategy recommendations based on analysis.
        
        Args:
            historical_data: Historical position rankings
            
        Returns:
            Recommendation text lines
        """
        analysis = []
        analysis.append("DRAFT STRATEGY RECOMMENDATIONS")
        analysis.append("="*50)
        analysis.append("")
        
        # Calculate position values for early rounds
        early_round_analysis = self._analyze_early_round_values(historical_data)
        
        analysis.append("ROUND 1-3 POSITIONAL STRATEGY:")
        analysis.append("-" * 30)
        for rec in early_round_analysis:
            analysis.append(f"  • {rec}")
        
        analysis.append("")
        analysis.append("KEY INSIGHTS:")
        analysis.append("-" * 15)
        
        # Generate key insights
        insights = self._generate_key_insights(historical_data)
        for insight in insights:
            analysis.append(f"  • {insight}")
        
        return analysis
    
    def _calculate_position_benchmarks(self, position_data: List[Dict], position: str) -> Dict:
        """Calculate key benchmarks for a position."""
        # Sort by points
        sorted_data = sorted(position_data, key=lambda x: x['points'], reverse=True)
        
        # Define rank ranges based on position
        if position == 'QB':
            rank_ranges = [(1, 5), (6, 12), (13, 24)]
            range_labels = ['1-5', '6-12', '13-24']
        elif position in ['RB', 'WR']:
            rank_ranges = [(1, 5), (6, 12), (13, 24), (25, 36)]
            range_labels = ['1-5', '6-12', '13-24', '25-36']
        else:  # TE
            rank_ranges = [(1, 3), (4, 8), (9, 16)]
            range_labels = ['1-3', '4-8', '9-16']
        
        # Calculate averages for each range
        rank_averages = []
        for i, (start, end) in enumerate(rank_ranges):
            players_in_range = [p for p in sorted_data if start <= p['position_rank'] <= end]
            if players_in_range:
                avg_points = np.mean([p['points'] for p in players_in_range])
                rank_averages.append((range_labels[i], avg_points))
        
        # Calculate drop-offs
        dropoffs = []
        for i in range(len(rank_averages) - 1):
            current_range, current_avg = rank_averages[i]
            next_range, next_avg = rank_averages[i + 1]
            dropoff = current_avg - next_avg
            dropoffs.append({
                'comparison': f"{position}{current_range} vs {position}{next_range}",
                'dropoff': dropoff
            })
        
        # Round analysis
        round_analysis = []
        round_groups = defaultdict(list)
        for player in position_data:
            round_groups[player['adp_round']].append(player['points'])
        
        for round_num in sorted(round_groups.keys()):
            if round_num <= 15 and round_num != 99 and len(round_groups[round_num]) >= 3:  # Exclude round 99 (invalid data)
                avg_points = np.mean(round_groups[round_num])
                count = len(round_groups[round_num])
                confidence = "HIGH" if count >= 10 else "MED" if count >= 5 else "LOW"
                round_analysis.append(f"Round {round_num}: {avg_points:.1f} avg points ({count} players, {confidence} confidence)")
        
        return {
            'rank_averages': rank_averages,
            'dropoffs': dropoffs,
            'round_analysis': round_analysis
        }
    
    def _analyze_early_round_values(self, historical_data: Dict) -> List[str]:
        """Analyze early round positional values."""
        recommendations = []
        
        # Calculate average points for each position in rounds 1-3
        round_averages = {}
        
        for position in POSITIONS:
            all_position_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_position_data.extend(year_data[position])
            
            if len(all_position_data) < 5:
                continue
            
            # Group by round
            round_groups = defaultdict(list)
            for player in all_position_data:
                if player['adp_round'] <= 3:  # Early rounds only
                    round_groups[player['adp_round']].append(player['points'])
            
            position_round_avg = {}
            for round_num, points in round_groups.items():
                if len(points) >= 2:
                    position_round_avg[round_num] = np.mean(points)
            
            round_averages[position] = position_round_avg
        
        # Generate recommendations based on drop-offs
        for round_num in [1, 2, 3]:
            round_values = []
            for position in POSITIONS:
                if position in round_averages and round_num in round_averages[position]:
                    round_values.append((position, round_averages[position][round_num]))
            
            if round_values:
                round_values.sort(key=lambda x: x[1], reverse=True)
                best_pos, best_avg = round_values[0]
                recommendations.append(f"Round {round_num}: {best_pos} historically strongest ({best_avg:.1f} avg points)")
        
        return recommendations
    
    def _generate_key_insights(self, historical_data: Dict) -> List[str]:
        """Generate key strategic insights."""
        insights = []
        
        # Calculate position scarcity (drop-off between top tiers)
        scarcity_analysis = {}
        
        for position in POSITIONS:
            all_data = []
            for year_data in historical_data.values():
                if position in year_data:
                    all_data.extend(year_data[position][:12])  # Top 12 at each position
            
            if len(all_data) < 10:
                continue
            
            # Sort by points and calculate top tier vs second tier
            sorted_data = sorted(all_data, key=lambda x: x['points'], reverse=True)
            
            if position == 'QB':
                top_tier = sorted_data[:5]
                second_tier = sorted_data[5:12]
            elif position in ['RB', 'WR']:
                top_tier = sorted_data[:6]
                second_tier = sorted_data[6:12]
            else:  # TE
                top_tier = sorted_data[:3]
                second_tier = sorted_data[3:8]
            
            if top_tier and second_tier:
                top_avg = np.mean([p['points'] for p in top_tier])
                second_avg = np.mean([p['points'] for p in second_tier])
                scarcity_analysis[position] = top_avg - second_avg
        
        # Generate insights based on scarcity
        if scarcity_analysis:
            most_scarce = max(scarcity_analysis.items(), key=lambda x: x[1])
            least_scarce = min(scarcity_analysis.items(), key=lambda x: x[1])
            
            insights.append(f"{most_scarce[0]} shows highest scarcity ({most_scarce[1]:.1f} point drop-off)")
            insights.append(f"{least_scarce[0]} shows lowest scarcity ({least_scarce[1]:.1f} point drop-off)")
        
        # Add general insights
        insights.append("Focus on positions with largest point differentials between tiers")
        insights.append("Consider waiting on positions with smaller drop-offs")
        
        return insights
    
    def _get_player_column(self, df: pd.DataFrame, data_type: str) -> Optional[str]:
        """Get the player name column."""
        possible_columns = ['Player', 'PLAYER', 'player', 'Name', 'NAME', 'name']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    
    def _get_position_column(self, df: pd.DataFrame, data_type: str) -> Optional[str]:
        """Get the position column."""
        possible_columns = ['Pos', 'POS', 'Position', 'POSITION', 'pos', 'position']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    
    def _get_adp_column(self, df: pd.DataFrame, data_type: str) -> Optional[str]:
        """Get the ADP ranking column."""
        possible_columns = ['AVG', 'Rank', 'RANK', 'rank', 'ADP', 'adp']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    
    def _get_points_column(self, df: pd.DataFrame, data_type: str) -> Optional[str]:
        """Get the fantasy points column."""
        possible_columns = ['TTL', 'Total', 'TOTAL', 'total', 'Points', 'POINTS', 'points']
        for col in possible_columns:
            if col in df.columns:
                return col
        return None
    
    def _adp_to_round(self, adp: float, league_size: int = 12) -> int:
        """Convert ADP to draft round."""
        if pd.isna(adp) or adp <= 0 or adp > 200:  # Filter out unrealistic ADPs
            return 99
        round_num = int((adp - 1) // league_size) + 1
        return round_num if round_num <= 15 else 99  # Cap at 15 rounds
    
    def _get_player_raw_stats(self, player_name: str, position: str, year: int) -> Dict:
        """
        Get raw stats data for a player including TDs and explosive plays.
        
        Args:
            player_name: Clean player name
            position: Player position
            year: Year of data
            
        Returns:
            Dictionary with raw stats data
        """
        stats_data = {}
        
        # Try to get advanced metrics data (for explosive plays)
        if hasattr(self.data_loader, 'advanced_data'):
            pos_key = f"{position}_{year}"
            if pos_key in self.data_loader.advanced_data:
                adv_data = self.data_loader.advanced_data[pos_key]
                player_adv = adv_data[adv_data['PLAYER'].str.contains(player_name, case=False, na=False)]
                if not player_adv.empty:
                    stats_data['40+ YDS'] = player_adv.iloc[0].get('40+ YDS', 0)
        
        # Try to get stats data (for TDs) - load stats file if needed
        try:
            import pandas as pd
            import os
            
            # Construct stats file path
            stats_file = os.path.join('fantasy_data', f'stats_{position.lower()}_{year}_half.csv')
            if os.path.exists(stats_file):
                stats_df = pd.read_csv(stats_file)
                
                # Handle multi-level headers - the player names are in the second column
                player_col = None
                if 'Unnamed: 1_level_0' in stats_df.columns:
                    player_col = 'Unnamed: 1_level_0'
                elif 'Player' in stats_df.columns:
                    player_col = 'Player'
                elif 'PLAYER' in stats_df.columns:
                    player_col = 'PLAYER'
                
                if player_col:
                    # Clean player name matching
                    player_stats = stats_df[stats_df[player_col].astype(str).str.contains(player_name, case=False, na=False)]
                    if not player_stats.empty:
                        # For QBs, get passing TDs from PASSING.5 column (TD column)
                        if position.lower() == 'qb':
                            # QB passing TDs are in 'PASSING.5' based on the data structure
                            if 'PASSING.5' in stats_df.columns:
                                stats_data['TD'] = pd.to_numeric(player_stats.iloc[0]['PASSING.5'], errors='coerce') or 0
                        else:
                            # For skill positions, sum rushing and receiving TDs
                            # RUSHING.2 is typically rushing TDs
                            rushing_tds = 0
                            if 'RUSHING.2' in stats_df.columns:
                                rushing_tds = pd.to_numeric(player_stats.iloc[0]['RUSHING.2'], errors='coerce') or 0
                            
                            # For WRs/TEs, also look for receiving TDs (usually in different files)
                            stats_data['TD'] = rushing_tds
                            
        except Exception as e:
            print(f"Warning: Could not load stats data for {player_name}: {e}")
            
        return stats_data
    
    def adjust_points_for_league_scoring(self, original_points: float, player_name: str, position: str, year: int) -> float:
        """
        Adjust fantasy points from 4pt passing TDs to 6pt passing TDs and add explosive play bonuses.
        
        Args:
            original_points: Original fantasy points (4pt passing TD system)
            player_name: Player name for stats lookup
            position: Player position
            year: Year of data
            
        Returns:
            Adjusted fantasy points for league scoring
        """
        adjusted_points = original_points
        
        # Get raw stats for this player
        raw_stats = self._get_player_raw_stats(player_name, position, year)
        
        if position.lower() == 'qb':
            # Adjust for 6pt passing TDs instead of 4pt
            pass_tds = raw_stats.get('TD', 0)
            td_adjustment = pass_tds * 2  # +2 points per passing TD
            adjusted_points += td_adjustment
            
            # Add explosive play bonus for QBs (40+ yard passes that become TDs)
            explosive_passes = raw_stats.get('40+ YDS', 0)
            # Estimate that ~15% of 40+ yard passes are TDs
            estimated_explosive_tds = explosive_passes * 0.15
            explosive_bonus = estimated_explosive_tds * 2  # +2 points per explosive TD
            adjusted_points += explosive_bonus
            
        else:
            # For skill positions, add explosive play bonus (40+ yard TDs)
            explosive_plays = raw_stats.get('40+ YDS', 0)
            # Estimate that ~25% of 40+ yard plays are TDs
            estimated_explosive_tds = explosive_plays * 0.25
            explosive_bonus = estimated_explosive_tds * 2  # +2 points per explosive TD
            adjusted_points += explosive_bonus
            
        return adjusted_points