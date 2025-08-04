"""
League Analysis Module for Fantasy Football Analyzer

Handles league-specific analysis and manager tendencies.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from collections import defaultdict

from config.constants import (
    SIGNIFICANT_REACH_THRESHOLD, SIGNIFICANT_VALUE_THRESHOLD,
    ADP_OUTPERFORM_THRESHOLD, MIN_PICKS_FOR_ANALYSIS
)


class LeagueAnalyzer:
    """
    Handles league-specific analysis and manager tendencies.
    """
    
    def __init__(self, data_loader):
        """
        Initialize the league analyzer.
        
        Args:
            data_loader: DataLoader instance
        """
        self.data_loader = data_loader
    
    def analyze_draft_reaches_and_values(self) -> str:
        """
        Analyze which managers reach for players vs find values, incorporating actual player performance.
        
        Returns:
            Analysis report as string
        """
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
        
        for year in self.data_loader.years:
            if year not in self.data_loader.draft_data or year not in self.data_loader.adp_data or year not in self.data_loader.results_data:
                continue
                
            draft_df = self.data_loader.draft_data[year]
            adp_df = self.data_loader.adp_data[year]
            results_df = self.data_loader.results_data[year]
            
            # Get column names
            draft_player_col = self.data_loader.get_player_column(draft_df, f"Draft {year}")
            adp_player_col = self.data_loader.get_player_column(adp_df, f"ADP {year}")
            adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {year}")
            results_player_col = self.data_loader.get_player_column(results_df, f"Results {year}")
            
            if not all([draft_player_col, adp_player_col, adp_col, results_player_col]):
                continue
            
            # Clean player names
            draft_df['Player_Clean'] = draft_df[draft_player_col].apply(self.data_loader.clean_player_name)
            adp_df['Player_Clean'] = adp_df[adp_player_col].apply(self.data_loader.clean_player_name)
            results_df['Player_Clean'] = results_df[results_player_col].apply(self.data_loader.clean_player_name)
            
            # Merge all three datasets
            merged = pd.merge(draft_df, adp_df, on='Player_Clean', how='inner')
            merged = pd.merge(merged, results_df, on='Player_Clean', how='inner')
            
            for _, pick in merged.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                draft_pick = pick.get('Pick_No', pick.get('pick_no', 0))
                adp = self.data_loader.safe_float_conversion(pick.get(adp_col))
                
                # Get final season ranking - use '#' column from results
                final_rank = self.data_loader.safe_float_conversion(pick.get('#', 0))
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
                if adp_vs_draft >= SIGNIFICANT_REACH_THRESHOLD:
                    manager_draft_data[manager]['adp_reaches'].append(pick_data)
                elif adp_vs_draft <= SIGNIFICANT_VALUE_THRESHOLD:
                    manager_draft_data[manager]['adp_values'].append(pick_data)
                
                # Track biggest steals and busts
                current_steal = manager_draft_data[manager]['biggest_steal']
                if not current_steal or draft_vs_finish > current_steal['draft_vs_finish']:
                    manager_draft_data[manager]['biggest_steal'] = pick_data
                
                current_bust = manager_draft_data[manager]['biggest_bust']
                if not current_bust or draft_vs_finish < current_bust['draft_vs_finish']:
                    manager_draft_data[manager]['biggest_bust'] = pick_data
        
        # Generate analysis report
        analysis_text.append("DRAFT REACHES AND VALUES ANALYSIS")
        analysis_text.append("="*60)
        
        for manager, data in manager_draft_data.items():
            if data['total_picks'] < MIN_PICKS_FOR_ANALYSIS:
                continue
            
            analysis_text.append(f"\n{manager.upper()} - DRAFT ANALYSIS")
            analysis_text.append("-" * 40)
            analysis_text.append(f"Total Picks: {data['total_picks']}")
            
            # Performance vs draft position
            if data['performance_vs_draft']:
                avg_performance = np.mean(data['performance_vs_draft'])
                analysis_text.append(f"Average Performance vs Draft Position: {avg_performance:+.1f} picks")
                analysis_text.append(f"  (Positive = outperformed draft position)")
            
            # Performance vs ADP
            if data['performance_vs_adp']:
                avg_adp_performance = np.mean(data['performance_vs_adp'])
                analysis_text.append(f"Average Performance vs ADP: {avg_adp_performance:+.1f} picks")
                analysis_text.append(f"  (Positive = outperformed ADP)")
            
            # Good vs bad picks
            good_picks = len(data['good_picks'])
            bad_picks = len(data['bad_picks'])
            total_picks = data['total_picks']
            
            analysis_text.append(f"\nPick Quality:")
            analysis_text.append(f"  Good Picks: {good_picks} ({good_picks/total_picks*100:.1f}%)")
            analysis_text.append(f"  Bad Picks: {bad_picks} ({bad_picks/total_picks*100:.1f}%)")
            
            # ADP reaches and values
            reaches = len(data['adp_reaches'])
            values = len(data['adp_values'])
            analysis_text.append(f"\nADP Analysis:")
            analysis_text.append(f"  Reaches: {reaches} ({reaches/total_picks*100:.1f}%)")
            analysis_text.append(f"  Values: {values} ({values/total_picks*100:.1f}%)")
            
            # Biggest steal and bust
            if data['biggest_steal']:
                steal = data['biggest_steal']
                analysis_text.append(f"\nBiggest Steal: {steal['player']} (Drafted {steal['draft_pick']}, Finished {steal['final_rank']}, +{steal['draft_vs_finish']} picks)")
            
            if data['biggest_bust']:
                bust = data['biggest_bust']
                analysis_text.append(f"Biggest Bust: {bust['player']} (Drafted {bust['draft_pick']}, Finished {bust['final_rank']}, {bust['draft_vs_finish']} picks)")
        
        return "\n".join(analysis_text)
    
    def analyze_league_mate_tendencies(self) -> str:
        """
        Analyze league mate draft tendencies and patterns.
        
        Returns:
            Analysis report as string
        """
        print("\n" + "="*60)
        print("LEAGUE MATE TENDENCIES ANALYSIS")
        print("="*60)
        
        analysis_text = []
        manager_tendencies = self._get_manager_tendencies()
        
        analysis_text.append("LEAGUE MATE DRAFT TENDENCIES")
        analysis_text.append("="*60)
        
        for manager, tendencies in manager_tendencies.items():
            if tendencies['total_picks'] < MIN_PICKS_FOR_ANALYSIS:
                continue
            
            analysis_text.append(f"\n{manager.upper()} - TENDENCIES")
            analysis_text.append("-" * 40)
            
            # Position preferences
            if tendencies['position_preferences']:
                analysis_text.append("Position Preferences:")
                for pos, count in sorted(tendencies['position_preferences'].items(), key=lambda x: x[1], reverse=True):
                    percentage = count / tendencies['total_picks'] * 100
                    analysis_text.append(f"  {pos}: {count} picks ({percentage:.1f}%)")
            
            # Round preferences
            if tendencies['round_preferences']:
                analysis_text.append("\nRound Preferences:")
                for round_num, count in sorted(tendencies['round_preferences'].items()):
                    percentage = count / tendencies['total_picks'] * 100
                    analysis_text.append(f"  Round {round_num}: {count} picks ({percentage:.1f}%)")
            
            # Performance patterns
            if tendencies['performance_patterns']:
                analysis_text.append("\nPerformance Patterns:")
                for pattern, count in tendencies['performance_patterns'].items():
                    percentage = count / tendencies['total_picks'] * 100
                    analysis_text.append(f"  {pattern}: {count} picks ({percentage:.1f}%)")
        
        return "\n".join(analysis_text)
    
    def analyze_league_draft_flow(self) -> str:
        """
        Analyze league draft flow and patterns.
        
        Returns:
            Analysis report as string
        """
        print("\n" + "="*60)
        print("LEAGUE DRAFT FLOW ANALYSIS")
        print("="*60)
        
        analysis_text = []
        
        analysis_text.append("LEAGUE DRAFT FLOW ANALYSIS")
        analysis_text.append("="*60)
        
        # Analyze draft flow by round
        for year in self.data_loader.years:
            if year not in self.data_loader.draft_data:
                continue
            
            draft_df = self.data_loader.draft_data[year]
            analysis_text.append(f"\n{year} DRAFT FLOW")
            analysis_text.append("-" * 30)
            
            # Group picks by round
            round_picks = defaultdict(list)
            for _, pick in draft_df.iterrows():
                pick_no = pick.get('Pick_No', pick.get('pick_no', 0))
                if pick_no > 0:
                    round_num = ((pick_no - 1) // 12) + 1  # Assuming 12-team league
                    round_picks[round_num].append(pick)
            
            # Analyze each round
            for round_num in sorted(round_picks.keys()):
                picks = round_picks[round_num]
                analysis_text.append(f"\nRound {round_num}:")
                
                # Position breakdown
                position_counts = defaultdict(int)
                for pick in picks:
                    position = pick.get('Position', pick.get('POS', 'Unknown'))
                    position_counts[position] += 1
                
                for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
                    percentage = count / len(picks) * 100
                    analysis_text.append(f"  {pos}: {count} picks ({percentage:.1f}%)")
        
        return "\n".join(analysis_text)
    
    def _get_manager_tendencies(self) -> Dict[str, Dict]:
        """
        Get manager draft tendencies.
        
        Returns:
            Dictionary of manager tendencies
        """
        manager_tendencies = defaultdict(lambda: {
            'total_picks': 0,
            'position_preferences': defaultdict(int),
            'round_preferences': defaultdict(int),
            'performance_patterns': defaultdict(int)
        })
        
        for year in self.data_loader.years:
            if year not in self.data_loader.draft_data:
                continue
            
            draft_df = self.data_loader.draft_data[year]
            
            for _, pick in draft_df.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', 'Unknown'))
                position = pick.get('Position', pick.get('POS', 'Unknown'))
                pick_no = pick.get('Pick_No', pick.get('pick_no', 0))
                
                if pick_no > 0:
                    round_num = ((pick_no - 1) // 12) + 1  # Assuming 12-team league
                    
                    manager_tendencies[manager]['total_picks'] += 1
                    manager_tendencies[manager]['position_preferences'][position] += 1
                    manager_tendencies[manager]['round_preferences'][round_num] += 1
                    
                    # Performance pattern (simplified)
                    if pick_no <= 36:  # First 3 rounds
                        manager_tendencies[manager]['performance_patterns']['Early Rounds'] += 1
                    elif pick_no <= 84:  # Rounds 4-7
                        manager_tendencies[manager]['performance_patterns']['Middle Rounds'] += 1
                    else:  # Later rounds
                        manager_tendencies[manager]['performance_patterns']['Late Rounds'] += 1
        
        return dict(manager_tendencies)
    
    def analyze_roneill19_tendencies(self) -> str:
        """
        Analyze specific manager (roneill19) tendencies.
        
        Returns:
            Analysis report as string
        """
        print("\n" + "="*60)
        print("RONEILL19 TENDENCIES ANALYSIS")
        print("="*60)
        
        analysis_text = []
        
        analysis_text.append("RONEILL19 DRAFT TENDENCIES")
        analysis_text.append("="*60)
        
        # Get all picks for roneill19
        roneill19_picks = []
        
        for year in self.data_loader.years:
            if year not in self.data_loader.draft_data:
                continue
            
            draft_df = self.data_loader.draft_data[year]
            
            for _, pick in draft_df.iterrows():
                manager = pick.get('Picked_By_Username', pick.get('picked_by_username', ''))
                if 'roneill19' in manager.lower():
                    roneill19_picks.append({
                        'year': year,
                        'player': pick.get('Player', pick.get('PLAYER', 'Unknown')),
                        'position': pick.get('Position', pick.get('POS', 'Unknown')),
                        'pick_no': pick.get('Pick_No', pick.get('pick_no', 0)),
                        'round': ((pick.get('Pick_No', pick.get('pick_no', 0)) - 1) // 12) + 1
                    })
        
        if not roneill19_picks:
            analysis_text.append("No picks found for roneill19")
            return "\n".join(analysis_text)
        
        # Analyze tendencies
        total_picks = len(roneill19_picks)
        analysis_text.append(f"Total Picks: {total_picks}")
        
        # Position breakdown
        position_counts = defaultdict(int)
        for pick in roneill19_picks:
            position_counts[pick['position']] += 1
        
        analysis_text.append("\nPosition Preferences:")
        for pos, count in sorted(position_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / total_picks * 100
            analysis_text.append(f"  {pos}: {count} picks ({percentage:.1f}%)")
        
        # Round breakdown
        round_counts = defaultdict(int)
        for pick in roneill19_picks:
            round_counts[pick['round']] += 1
        
        analysis_text.append("\nRound Preferences:")
        for round_num in sorted(round_counts.keys()):
            count = round_counts[round_num]
            percentage = count / total_picks * 100
            analysis_text.append(f"  Round {round_num}: {count} picks ({percentage:.1f}%)")
        
        # Recent picks
        analysis_text.append("\nRecent Picks (Last 10):")
        recent_picks = sorted(roneill19_picks, key=lambda x: (x['year'], x['pick_no']))[-10:]
        for pick in recent_picks:
            analysis_text.append(f"  {pick['year']} Round {pick['round']}: {pick['player']} ({pick['position']})")
        
        return "\n".join(analysis_text)
    
    def calculate_league_position_averages(self) -> Dict[str, float]:
        """
        Calculate league-wide position averages.
        
        Returns:
            Dictionary of position averages
        """
        position_averages = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            all_adps = []
            
            for year in self.data_loader.years:
                if year not in self.data_loader.adp_data:
                    continue
                
                adp_df = self.data_loader.adp_data[year]
                adp_col = self.data_loader.get_adp_column(adp_df, f"ADP {year}")
                pos_col = self.data_loader.get_position_column(adp_df, f"ADP {year}")
                
                if adp_col and pos_col:
                    pos_data = adp_df[adp_df[pos_col].notna() & adp_df[pos_col].str.upper().str.startswith(position.upper())]
                    for _, row in pos_data.iterrows():
                        adp = self.data_loader.safe_float_conversion(row.get(adp_col))
                        if adp > 0:
                            all_adps.append(adp)
            
            if all_adps:
                position_averages[position] = np.mean(all_adps)
        
        return position_averages 