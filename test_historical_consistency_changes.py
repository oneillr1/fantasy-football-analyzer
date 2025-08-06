#!/usr/bin/env python3
"""
Test script to verify historical scoring changes:
1. Removal of peak score from historical scoring
2. Addition of average PPG score to historical scoring
3. New mean-to-std ratio consistency metric
4. Peak PPG still available for ML models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

def test_historical_consistency_changes():
    """Test the new historical scoring system with mean-to-std ratio consistency."""
    
    # Initialize the analyzer
    analyzer = FantasyFootballAnalyzer("fantasy_data")
    
    # Test players with different profiles
    test_players = [
        ("Patrick Mahomes", "QB"),
        ("Christian McCaffrey", "RB"),
        ("Tyreek Hill", "WR"),
        ("Travis Kelce", "TE"),
        ("Josh Allen", "QB"),
        ("Saquon Barkley", "RB"),
        ("Stefon Diggs", "WR"),
        ("Mark Andrews", "TE")
    ]
    
    print("=== TESTING HISTORICAL CONSISTENCY CHANGES ===")
    print("Testing new mean-to-std ratio consistency and average PPG scoring\n")
    
    for player, position in test_players:
        print(f"\n--- {player} ({position}) ---")
        
        try:
            # Get historical performance data
            historical_data = analyzer.player_analyzer.calculate_historical_performance(player, position)
            
            if historical_data.get('has_historical_data'):
                print(f"Years of data: {historical_data.get('years_of_data', 0)}")
                print(f"ADP differential: {historical_data.get('adp_differential', 0):.2f}")
                print(f"Peak PPG: {historical_data.get('peak_ppg', 0):.2f}")
                print(f"Average PPG: {historical_data.get('avg_ppg', 0):.2f}")
                print(f"PPG Consistency (mean/std ratio): {historical_data.get('ppg_consistency', 0):.2f}")
                print(f"Injury adjustment: {historical_data.get('injury_adjustment', 0):.2f}")
                
                # Test the new historical scoring
                historical_score = analyzer.scoring_engine._calculate_enhanced_historical_score(
                    historical_data, position)
                print(f"Historical Score: {historical_score:.2f}")
                
                # Test that peak PPG is still available for ML models
                peak_score = analyzer.scoring_engine._calculate_position_specific_peak_score(
                    historical_data.get('peak_ppg', 0), position)
                print(f"Peak Score (for ML): {peak_score:.2f}")
                
                # Test the new average PPG score
                avg_ppg_score = analyzer.scoring_engine._calculate_position_specific_avg_ppg_score(
                    historical_data.get('avg_ppg', 0), position)
                print(f"Average PPG Score: {avg_ppg_score:.2f}")
                
            else:
                print("No historical data available")
                
        except Exception as e:
            print(f"Error processing {player}: {e}")
    
    # Save debug logs
    analyzer.data_loader.save_debug_logs()
    print(f"\nDebug logs saved successfully")

if __name__ == "__main__":
    test_historical_consistency_changes() 