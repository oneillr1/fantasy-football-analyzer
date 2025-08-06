#!/usr/bin/env python3
"""
Test script to verify peak PPG changes and new ML scoring weights.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyAnalyzer

def test_peak_ppg_changes():
    """Test the new peak PPG calculation and ML scoring changes."""
    print("Testing Peak PPG Changes and New ML Scoring Weights")
    print("=" * 60)
    
    # Initialize the analyzer
    analyzer = FantasyAnalyzer()
    
    # Test players from each position
    test_players = [
        ("Patrick Mahomes", "QB"),
        ("Josh Allen", "QB"),
        ("Christian McCaffrey", "RB"),
        ("Breece Hall", "RB"),
        ("Tyreek Hill", "WR"),
        ("CeeDee Lamb", "WR"),
        ("Travis Kelce", "TE"),
        ("Sam LaPorta", "TE")
    ]
    
    for player, position in test_players:
        print(f"\n{player} ({position})")
        print("-" * 40)
        
        try:
            # Get historical performance
            historical_perf = analyzer.player_analyzer.calculate_historical_performance(player, position)
            
            print(f"Historical Data:")
            print(f"  Years of data: {historical_perf.get('years_of_data', 0)}")
            print(f"  ADP differential: {historical_perf.get('adp_differential', 'N/A')}")
            print(f"  Peak season (total): {historical_perf.get('peak_season', 'N/A')}")
            print(f"  Peak PPG: {historical_perf.get('peak_ppg', 'N/A')}")  # NEW
            print(f"  PPG consistency: {historical_perf.get('ppg_consistency', 'N/A')}")
            print(f"  Injury adjustment: {historical_perf.get('injury_adjustment', 'N/A')}")
            
            # Calculate historical score with new peak PPG
            if historical_perf.get('has_historical_data', False):
                hist_score = analyzer.scoring_engine._calculate_enhanced_historical_score(
                    historical_perf, position)
                print(f"  Historical Score: {hist_score:.2f}")
                
                # Test peak score calculation
                peak_ppg = historical_perf.get('peak_ppg')
                if peak_ppg is not None:
                    peak_score = analyzer.scoring_engine._calculate_position_specific_peak_score(
                        peak_ppg, position)
                    print(f"  Peak Score (PPG-based): {peak_score:.2f}")
            
            # Get ML predictions
            if position in analyzer.ml_models.ml_models:
                # Get advanced data for feature extraction
                advanced_data = analyzer.data_loader.get_advanced_data(player, position, 2024)
                if advanced_data is not None:
                    features = analyzer.ml_models._extract_enhanced_features_with_age_injury(
                        advanced_data, position, player, 2024)
                    
                    if features is not None:
                        ml_predictions = analyzer.ml_models.generate_predictions(
                            player, position, features, debug_mode=True)
                        
                        print(f"ML Predictions:")
                        print(f"  Breakout: {ml_predictions.get('breakout', 'N/A')}")
                        print(f"  Consistency: {ml_predictions.get('consistency', 'N/A')}")
                        print(f"  Positional Value: {ml_predictions.get('positional_value', 'N/A')}")
                        print(f"  Injury Risk: {ml_predictions.get('injury_risk', 'N/A')}")
                        
                        # Calculate ML score with new weights
                        ml_score = analyzer.scoring_engine._calculate_ml_predictions_score(
                            ml_predictions, player, position)
                        print(f"  ML Score (new weights): {ml_score:.2f}")
            
        except Exception as e:
            print(f"Error processing {player}: {e}")
    
    # Save debug logs
    analyzer.data_loader.save_debug_logs()
    print(f"\nDebug logs saved to: {analyzer.data_loader.debug_log_file}")

if __name__ == "__main__":
    test_peak_ppg_changes() 