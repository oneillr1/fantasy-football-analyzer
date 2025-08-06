"""
Test Script for ML Scoring Improvements

This script tests the improved ML scoring system to verify that scores
are now in the 2-10 range with better distribution.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

def test_improved_ml_scoring():
    """Test the improved ML scoring system."""
    print("🧪 TESTING IMPROVED ML SCORING")
    print("="*50)
    
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
    
    # Test players
    test_players = {
        'QB': ['Lamar Jackson (BAL)', 'Josh Allen (BUF)'],
        'RB': ['Christian McCaffrey (SF)', 'Saquon Barkley (PHI)'],
        'WR': ['Tyreek Hill (MIA)', 'CeeDee Lamb (DAL)'],
        'TE': ['Travis Kelce (KC)', 'Sam LaPorta (DET)']
    }
    
    all_scores = []
    
    for position, players in test_players.items():
        print(f"\n{position} PLAYERS:")
        for player in players:
            try:
                # Get player data
                current_year = max(analyzer.data_loader.years)
                df = analyzer.data_loader.advanced_data[position][current_year]
                player_col = analyzer.data_loader.get_player_column(df, f"Advanced {position}")
                
                # Find player
                player_row = None
                for _, row in df.iterrows():
                    if player in str(row.get(player_col, '')):
                        player_row = row
                        break
                
                if player_row is None:
                    print(f"❌ {player} not found")
                    continue
                
                # Extract features
                features = analyzer.player_analyzer.ml_models._extract_enhanced_features_with_age_injury(
                    player_row, position, player, current_year)
                
                if features is None:
                    print(f"❌ {player} features failed")
                    continue
                
                # Generate predictions
                ml_predictions = analyzer.player_analyzer.ml_models.generate_predictions(
                    player, position, features, debug_mode=True)
                
                # Calculate ML score with improved normalization
                ml_score = analyzer.scoring_engine._calculate_ml_predictions_score(
                    ml_predictions, player, position)
                
                # Get other component scores
                historical_perf = analyzer.player_analyzer.calculate_historical_performance(player, position)
                historical_score = analyzer.scoring_engine._calculate_enhanced_historical_score(historical_perf, position)
                
                injury_profile = analyzer.player_analyzer.get_injury_profile(player, position)
                injury_score = analyzer.scoring_engine._calculate_injury_profile_score(injury_profile, player, position)
                
                # Calculate overall score
                overall_score = (
                    historical_score * 0.10 +
                    ml_score * 0.30 +
                    injury_score * 0.25
                )  # Simplified for testing
                
                all_scores.append({
                    'player': player,
                    'position': position,
                    'ml_score': ml_score,
                    'historical_score': historical_score,
                    'injury_score': injury_score,
                    'overall_score': overall_score,
                    'predictions': ml_predictions
                })
                
                print(f"{player}:")
                print(f"  ML Score: {ml_score:.2f}/10")
                print(f"  Historical: {historical_score:.2f}/10")
                print(f"  Injury: {injury_score:.2f}/10")
                print(f"  Overall: {overall_score:.2f}/10")
                print(f"  Raw Predictions: {ml_predictions}")
                print()
                
            except Exception as e:
                print(f"❌ Error testing {player}: {e}")
    
    # Analyze score distribution
    ml_scores = [s['ml_score'] for s in all_scores]
    historical_scores = [s['historical_score'] for s in all_scores]
    injury_scores = [s['injury_score'] for s in all_scores]
    overall_scores = [s['overall_score'] for s in all_scores]
    
    print("📊 SCORE DISTRIBUTION ANALYSIS:")
    print("="*50)
    
    print(f"ML Scores:")
    print(f"  Mean: {np.mean(ml_scores):.2f}")
    print(f"  Median: {np.median(ml_scores):.2f}")
    print(f"  Min: {np.min(ml_scores):.2f}")
    print(f"  Max: {np.max(ml_scores):.2f}")
    print(f"  Std: {np.std(ml_scores):.2f}")
    
    print(f"\nHistorical Scores:")
    print(f"  Mean: {np.mean(historical_scores):.2f}")
    print(f"  Median: {np.median(historical_scores):.2f}")
    print(f"  Min: {np.min(historical_scores):.2f}")
    print(f"  Max: {np.max(historical_scores):.2f}")
    print(f"  Std: {np.std(historical_scores):.2f}")
    
    print(f"\nInjury Scores:")
    print(f"  Mean: {np.mean(injury_scores):.2f}")
    print(f"  Median: {np.median(injury_scores):.2f}")
    print(f"  Min: {np.min(injury_scores):.2f}")
    print(f"  Max: {np.max(injury_scores):.2f}")
    print(f"  Std: {np.std(injury_scores):.2f}")
    
    print(f"\nOverall Scores:")
    print(f"  Mean: {np.mean(overall_scores):.2f}")
    print(f"  Median: {np.median(overall_scores):.2f}")
    print(f"  Min: {np.min(overall_scores):.2f}")
    print(f"  Max: {np.max(overall_scores):.2f}")
    print(f"  Std: {np.std(overall_scores):.2f}")
    
    # Check if improvements worked
    print(f"\n✅ IMPROVEMENT VERIFICATION:")
    print("="*50)
    
    if np.min(ml_scores) >= 2.0:
        print("✓ ML scores now start at 2.0+ (was 2.4 before)")
    else:
        print("❌ ML scores still too low")
    
    if np.max(ml_scores) >= 6.0:
        print("✓ ML scores now reach 6.0+ (was 4.05 before)")
    else:
        print("❌ ML scores still too compressed")
    
    if np.std(ml_scores) >= 1.0:
        print("✓ ML scores now have good variance")
    else:
        print("❌ ML scores still too compressed")
    
    if np.std(injury_scores) >= 1.0:
        print("✓ Injury scores now have variance (was 0.40 before)")
    else:
        print("❌ Injury scores still too uniform")
    
    return all_scores

def test_normalization_formula():
    """Test the improved normalization formula."""
    print("\n🧮 TESTING NORMALIZATION FORMULA")
    print("="*50)
    
    # Test the old vs new normalization
    test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("Raw → Old → New")
    for raw in test_values:
        old_score = raw * 10  # Old formula
        new_score = (raw ** 0.5) * 15 - 1  # New formula
        new_score = max(0.0, min(10.0, new_score))
        
        print(f"{raw:.1f} → {old_score:.1f} → {new_score:.1f}/10")

if __name__ == "__main__":
    test_normalization_formula()
    test_improved_ml_scoring() 