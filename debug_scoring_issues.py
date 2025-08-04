"""
Comprehensive Debug Script for Scoring System Issues

This script investigates three critical issues:
1. ML scores are extremely low (highest 2.2/10)
2. Injury scores are identical (all 7.50/10) - likely defaulting
3. Advanced metrics are low for good players

Usage: python debug_scoring_issues.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

def debug_ml_scores():
    """Debug why ML scores are so low."""
    print("🔍 DEBUGGING ML SCORES")
    print("="*50)
    
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
    
    # Test with a specific player
    test_player = "Lamar Jackson (BAL)"
    position = "QB"
    
    print(f"Testing ML scores for: {test_player}")
    
    # Get player data
    current_year = max(analyzer.data_loader.years)
    df = analyzer.data_loader.advanced_data[position][current_year]
    player_col = analyzer.data_loader.get_player_column(df, f"Advanced {position}")
    
    # Find player
    player_row = None
    for _, row in df.iterrows():
        if test_player in str(row.get(player_col, '')):
            player_row = row
            break
    
    if player_row is None:
        print("❌ Player not found")
        return
    
    print(f"✓ Player found in advanced data")
    
    print(f"\n1. EXTRACTING FEATURES:")
    features = analyzer.player_analyzer.ml_models._extract_enhanced_features_with_age_injury(player_row, position, test_player, current_year)
    print(f"Features extracted: {features is not None}")
    if features is not None:
        print(f"Number of features: {len(features)}")
        print(f"Feature values: {features}")
        print(f"Feature range: {np.min(features):.3f} to {np.max(features):.3f}")
    
    print(f"\n2. GENERATING PREDICTIONS:")
    ml_predictions = analyzer.player_analyzer.ml_models.generate_predictions(test_player, position, features)
    print(f"ML Predictions: {ml_predictions}")
    
    print(f"\n3. CHECKING MODEL TRAINING:")
    # Check if models are trained
    for target_name in ['breakout', 'consistency', 'positional_value']:
        if target_name in analyzer.player_analyzer.ml_models.ml_models:
            if position in analyzer.player_analyzer.ml_models.ml_models[target_name]:
                model = analyzer.player_analyzer.ml_models.ml_models[target_name][position]
                print(f"✓ {target_name} model exists for {position}")
            else:
                print(f"❌ {target_name} model missing for {position}")
        else:
            print(f"❌ {target_name} model type missing")
    
    print(f"\n4. CHECKING FEATURE SCALING:")
    if hasattr(analyzer.player_analyzer.ml_models, 'scalers'):
        for target_name in ['breakout', 'consistency', 'positional_value']:
            scaler_key = f"{target_name}_{position}"
            if scaler_key in analyzer.player_analyzer.ml_models.scalers:
                print(f"✓ {target_name} scaler exists")
            else:
                print(f"❌ {target_name} scaler missing")
    
    print(f"\n5. TESTING ML SCORE CALCULATION:")
    # Test the ML score calculation directly
    from modules.scoring_engine import ScoringEngine
    scoring_engine = ScoringEngine(analyzer.data_loader)
    ml_score = scoring_engine._calculate_ml_predictions_score(ml_predictions, test_player, position)
    print(f"Calculated ML Score: {ml_score:.2f}/10")
    
    # Show raw vs normalized scores
    for pred_type, score in ml_predictions.items():
        raw_score = score
        normalized_score = max(0.0, min(10.0, score * 10.0))
        print(f"  {pred_type}: {raw_score:.3f} → {normalized_score:.2f}/10")

def debug_injury_scores():
    """Debug why injury scores are identical."""
    print("\n🔍 DEBUGGING INJURY SCORES")
    print("="*50)
    
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
    
    test_players = [
        ("Lamar Jackson (BAL)", "QB"),
        ("Saquon Barkley (PHI)", "RB"),
        ("Ja'Marr Chase (CIN)", "WR"),
        ("George Kittle (SF)", "TE")
    ]
    
    for player_name, position in test_players:
        print(f"\nTesting injury profile for: {player_name}")
        
        # Get injury profile
        injury_profile = analyzer.player_analyzer.get_injury_profile(player_name, position)
        print(f"Injury Profile: {injury_profile}")
        
        # Check if it's using default values
        if injury_profile.get('injury_risk', 0) == 0.5:
            print("⚠️  WARNING: Using default injury risk of 0.5")
        
        # Check injury data availability
        if position in analyzer.data_loader.injury_data:
            current_year = max(analyzer.data_loader.years)
            if current_year in analyzer.data_loader.injury_data[position]:
                injury_df = analyzer.data_loader.injury_data[position][current_year]
                print(f"Injury data available: {len(injury_df)} rows")
                
                # Look for the player
                player_col = analyzer.data_loader.get_player_column(injury_df, f"Injury {position}")
                if player_col:
                    player_found = False
                    for _, row in injury_df.iterrows():
                        if player_name in str(row.get(player_col, '')):
                            print(f"✓ Player found in injury data")
                            print(f"Injury data row: {dict(row)}")
                            player_found = True
                            break
                    if not player_found:
                        print("❌ Player not found in injury data")
                        # Show available players
                        print("Available players in injury data:")
                        for i, name in enumerate(injury_df[player_col].head(5)):
                            print(f"  {i+1}. {name}")
                else:
                    print("❌ No player column found in injury data")
            else:
                print(f"❌ No injury data for {position} in {current_year}")
        else:
            print(f"❌ No injury data for {position}")
        
        # Test injury score calculation
        from modules.scoring_engine import ScoringEngine
        scoring_engine = ScoringEngine(analyzer.data_loader)
        injury_score = scoring_engine._calculate_injury_profile_score(injury_profile, player_name, position)
        print(f"Calculated Injury Score: {injury_score:.2f}/10")
        
        # Show breakdown
        injury_risk = injury_profile.get('injury_risk', 0.5)
        games_missed = injury_profile.get('games_missed', 0)
        injury_history = injury_profile.get('injury_history', [])
        
        risk_score = max(0.0, min(10.0, (1.0 - injury_risk) * 10.0))
        games_penalty = min(5.0, games_missed * 0.5)
        games_score = max(0.0, 10.0 - games_penalty)
        history_penalty = min(3.0, len(injury_history) * 0.5)
        history_score = max(0.0, 10.0 - history_penalty)
        
        print(f"  Risk Score: {risk_score:.2f} (risk: {injury_risk:.2f})")
        print(f"  Games Score: {games_score:.2f} (missed: {games_missed})")
        print(f"  History Score: {history_score:.2f} (history: {len(injury_history)} items)")
        print(f"  Weighted Average: {risk_score*0.5 + games_score*0.3 + history_score*0.2:.2f}")

def debug_advanced_metrics():
    """Debug why advanced metrics are low for good players."""
    print("\n🔍 DEBUGGING ADVANCED METRICS")
    print("="*50)
    
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
    
    test_players = [
        ("Lamar Jackson (BAL)", "QB"),
        ("Saquon Barkley (PHI)", "RB"),
        ("Ja'Marr Chase (CIN)", "WR"),
        ("George Kittle (SF)", "TE")
    ]
    
    for player_name, position in test_players:
        print(f"\nTesting advanced metrics for: {player_name}")
        
        # Get player data
        current_year = max(analyzer.data_loader.years)
        df = analyzer.data_loader.advanced_data[position][current_year]
        player_col = analyzer.data_loader.get_player_column(df, f"Advanced {position}")
        
        # Find player
        player_row = None
        for _, row in df.iterrows():
            if player_name in str(row.get(player_col, '')):
                player_row = row
                break
        
        if player_row is None:
            print("❌ Player not found in advanced data")
            continue
        
        print(f"✓ Player found in advanced data")
        
        # Get key metrics for position
        from config.constants import KEY_METRICS
        key_metrics = KEY_METRICS.get(position, [])
        print(f"Key metrics for {position}: {key_metrics}")
        
        # Check each metric
        print(f"\nMetric values for {player_name}:")
        for metric in key_metrics:
            if metric in player_row.index:
                value = player_row[metric]
                if not pd.isna(value):
                    print(f"  {metric}: {value}")
                else:
                    print(f"  {metric}: NaN")
            else:
                print(f"  {metric}: Not in data")
        
        # Test normalization
        print(f"\nTesting metric normalization:")
        scoring_engine = analyzer.scoring_engine
        for metric in key_metrics[:5]:  # Test first 5 metrics
            if metric in player_row.index and not pd.isna(player_row[metric]):
                raw_value = player_row[metric]
                
                # Convert string values to float
                if isinstance(raw_value, str):
                    # Remove % and convert to float
                    raw_value = raw_value.replace('%', '').replace(',', '')
                    try:
                        raw_value = float(raw_value)
                    except ValueError:
                        print(f"  {metric}: Cannot convert '{raw_value}' to float")
                        continue
                
                # Determine category
                category = None
                for cat, metrics in KEY_METRICS.items():
                    if metric in metrics:
                        category = cat
                        break
                
                if category:
                    try:
                        normalized = scoring_engine.normalize_metric_universal(raw_value, metric, category)
                        print(f"  {metric} ({category}): {raw_value} → {normalized:.2f}")
                    except Exception as e:
                        print(f"  {metric}: Error normalizing - {e}")
        
        # Test advanced score calculation
        advanced_metrics = {}
        for metric in key_metrics:
            if metric in player_row.index and not pd.isna(player_row[metric]):
                advanced_metrics[metric] = player_row[metric]
        
        if advanced_metrics:
            advanced_score_result = scoring_engine.calculate_advanced_score_universal(player_row, position)
            if isinstance(advanced_score_result, tuple):
                advanced_score = advanced_score_result[0]  # Get the score from tuple
            else:
                advanced_score = advanced_score_result
            print(f"\nAdvanced Score: {advanced_score:.2f}/10")
            print(f"Number of metrics used: {len(advanced_metrics)}")

def debug_scoring_weights():
    """Debug the scoring weights and their impact."""
    print("\n🔍 DEBUGGING SCORING WEIGHTS")
    print("="*50)
    
    from config.constants import SCORING_WEIGHTS, PROFILE_WEIGHTS
    
    print("Current Scoring Weights:")
    for category, weight in SCORING_WEIGHTS.items():
        print(f"  {category}: {weight}")
    
    print("\nCurrent Profile Weights:")
    for category, weight in PROFILE_WEIGHTS.items():
        print(f"  {category}: {weight}")

def debug_comprehensive_test():
    """Run a comprehensive test with detailed breakdowns."""
    print("\n🔍 COMPREHENSIVE SCORING DEBUG")
    print("="*80)
    
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
    
    test_players = [
        ("Lamar Jackson (BAL)", "QB"),
        ("Saquon Barkley (PHI)", "RB"),
        ("Ja'Marr Chase (CIN)", "WR"),
        ("George Kittle (SF)", "TE")
    ]
    
    for player_name, position in test_players:
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE DEBUG: {player_name} ({position})")
        print(f"{'='*80}")
        
        # Get player data
        current_year = max(analyzer.data_loader.years)
        df = analyzer.data_loader.advanced_data[position][current_year]
        player_col = analyzer.data_loader.get_player_column(df, f"Advanced {position}")
        
        # Find player
        player_row = None
        for _, row in df.iterrows():
            if player_name in str(row.get(player_col, '')):
                player_row = row
                break
        
        if player_row is None:
            print(f"❌ Player {player_name} not found")
            continue
        
        # Get all components
        from config.constants import KEY_METRICS
        key_metrics = KEY_METRICS.get(position, [])
        advanced_metrics = {}
        for metric in key_metrics:
            if metric in player_row.index and not pd.isna(player_row[metric]):
                advanced_metrics[metric] = player_row[metric]
        
        historical_perf = analyzer.player_analyzer.calculate_historical_performance(player_name, position)
        injury_profile = analyzer.player_analyzer.get_injury_profile(player_name, position)
        
        features = analyzer.player_analyzer.ml_models._extract_enhanced_features_with_age_injury(player_row, position, player_name, current_year)
        ml_predictions = {}
        if features is not None:
            ml_predictions = analyzer.player_analyzer.ml_models.generate_predictions(player_name, position, features)
        
        # Calculate scores
        overall_scores = analyzer.scoring_engine.calculate_overall_profile_score(
            historical_perf, advanced_metrics, ml_predictions, injury_profile,
            debug_mode=True, position=position, player_row=player_row, player_name=player_name)
        
        # Display comprehensive breakdown
        print(f"\n📊 COMPREHENSIVE SCORING BREAKDOWN:")
        print(f"Historical Score: {overall_scores.get('historical_score', 0):.2f}/10")
        print(f"Advanced Score: {overall_scores.get('advanced_score', 0):.2f}/10")
        print(f"ML Score: {overall_scores.get('ml_score', 0):.2f}/10")
        print(f"Injury Score: {overall_scores.get('injury_score', 0):.2f}/10")
        print(f"FINAL SCORE: {overall_scores.get('final_score', 0):.2f}/10")
        
        # Show raw ML predictions
        print(f"\n🤖 ML PREDICTIONS (Raw):")
        for pred_type, score in ml_predictions.items():
            print(f"  {pred_type}: {score:.3f}")
        
        # Show injury profile details
        print(f"\n🏥 INJURY PROFILE:")
        print(f"  Risk: {injury_profile.get('injury_risk', 0):.2f}")
        print(f"  Games Missed: {injury_profile.get('games_missed', 0)}")
        print(f"  Has Data: {injury_profile.get('has_injury_data', False)}")
        
        # Show key advanced metrics
        print(f"\n📈 KEY ADVANCED METRICS:")
        for metric, value in list(advanced_metrics.items())[:8]:
            print(f"  {metric}: {value}")

def main():
    """Run all debug functions."""
    print("🔍 COMPREHENSIVE SCORING DEBUG")
    print("="*80)
    
    debug_ml_scores()
    debug_injury_scores()
    debug_advanced_metrics()
    debug_scoring_weights()
    debug_comprehensive_test()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("Review the output above to identify:")
    print("1. ML model training and prediction issues")
    print("2. Injury data loading and default value issues")
    print("3. Advanced metrics normalization issues")

if __name__ == "__main__":
    main() 