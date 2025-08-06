"""
Test script for the scoring system with detailed breakdowns.
This script tests the comprehensive metrics and scoring system.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

def test_specific_players():
    """Test specific players for before/after weight change comparison."""
    print("Specific Players Scoring Test")
    print("="*50)
    print("Testing specific players for weight change comparison")
    print("="*50)
    
    # Define specific players to test (8 players total - 2 from each position)
    specific_players = [
        ("Patrick Mahomes (KC)", "QB"),
        ("Josh Allen (BUF)", "QB"),
        ("Christian McCaffrey (SF)", "RB"),
        ("Breece Hall (NYJ)", "RB"),
        ("Tyreek Hill (MIA)", "WR"),
        ("CeeDee Lamb (DAL)", "WR"),
        ("Travis Kelce (KC)", "TE"),
        ("Sam LaPorta (DET)", "TE")
    ]
    
    try:
        # Initialize the analyzer with testing mode enabled
        print("\n1. Initializing analyzer with testing mode...")
        analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
        print("✓ Analyzer initialized successfully")
        
        # Test data loading
        print("\n2. Testing data loading...")
        if analyzer.data_loader.years:
            print(f"✓ Data loaded for years: {analyzer.data_loader.years}")
        else:
            print("⚠ No years found in data")
            return
        
        # Test each specific player
        print("\n3. Testing specific players...")
        results = {}
        
        for player_name, position in specific_players:
            print(f"\n{'='*60}")
            print(f"TESTING: {player_name} ({position})")
            print(f"{'='*60}")
            
            # Find player in advanced data
            current_year = max(analyzer.data_loader.years)
            if position not in analyzer.data_loader.advanced_data or current_year not in analyzer.data_loader.advanced_data[position]:
                print(f"❌ No advanced data found for {position} in {current_year}")
                continue
                
            df = analyzer.data_loader.advanced_data[position][current_year]
            player_col = analyzer.data_loader.get_player_column(df, f"Advanced {position}")
            
            if not player_col:
                print(f"❌ No player column found in {position} data")
                continue
            
            # Find the specific player
            player_found = False
            player_row = None
            
            # Try exact match first
            for _, row in df.iterrows():
                if player_name in str(row.get(player_col, '')):
                    player_found = True
                    player_row = row
                    break
            
            # If not found, try with cleaned name
            if not player_found:
                cleaned_player_name = analyzer.data_loader.clean_player_name_for_matching(player_name)
                for _, row in df.iterrows():
                    row_player_name = str(row.get(player_col, ''))
                    cleaned_row_name = analyzer.data_loader.clean_player_name_for_matching(row_player_name)
                    if cleaned_player_name in cleaned_row_name or cleaned_row_name in cleaned_player_name:
                        player_found = True
                        player_row = row
                        break
            
            if not player_found:
                print(f"❌ Player {player_name} not found in {position} data")
                print(f"Available players in {position}:")
                for i, name in enumerate(df[player_col].head(5)):
                    print(f"  {i+1}. {name}")
                continue
            
            # Get advanced metrics
            key_metrics = analyzer.scoring_engine.data_loader.get_real_stat_value.__globals__['KEY_METRICS'].get(position, [])
            advanced_metrics = {}
            for metric in key_metrics:
                if metric in player_row.index and not pd.isna(player_row[metric]):
                    advanced_metrics[metric] = player_row[metric]
            
            # Get historical performance
            historical_perf = analyzer.player_analyzer.calculate_historical_performance(player_name, position)
            
            # Get injury profile
            injury_profile = analyzer.player_analyzer.get_injury_profile(player_name, position)
            
            # Get ML predictions
            features = analyzer.player_analyzer.ml_models._extract_enhanced_features_with_age_injury(player_row, position, player_name, current_year)
            ml_predictions = {}
            if features is not None:
                ml_predictions = analyzer.player_analyzer.ml_models.generate_predictions(player_name, position, features)
            
            # Calculate overall scores
            overall_scores = analyzer.scoring_engine.calculate_overall_profile_score(
                historical_perf, advanced_metrics, ml_predictions, injury_profile,
                debug_mode=True, position=position, player_row=player_row, player_name=player_name)
            
            # Store results
            results[player_name] = {
                'position': position,
                'historical_score': overall_scores.get('historical_score', 0),
                'advanced_score': overall_scores.get('advanced_score', 0),
                'ml_score': overall_scores.get('ml_score', 0),
                'injury_score': overall_scores.get('injury_score', 0),
                'final_score': overall_scores.get('final_score', 0),
                'advanced_metrics': advanced_metrics,
                'historical_perf': historical_perf,
                'ml_predictions': ml_predictions,
                'injury_profile': injury_profile
            }
            
            # Display detailed breakdown
            print(f"\nDETAILED BREAKDOWN FOR {player_name}:")
            print("-" * 50)
            print(f"Historical Score: {overall_scores.get('historical_score', 0):.2f}/10")
            print(f"Advanced Score: {overall_scores.get('advanced_score', 0):.2f}/10")
            print(f"ML Score: {overall_scores.get('ml_score', 0):.2f}/10")
            print(f"Injury Score: {overall_scores.get('injury_score', 0):.2f}/10")
            print(f"FINAL SCORE: {overall_scores.get('final_score', 0):.2f}/10")
            
            # Show key metrics
            if advanced_metrics:
                print(f"\nKey Advanced Metrics ({len(advanced_metrics)} available):")
                for metric, value in list(advanced_metrics.items())[:8]:  # Show top 8
                    if isinstance(value, (int, float)):
                        print(f"  • {metric}: {value:.2f}")
                    else:
                        print(f"  • {metric}: {value}")
        
        # Summary
        print(f"\n{'='*80}")
        print("SPECIFIC PLAYERS TEST SUMMARY")
        print(f"{'='*80}")
        for player_name, result in results.items():
            print(f"{player_name} ({result['position']}): {result['final_score']:.2f}/10")
        
        return results
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_enhanced_scoring():
    """Test the enhanced scoring system with detailed breakdowns."""
    print("Enhanced Scoring System Test")
    print("="*50)
    print("Goal: Identify score compression issues and expand ranges to 2-10")
    print("="*50)
    
    try:
        # Initialize the analyzer with testing mode enabled
        print("\n1. Initializing analyzer with testing mode...")
        analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
        print("✓ Analyzer initialized successfully")
        
        # Test data loading
        print("\n2. Testing data loading...")
        if analyzer.data_loader.years:
            print(f"✓ Data loaded for years: {analyzer.data_loader.years}")
        else:
            print("⚠ No years found in data")
            return
        
        # Run enhanced scoring test with 3 players for detailed analysis
        print("\n3. Running enhanced scoring test...")
        print("Testing 3 players with detailed breakdowns...")
        test_result = analyzer.run_scoring_test(num_players=3)
        
        print("\n" + "="*80)
        print("ENHANCED SCORING TEST COMPLETED")
        print("="*80)
        print("Review the detailed breakdowns above to identify:")
        print("• Score compression issues (scores stuck in 5-7 range)")
        print("• Individual metric contributions")
        print("• Component scoring ranges")
        print("• Final score distribution")
        
        return test_result
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "specific":
        test_specific_players()
    else:
        test_enhanced_scoring() 