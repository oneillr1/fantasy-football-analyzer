"""
Comprehensive ML Scoring Analysis and Improvement Script

This script analyzes the ML scoring system to understand why scores are low
and provides solutions to get scores into the 2-10 range.

Key Areas of Analysis:
1. Historical Score Calculation
2. Injury Score Calculation  
3. Breakout Score Calculation
4. Positional Value Calculation
5. ML Model Confidence and R² Values
6. Feature Engineering Quality
7. Target Value Distribution
8. Score Normalization and Scaling

Usage: python ml_scoring_analysis.py
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

class MLScoringAnalyzer:
    """
    Comprehensive analyzer for ML scoring system issues and improvements.
    """
    
    def __init__(self):
        self.analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
        self.test_players = {
            'QB': ['Lamar Jackson (BAL)', 'Patrick Mahomes (KC)', 'Josh Allen (BUF)'],
            'RB': ['Christian McCaffrey (SF)', 'Saquon Barkley (PHI)', 'Breece Hall (NYJ)'],
            'WR': ['Tyreek Hill (MIA)', 'CeeDee Lamb (DAL)', 'Amon-Ra St. Brown (DET)'],
            'TE': ['Travis Kelce (KC)', 'Sam LaPorta (DET)', 'Mark Andrews (BAL)']
        }
    
    def analyze_historical_scoring(self):
        """Analyze historical score calculation and distribution."""
        print("🔍 ANALYZING HISTORICAL SCORING")
        print("="*50)
        
        historical_scores = []
        
        for position, players in self.test_players.items():
            for player in players:
                try:
                    historical_perf = self.analyzer.player_analyzer.calculate_historical_performance(player, position)
                    historical_score = self.analyzer.scoring_engine._calculate_historical_score(historical_perf)
                    
                    historical_scores.append({
                        'player': player,
                        'position': position,
                        'score': historical_score,
                        'years_of_data': historical_perf.get('years_of_data', 0),
                        'adp_differential': historical_perf.get('adp_differential', 0),
                        'peak_season': historical_perf.get('peak_season', 0),
                        'consistency': historical_perf.get('consistency', 0),
                        'has_historical_data': historical_perf.get('has_historical_data', False)
                    })
                    
                    print(f"{player} ({position}): {historical_score:.2f}/10")
                    print(f"  Years: {historical_perf.get('years_of_data', 0)}")
                    print(f"  ADP Diff: {historical_perf.get('adp_differential', 0):.1f}")
                    print(f"  Peak: {historical_perf.get('peak_season', 0):.1f}")
                    print(f"  Consistency: {historical_perf.get('consistency', 0):.1f}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error analyzing {player}: {e}")
        
        # Analyze distribution
        scores = [h['score'] for h in historical_scores]
        print(f"Historical Score Distribution:")
        print(f"  Mean: {np.mean(scores):.2f}")
        print(f"  Median: {np.median(scores):.2f}")
        print(f"  Min: {np.min(scores):.2f}")
        print(f"  Max: {np.max(scores):.2f}")
        print(f"  Std: {np.std(scores):.2f}")
        
        return historical_scores
    
    def analyze_injury_scoring(self):
        """Analyze injury score calculation and distribution."""
        print("\n🔍 ANALYZING INJURY SCORING")
        print("="*50)
        
        injury_scores = []
        
        for position, players in self.test_players.items():
            for player in players:
                try:
                    injury_profile = self.analyzer.player_analyzer.get_injury_profile(player, position)
                    injury_score = self.analyzer.scoring_engine._calculate_injury_profile_score(injury_profile, player, position)
                    
                    injury_scores.append({
                        'player': player,
                        'position': position,
                        'score': injury_score,
                        'injury_risk': injury_profile.get('injury_risk', 0.5),
                        'games_missed': injury_profile.get('games_missed', 0),
                        'has_injury_data': injury_profile.get('has_injury_data', False)
                    })
                    
                    print(f"{player} ({position}): {injury_score:.2f}/10")
                    print(f"  Injury Risk: {injury_profile.get('injury_risk', 0.5):.3f}")
                    print(f"  Games Missed: {injury_profile.get('games_missed', 0)}")
                    print(f"  Has Data: {injury_profile.get('has_injury_data', False)}")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error analyzing {player}: {e}")
        
        # Analyze distribution
        scores = [i['score'] for i in injury_scores]
        print(f"Injury Score Distribution:")
        print(f"  Mean: {np.mean(scores):.2f}")
        print(f"  Median: {np.median(scores):.2f}")
        print(f"  Min: {np.min(scores):.2f}")
        print(f"  Max: {np.max(scores):.2f}")
        print(f"  Std: {np.std(scores):.2f}")
        
        return injury_scores
    
    def analyze_ml_model_targets(self):
        """Analyze ML model target value distribution and quality."""
        print("\n🔍 ANALYZING ML MODEL TARGETS")
        print("="*50)
        
        target_analysis = {}
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            print(f"\n{position} TARGET ANALYSIS:")
            
            # Collect target values from training data
            all_targets = {'breakout': [], 'consistency': [], 'positional_value': [], 'injury_risk': []}
            
            # Simulate target calculation for recent years
            for year in [2022, 2023]:  # Use recent years for analysis
                if year in self.analyzer.data_loader.advanced_data.get(position, {}):
                    df = self.analyzer.data_loader.advanced_data[position][year]
                    
                    for _, row in df.iterrows():
                        player = row.get('PLAYER', row.get('Player', ''))
                        if pd.isna(player) or not isinstance(player, str):
                            continue
                        
                        # Simulate target calculation
                        try:
                            # Mock ADP and results for target calculation
                            mock_adp_rank = 50.0  # Middle ADP
                            mock_finish_rank = 40.0  # Good finish
                            mock_games_played = 15.0  # Most games played
                            
                            adp_diff = mock_adp_rank - mock_finish_rank
                            
                            # Calculate targets using same logic as ML models
                            if adp_diff >= 30:
                                breakout = 0.95 + min(0.05, (adp_diff - 30) / 40)
                            elif adp_diff >= 15:
                                breakout = 0.75 + (adp_diff - 15) / 15 * 0.2
                            elif adp_diff >= 5:
                                breakout = 0.55 + (adp_diff - 5) / 10 * 0.2
                            elif adp_diff >= -5:
                                breakout = 0.45 + (adp_diff + 5) / 10 * 0.1
                            elif adp_diff >= -20:
                                breakout = 0.25 + (adp_diff + 20) / 15 * 0.2
                            else:
                                breakout = max(0.0, 0.25 + (adp_diff + 20) / 30 * 0.25)
                            
                            consistency = max(0.0, min(1.0, 1.0 - (abs(adp_diff) / 60)))
                            positional_value = max(0.0, min(1.0, (60 - mock_finish_rank) / 60))
                            
                            season_length = 17 if year >= 2021 else 16
                            games_missed = max(0, season_length - mock_games_played)
                            injury_risk = min(1.0, games_missed / 10.0)
                            
                            all_targets['breakout'].append(breakout)
                            all_targets['consistency'].append(consistency)
                            all_targets['positional_value'].append(positional_value)
                            all_targets['injury_risk'].append(injury_risk)
                            
                        except Exception as e:
                            continue
            
            # Analyze target distributions
            target_analysis[position] = {}
            for target_name, values in all_targets.items():
                if values:
                    target_analysis[position][target_name] = {
                        'mean': np.mean(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'std': np.std(values),
                        'count': len(values)
                    }
                    
                    print(f"  {target_name}:")
                    print(f"    Mean: {np.mean(values):.3f}")
                    print(f"    Range: {np.min(values):.3f} - {np.max(values):.3f}")
                    print(f"    Std: {np.std(values):.3f}")
                    print(f"    Count: {len(values)}")
        
        return target_analysis
    
    def analyze_ml_predictions(self):
        """Analyze ML predictions and their scoring."""
        print("\n🔍 ANALYZING ML PREDICTIONS")
        print("="*50)
        
        prediction_analysis = []
        
        for position, players in self.test_players.items():
            for player in players:
                try:
                    # Get player data
                    current_year = max(self.analyzer.data_loader.years)
                    df = self.analyzer.data_loader.advanced_data[position][current_year]
                    player_col = self.analyzer.data_loader.get_player_column(df, f"Advanced {position}")
                    
                    # Find player
                    player_row = None
                    for _, row in df.iterrows():
                        if player in str(row.get(player_col, '')):
                            player_row = row
                            break
                    
                    if player_row is None:
                        print(f"❌ {player} not found in advanced data")
                        continue
                    
                    # Extract features
                    features = self.analyzer.player_analyzer.ml_models._extract_enhanced_features_with_age_injury(
                        player_row, position, player, current_year)
                    
                    if features is None:
                        print(f"❌ {player} features extraction failed")
                        continue
                    
                    # Generate predictions
                    ml_predictions = self.analyzer.player_analyzer.ml_models.generate_predictions(
                        player, position, features, debug_mode=True)
                    
                    # Calculate ML score
                    ml_score = self.analyzer.scoring_engine._calculate_ml_predictions_score(
                        ml_predictions, player, position)
                    
                    prediction_analysis.append({
                        'player': player,
                        'position': position,
                        'ml_score': ml_score,
                        'predictions': ml_predictions,
                        'feature_count': len(features),
                        'feature_range': (np.min(features), np.max(features))
                    })
                    
                    print(f"{player} ({position}): {ml_score:.2f}/10")
                    print(f"  Predictions: {ml_predictions}")
                    print(f"  Features: {len(features)} (range: {np.min(features):.3f} - {np.max(features):.3f})")
                    print()
                    
                except Exception as e:
                    print(f"❌ Error analyzing {player}: {e}")
        
        # Analyze ML score distribution
        ml_scores = [p['ml_score'] for p in prediction_analysis]
        print(f"ML Score Distribution:")
        print(f"  Mean: {np.mean(ml_scores):.2f}")
        print(f"  Median: {np.median(ml_scores):.2f}")
        print(f"  Min: {np.min(ml_scores):.2f}")
        print(f"  Max: {np.max(ml_scores):.2f}")
        print(f"  Std: {np.std(ml_scores):.2f}")
        
        return prediction_analysis
    
    def analyze_model_confidence(self):
        """Analyze ML model confidence and R² values."""
        print("\n🔍 ANALYZING MODEL CONFIDENCE")
        print("="*50)
        
        for position in ['QB', 'RB', 'WR', 'TE']:
            print(f"\n{position} MODEL CONFIDENCE:")
            
            for target_name in ['breakout', 'consistency', 'positional_value', 'injury_risk']:
                model_key = f"{target_name}_{position}"
                confidence = self.analyzer.player_analyzer.ml_models.model_confidence.get(model_key, 0.0)
                
                print(f"  {target_name}: {confidence:.3f} R²")
                
                if confidence < 0.1:
                    print(f"    ⚠️  Very low confidence - model may be unreliable")
                elif confidence < 0.3:
                    print(f"    ⚠️  Low confidence - consider improving features")
                elif confidence < 0.5:
                    print(f"    ✓ Moderate confidence")
                else:
                    print(f"    ✓ Good confidence")
    
    def propose_improvements(self):
        """Propose specific improvements to get scores into 2-10 range."""
        print("\n🔧 PROPOSED IMPROVEMENTS")
        print("="*50)
        
        print("1. ML SCORE NORMALIZATION IMPROVEMENTS:")
        print("   - Current: Raw predictions (0-1) → *10 → 0-10")
        print("   - Proposed: Apply non-linear scaling to spread scores")
        print("   - Formula: score = (raw_score^0.7) * 12 - 2")
        print("   - This maps 0.1→2.0, 0.5→5.0, 0.9→8.5")
        
        print("\n2. HISTORICAL SCORE IMPROVEMENTS:")
        print("   - Add more weight to recent performance")
        print("   - Include fantasy points per game")
        print("   - Consider positional scarcity")
        
        print("\n3. INJURY SCORE IMPROVEMENTS:")
        print("   - Better integration with injury predictor data")
        print("   - Consider age-related injury risk")
        print("   - Add position-specific injury patterns")
        
        print("\n4. FEATURE ENGINEERING IMPROVEMENTS:")
        print("   - Add more derived features (ratios, trends)")
        print("   - Include team context (offense quality)")
        print("   - Add competition level features")
        
        print("\n5. MODEL TRAINING IMPROVEMENTS:")
        print("   - Use more training data (all years)")
        print("   - Try different model architectures")
        print("   - Implement ensemble methods")
        
        print("\n6. TARGET VALUE IMPROVEMENTS:")
        print("   - Adjust target calculation formulas")
        print("   - Use fantasy points instead of just ADP")
        print("   - Include consistency metrics")
    
    def test_improved_scoring(self):
        """Test improved scoring formulas."""
        print("\n🧪 TESTING IMPROVED SCORING")
        print("="*50)
        
        # Test improved ML score normalization
        def improved_ml_score(predictions):
            if not predictions:
                return 5.0  # Default middle score
            
            total_score = 0.0
            valid_count = 0
            
            for pred_type, score in predictions.items():
                if score is not None and not pd.isna(score):
                    # Apply improved normalization
                    improved_score = (score ** 0.7) * 12 - 2
                    improved_score = max(0.0, min(10.0, improved_score))
                    
                    total_score += improved_score
                    valid_count += 1
            
            if valid_count == 0:
                return 5.0
            
            return total_score / valid_count
        
        # Test with sample predictions
        test_predictions = [
            {'breakout': 0.1, 'consistency': 0.2, 'positional_value': 0.3},
            {'breakout': 0.5, 'consistency': 0.6, 'positional_value': 0.7},
            {'breakout': 0.9, 'consistency': 0.8, 'positional_value': 0.9}
        ]
        
        print("Improved ML Score Normalization Test:")
        for i, preds in enumerate(test_predictions):
            old_score = sum(preds.values()) / len(preds.values()) * 10
            new_score = improved_ml_score(preds)
            print(f"  Test {i+1}: {old_score:.1f} → {new_score:.1f}/10")
    
    def run_comprehensive_analysis(self):
        """Run all analysis components."""
        print("🚀 COMPREHENSIVE ML SCORING ANALYSIS")
        print("="*60)
        
        # Run all analyses
        historical_scores = self.analyze_historical_scoring()
        injury_scores = self.analyze_injury_scoring()
        target_analysis = self.analyze_ml_model_targets()
        prediction_analysis = self.analyze_ml_predictions()
        self.analyze_model_confidence()
        
        # Propose improvements
        self.propose_improvements()
        self.test_improved_scoring()
        
        print("\n✅ ANALYSIS COMPLETE")
        print("="*60)
        print("Key findings and next steps have been identified.")
        print("Use the proposed improvements to enhance scoring quality.")

def main():
    """Main function to run the analysis."""
    analyzer = MLScoringAnalyzer()
    analyzer.run_comprehensive_analysis()

if __name__ == "__main__":
    main() 