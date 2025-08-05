"""
Test script to verify injury data parsing fixes and age factor implementation.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

def test_injury_fixes():
    """Test the injury data parsing fixes and age factor."""
    print("🧪 TESTING INJURY FIXES AND AGE FACTOR")
    print("="*50)
    
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
    
    # Test players with known injury histories
    test_players = {
        'RB': ['Christian McCaffrey (SF)', 'Saquon Barkley (PHI)', 'Bijan Robinson (ATL)'],
        'WR': ['Tyreek Hill (MIA)', 'CeeDee Lamb (DAL)'],
        'TE': ['Travis Kelce (KC)', 'Sam LaPorta (DET)']
    }
    
    all_results = []
    
    for position, players in test_players.items():
        print(f"\n{position} PLAYERS:")
        for player in players:
            try:
                # Get injury profile with fixes
                injury_profile = analyzer.player_analyzer.get_injury_profile(player, position)
                
                # Calculate injury score with age factor
                injury_score = analyzer.scoring_engine._calculate_injury_profile_score(
                    injury_profile, player, position)
                
                # Get player age
                age = injury_profile.get('age', 'Unknown')
                
                # Display results
                print(f"\n{player}:")
                print(f"  Age: {age}")
                print(f"  Injury Risk: {injury_profile.get('injury_risk', 'N/A')}")
                print(f"  Games Missed: {injury_profile.get('games_missed', 'N/A')}")
                print(f"  Has Injury Data: {injury_profile.get('has_injury_data', False)}")
                print(f"  Injury Score: {injury_score:.2f}/10")
                
                all_results.append({
                    'player': player,
                    'position': position,
                    'age': age,
                    'injury_risk': injury_profile.get('injury_risk'),
                    'games_missed': injury_profile.get('games_missed'),
                    'injury_score': injury_score,
                    'has_injury_data': injury_profile.get('has_injury_data')
                })
                
            except Exception as e:
                print(f"❌ Error testing {player}: {e}")
    
    # Analyze results
    print(f"\n📊 ANALYSIS:")
    print("="*50)
    
    injury_scores = [r['injury_score'] for r in all_results]
    ages = [r['age'] for r in all_results if r['age'] != 'Unknown']
    
    print(f"Injury Scores:")
    print(f"  Mean: {np.mean(injury_scores):.2f}")
    print(f"  Median: {np.median(injury_scores):.2f}")
    print(f"  Min: {np.min(injury_scores):.2f}")
    print(f"  Max: {np.max(injury_scores):.2f}")
    print(f"  Std: {np.std(injury_scores):.2f}")
    
    if ages:
        print(f"\nAges:")
        print(f"  Mean: {np.mean(ages):.1f}")
        print(f"  Min: {np.min(ages):.1f}")
        print(f"  Max: {np.max(ages):.1f}")
    
    # Check specific cases
    print(f"\n✅ VERIFICATION:")
    print("="*50)
    
    # Check Christian McCaffrey specifically
    mccaffrey_results = [r for r in all_results if 'McCaffrey' in r['player']]
    if mccaffrey_results:
        mccaffrey = mccaffrey_results[0]
        print(f"Christian McCaffrey:")
        print(f"  Age: {mccaffrey['age']}")
        print(f"  Injury Risk: {mccaffrey['injury_risk']}")
        print(f"  Games Missed: {mccaffrey['games_missed']}")
        print(f"  Injury Score: {mccaffrey['injury_score']:.2f}/10")
        
        # Should now be much lower due to high injury risk
        if mccaffrey['injury_score'] < 6.0:
            print("  ✓ Injury score now correctly reflects high risk")
        else:
            print("  ❌ Injury score still too high for high risk player")
    
    return all_results

if __name__ == "__main__":
    test_injury_fixes() 