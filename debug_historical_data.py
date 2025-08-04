"""
Debug script to investigate historical data matching issues
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer

def debug_historical_data():
    """Debug historical data matching issues"""
    print("Historical Data Debug")
    print("="*50)

    try:
        # Initialize analyzer
        analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
        print("✓ Analyzer initialized")

        # Test specific players that we know should have historical data
        test_players = [
            ("Tee Higgins (CIN)", "WR"),
            ("Justin Jefferson (MIN)", "WR"),
            ("Jameson Williams (DET)", "WR")
        ]

        current_year = max(analyzer.data_loader.years)
        print(f"\nTesting with year: {current_year}")

        for player_name, position in test_players:
            print(f"\n{'='*60}")
            print(f"DEBUGGING: {player_name} ({position})")
            print(f"{'='*60}")

            # Test player name cleaning
            cleaned_name = analyzer.data_loader.clean_player_name(player_name)
            cleaned_for_matching = analyzer.data_loader.clean_player_name_for_matching(player_name)

            print(f"Original name: '{player_name}'")
            print(f"Cleaned name: '{cleaned_name}'")
            print(f"Cleaned for matching: '{cleaned_for_matching}'")

            # Check if player exists in advanced data
            if position in analyzer.data_loader.advanced_data and current_year in analyzer.data_loader.advanced_data[position]:
                advanced_df = analyzer.data_loader.advanced_data[position][current_year]

                # Find player in advanced data
                player_found = False
                for col in ['PLAYER', 'Player', 'NAME', 'Name']:
                    if col in advanced_df.columns:
                        matches = advanced_df[advanced_df[col].str.contains(cleaned_name, case=False, na=False)]
                        if not matches.empty:
                            print(f"✓ Found in advanced data: {col} column")
                            print(f"  Raw name in data: '{matches[col].iloc[0]}'")
                            player_found = True
                            break

                if not player_found:
                    print("❌ Not found in advanced data")
                    # Show some sample names from advanced data
                    if 'Player' in advanced_df.columns:
                        print(f"Sample {position} players in advanced data:")
                        for i, name in enumerate(advanced_df['Player'].head(5)):
                            print(f"  {i+1}. '{name}' -> cleaned: '{analyzer.data_loader.clean_player_name(name)}'")
                    continue

            # Check ADP data
            if current_year in analyzer.data_loader.adp_data:
                adp_df = analyzer.data_loader.adp_data[current_year]
                print(f"\nADP Data columns: {list(adp_df.columns)}")

                # Find position column
                pos_col = None
                for col in ['Position', 'Pos', 'POSITION', 'POS', 'position']:
                    if col in adp_df.columns:
                        pos_col = col
                        break

                if pos_col:
                    # Filter by position using startswith for "WR1", "RB2" format
                    pos_filtered = adp_df[adp_df[pos_col].notna() & adp_df[pos_col].str.upper().str.startswith(position.upper())]
                    print(f"Players in {position} position: {len(pos_filtered)}")

                    # Find player name column
                    player_col = None
                    for col in ['Player', 'PLAYER', 'Name', 'NAME', 'player_name']:
                        if col in pos_filtered.columns:
                            player_col = col
                            break

                    if player_col:
                        # Look for player with different matching strategies
                        print(f"\nTrying different matching strategies:")

                        # Strategy 1: Exact match with cleaned name
                        exact_matches = pos_filtered[pos_filtered[player_col].str.lower() == cleaned_name.lower()]
                        if not exact_matches.empty:
                            print(f"✓ Exact match found: '{exact_matches[player_col].iloc[0]}'")
                        else:
                            print("❌ No exact match found")

                        # Strategy 2: Contains match with cleaned name
                        contains_matches = pos_filtered[pos_filtered[player_col].str.contains(cleaned_name, case=False, na=False)]
                        if not contains_matches.empty:
                            print(f"✓ Contains match found: '{contains_matches[player_col].iloc[0]}'")
                        else:
                            print("❌ No contains match found")

                        # Strategy 3: Try with original name (before cleaning)
                        original_name = player_name.split('(')[0].strip()
                        original_matches = pos_filtered[pos_filtered[player_col].str.contains(original_name, case=False, na=False)]
                        if not original_matches.empty:
                            print(f"✓ Original name match found: '{original_matches[player_col].iloc[0]}'")
                        else:
                            print("❌ No original name match found")

                        # Show some sample names from ADP data
                        print(f"\nSample {position} players in ADP data:")
                        for i, name in enumerate(pos_filtered[player_col].head(5)):
                            print(f"  {i+1}. '{name}' -> cleaned: '{analyzer.data_loader.clean_player_name(name)}'")

            # Check results data
            if current_year in analyzer.data_loader.results_data:
                results_df = analyzer.data_loader.results_data[current_year]
                print(f"\nResults Data columns: {list(results_df.columns)}")

                # Find position column
                pos_col = None
                for col in ['Position', 'Pos', 'POSITION', 'POS', 'position']:
                    if col in results_df.columns:
                        pos_col = col
                        break

                if pos_col:
                    # Filter by position
                    pos_filtered = results_df[results_df[pos_col].str.upper() == position.upper()]
                    print(f"Players in {position} position: {len(pos_filtered)}")

                    # Find player name column
                    player_col = None
                    for col in ['Player', 'PLAYER', 'Name', 'NAME', 'player_name']:
                        if col in pos_filtered.columns:
                            player_col = col
                            break

                    if player_col:
                        # Look for player with different matching strategies
                        print(f"\nTrying different matching strategies:")

                        # Strategy 1: Exact match with cleaned name
                        exact_matches = pos_filtered[pos_filtered[player_col].str.lower() == cleaned_name.lower()]
                        if not exact_matches.empty:
                            print(f"✓ Exact match found: '{exact_matches[player_col].iloc[0]}'")
                        else:
                            print("❌ No exact match found")

                        # Strategy 2: Contains match with cleaned name
                        contains_matches = pos_filtered[pos_filtered[player_col].str.contains(cleaned_name, case=False, na=False)]
                        if not contains_matches.empty:
                            print(f"✓ Contains match found: '{contains_matches[player_col].iloc[0]}'")
                        else:
                            print("❌ No contains match found")

                        # Strategy 3: Try with original name (before cleaning)
                        original_name = player_name.split('(')[0].strip()
                        original_matches = pos_filtered[pos_filtered[player_col].str.contains(original_name, case=False, na=False)]
                        if not original_matches.empty:
                            print(f"✓ Original name match found: '{original_matches[player_col].iloc[0]}'")
                        else:
                            print("❌ No original name match found")

                        # Show some sample names from results data
                        print(f"\nSample {position} players in results data:")
                        for i, name in enumerate(pos_filtered[player_col].head(5)):
                            print(f"  {i+1}. '{name}' -> cleaned: '{analyzer.data_loader.clean_player_name(name)}'")

            # Test historical performance calculation
            print(f"\nTesting historical performance calculation...")
            hist_perf = analyzer.player_analyzer.calculate_historical_performance(player_name, position)
            print(f"Historical performance result:")
            print(f"  - Years of data: {hist_perf.get('years_of_data', 0)}")
            print(f"  - Has historical data: {hist_perf.get('has_historical_data', False)}")
            print(f"  - Data quality: {hist_perf.get('data_quality', 'unknown')}")

            if hist_perf.get('years_of_data', 0) > 0:
                print(f"  - ADP differentials: {hist_perf.get('adp_differential', 0):.1f}")
                print(f"  - Fantasy points: {hist_perf.get('peak_season', 0):.1f}")
            else:
                print(f"  ❌ No historical data found!")

        print(f"\n{'='*60}")
        print("DEBUG COMPLETE")
        print(f"{'='*60}")

    except Exception as e:
        print(f"❌ Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_historical_data() 