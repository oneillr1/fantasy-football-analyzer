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
    test_enhanced_scoring() 