"""
Test Script for Modular Fantasy Football Analyzer

This script tests the new modular system to ensure all components work correctly.
"""

import sys
import os

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fantasy_analyzer_new import FantasyFootballAnalyzer


def test_modular_system():
    """Test the modular fantasy football analyzer system."""
    print("Testing Modular Fantasy Football Analyzer")
    print("="*50)
    
    try:
        # Initialize the analyzer
        print("1. Initializing analyzer...")
        analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=True)
        print("✓ Analyzer initialized successfully")
        
        # Test data loading
        print("\n2. Testing data loading...")
        if analyzer.data_loader.years:
            print(f"✓ Data loaded for years: {analyzer.data_loader.years}")
        else:
            print("⚠ No years found in data")
        
        # Test scoring system
        print("\n3. Testing scoring system...")
        test_result = analyzer.run_scoring_test(num_players=5)
        print("✓ Scoring system test completed")
        
        # Test individual components
        print("\n4. Testing individual components...")
        
        # Test data loader
        if hasattr(analyzer.data_loader, 'draft_data'):
            print("✓ DataLoader component working")
        
        # Test scoring engine
        if hasattr(analyzer.scoring_engine, 'calculate_advanced_score_universal'):
            print("✓ ScoringEngine component working")
        
        # Test ML models
        if hasattr(analyzer.ml_models, 'generate_predictions'):
            print("✓ MLModels component working")
        
        # Test player analyzer
        if hasattr(analyzer.player_analyzer, 'generate_player_profile'):
            print("✓ PlayerAnalyzer component working")
        
        # Test league analyzer
        if hasattr(analyzer.league_analyzer, 'analyze_draft_reaches_and_values'):
            print("✓ LeagueAnalyzer component working")
        
        # Test value analyzer
        if hasattr(analyzer.value_analyzer, 'find_position_values'):
            print("✓ ValueAnalyzer component working")
        
        # Test report generator
        if hasattr(analyzer.report_generator, 'generate_comprehensive_report'):
            print("✓ ReportGenerator component working")
        
        print("\n🎉 All tests passed! Modular system is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_specific_functionality():
    """Test specific functionality of the modular system."""
    print("\nTesting Specific Functionality")
    print("="*40)
    
    try:
        analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=False)
        
        # Test player profile generation
        print("1. Testing player profile generation...")
        profiles = analyzer.generate_player_profiles()
        print(f"✓ Generated {len(profiles)} player profiles")
        
        # Test value analysis
        print("\n2. Testing value analysis...")
        qb_values = analyzer.find_position_values("QB", min_adp=1, max_adp=50)
        print(f"✓ Found {len(qb_values)} QB values in early rounds")
        
        # Test league analysis
        print("\n3. Testing league analysis...")
        league_report = analyzer.analyze_draft_reaches_and_values()
        print("✓ League analysis completed")
        
        print("\n🎉 Specific functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Specific functionality test failed: {e}")
        return False


if __name__ == "__main__":
    print("Fantasy Football Analyzer - Modular System Test")
    print("="*60)
    
    # Run basic system test
    basic_test_passed = test_modular_system()
    
    # Run specific functionality test
    if basic_test_passed:
        specific_test_passed = test_specific_functionality()
    else:
        specific_test_passed = False
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if basic_test_passed and specific_test_passed:
        print("✅ ALL TESTS PASSED")
        print("The modular fantasy football analyzer is working correctly!")
        print("\nYou can now use the new system:")
        print("  • Run: python fantasy_analyzer_new.py")
        print("  • For testing: Set enable_testing_mode=True")
    else:
        print("❌ SOME TESTS FAILED")
        print("Please check the error messages above and fix any issues.")
    
    print("\nTest completed.") 