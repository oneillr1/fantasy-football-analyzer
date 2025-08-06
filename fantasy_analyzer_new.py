"""
Fantasy Football Analyzer - Main Orchestrator

This is the new modular version of the fantasy football analyzer that coordinates
all the specialized modules for data loading, scoring, ML models, analysis, and reporting.
"""

from typing import Dict, List, Optional, Any
import warnings

from modules.data_loader import DataLoader
from modules.scoring_engine import ScoringEngine
from modules.ml_models import MLModels
from modules.player_analysis import PlayerAnalyzer
from modules.league_analysis import LeagueAnalyzer
from modules.value_analysis import ValueAnalyzer
from modules.reporting import ReportGenerator
from modules.utils import FantasyUtils

warnings.filterwarnings('ignore')


class FantasyFootballAnalyzer:
    """
    Main orchestrator for the Fantasy Football Analyzer.
    
    This class coordinates all the specialized modules to provide comprehensive
    fantasy football analysis capabilities.
    """
    
    def __init__(self, data_directory: str, enable_league_analysis: bool = False, enable_testing_mode: bool = False):
        """
        Initialize the analyzer with modular components.
        
        Args:
            data_directory: Path to directory containing CSV files
            enable_league_analysis: If True, enables league mate analysis features
            enable_testing_mode: If True, enables testing/feedback mode
        """
        # Initialize core components
        self.data_loader = DataLoader(data_directory)
        self.scoring_engine = ScoringEngine(self.data_loader)
        self.ml_models = MLModels(self.data_loader, self.scoring_engine)  # NEW: Pass scoring engine
        
        # Initialize analysis components
        self.player_analyzer = PlayerAnalyzer(self.data_loader, self.scoring_engine, self.ml_models)
        self.league_analyzer = LeagueAnalyzer(self.data_loader)
        self.value_analyzer = ValueAnalyzer(self.data_loader, self.scoring_engine)
        self.report_generator = ReportGenerator(
            self.data_loader, self.scoring_engine, self.player_analyzer, 
            self.league_analyzer, self.value_analyzer
        )
        
        # Configuration
        self.enable_league_analysis = enable_league_analysis
        self.enable_testing_mode = enable_testing_mode
        
        # Load data and train models
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the system by loading data and training models."""
        print("Initializing Fantasy Football Analyzer...")
        
        # 1. Diagnose and load data
        self.data_loader.diagnose_csv_files()
        self.data_loader.load_data()
        
        # 2. Train ML models
        self.ml_models.train_models()
        
        print("System initialization complete!")
    
    def diagnose_csv_files(self) -> None:
        """Diagnose CSV files for potential issues."""
        self.data_loader.diagnose_csv_files()
    
    def load_data(self) -> None:
        """Load all data files."""
        self.data_loader.load_data()
    
    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        return self.report_generator.generate_comprehensive_report()
    
    def save_report(self, filename: str = "fantasy_analysis_report.txt") -> str:
        """Save the comprehensive report to a file."""
        return self.report_generator.save_report(filename)
    
    def generate_player_profiles(self) -> List[Dict[str, Any]]:
        """Generate comprehensive player profiles."""
        return self.report_generator.generate_player_profiles()
    
    def save_player_profiles(self, profiles: List[Dict[str, Any]], filename: str = "player_profiles.txt") -> str:
        """Save player profiles to a text file."""
        return self.report_generator.save_player_profiles(profiles, filename)
    
    def save_profiles_json(self, profiles: List[Dict[str, Any]], filename: str = "player_profiles.json") -> str:
        """Save player profiles to a JSON file."""
        return self.report_generator.save_profiles_json(profiles, filename)
    
    def run_scoring_test(self, num_players: int = 10) -> str:
        """Run a scoring system test."""
        return self.report_generator.run_scoring_test(num_players)
    
    def analyze_draft_reaches_and_values(self) -> str:
        """Analyze draft reaches and values."""
        return self.league_analyzer.analyze_draft_reaches_and_values()
    
    def analyze_league_mate_tendencies(self) -> str:
        """Analyze league mate tendencies."""
        return self.league_analyzer.analyze_league_mate_tendencies()
    
    def analyze_league_draft_flow(self) -> str:
        """Analyze league draft flow."""
        return self.league_analyzer.analyze_league_draft_flow()
    
    def analyze_roneill19_tendencies(self) -> str:
        """Analyze roneill19 tendencies."""
        return self.league_analyzer.analyze_roneill19_tendencies()
    
    def analyze_player_value_predictions(self) -> str:
        """Analyze player value predictions."""
        return self.value_analyzer.analyze_player_value_predictions()
    
    def analyze_adp_value_mismatches(self) -> str:
        """Analyze ADP value mismatches."""
        return self.value_analyzer.analyze_adp_value_mismatches()
    
    def find_position_values(self, position: str, min_adp: int = 1, max_adp: int = 200) -> List[Dict[str, Any]]:
        """Find value players for a specific position."""
        return self.value_analyzer.find_position_values(position, min_adp, max_adp)
    
    def generate_player_profile(self, player_name: str, position: str, current_year: int = 2024) -> Dict[str, Any]:
        """Generate a comprehensive player profile."""
        return self.player_analyzer.generate_player_profile(player_name, position, current_year)
    
    def calculate_historical_performance(self, player: str, position: str) -> Dict[str, float]:
        """Calculate historical performance for a player."""
        return self.player_analyzer.calculate_historical_performance(player, position)
    
    def get_injury_profile(self, player: str, position: str) -> Dict[str, Any]:
        """Get injury profile for a player."""
        return self.player_analyzer.get_injury_profile(player, position)
    
    def perform_basic_trend_analysis(self, player_name: str, position: str) -> Dict[str, Any]:
        """Perform basic trend analysis for a player."""
        return self.player_analyzer.perform_basic_trend_analysis(player_name, position)
    
    def get_actual_fantasy_points(self, player_name: str, position: str, year: int = 2024) -> Optional[float]:
        """Get actual fantasy points for a player."""
        return self.player_analyzer.get_actual_fantasy_points(player_name, position, year)
    
    def calculate_advanced_score_universal(self, player_row, position: str) -> tuple:
        """Calculate advanced score using universal system."""
        return self.scoring_engine.calculate_advanced_score_universal(player_row, position)
    
    def calculate_overall_profile_score(self, historical_perf: Dict[str, float], 
                                      advanced_metrics: Dict[str, Any], 
                                      ml_predictions: Dict[str, float], 
                                      injury_profile: Dict[str, Any], 
                                      debug_mode: bool = False,
                                      position: str = None, 
                                      player_row: Any = None, 
                                      player_name: str = "unknown") -> Dict[str, float]:
        """Calculate overall profile score."""
        return self.scoring_engine.calculate_overall_profile_score(
            historical_perf, advanced_metrics, ml_predictions, injury_profile,
            debug_mode, position, player_row, player_name
        )
    
    def save_debug_logs(self, filename_prefix: str = "data_debug"):
        """Save debug logs."""
        self.data_loader.save_debug_logs(filename_prefix)


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your data directory
    # Set enable_testing_mode=True to run scoring tests instead of full analysis
    analyzer = FantasyFootballAnalyzer("fantasy_data", enable_testing_mode=False)
    
    # Check if testing mode is enabled
    if analyzer.enable_testing_mode:
        print("\nTESTING MODE ENABLED")
        print("="*40)
        print("Running scoring system test with 10 random players...")
        analyzer.run_scoring_test()
        print("\nTesting complete! To run full analysis, set enable_testing_mode=False")
        exit()
    
    # Generate dual output system
    print("\n" + "="*60)
    print("GENERATING DUAL OUTPUT SYSTEM")
    print("="*60)
    
    # 1. Generate league analysis (existing functionality)
    print("Generating league analysis report...")
    league_report = analyzer.save_report("league_analysis.txt")
    print(f"League analysis saved to: {league_report}")
    
    # 2. Generate ML-enhanced player profiles
    print("\nGenerating ML-enhanced player profiles...")
    player_profiles = analyzer.generate_player_profiles()
    profiles_file = analyzer.save_player_profiles(player_profiles, "player_profiles.txt")
    json_file = analyzer.save_profiles_json(player_profiles, "player_profiles.json")
    
    print(f" Player profiles saved to: {profiles_file}")
    print(f" JSON profiles for Custom GPT saved to: {json_file}")
    print(f" Generated profiles for {len(player_profiles)} players")
    
    # Summary
    print(f"\n ANALYSIS COMPLETE!")
    print("="*40)
    print("Generated Files:")
    print(f"  • {league_report} - League draft tendencies and manager profiles")
    print(f"  • {profiles_file} - ML-enhanced player profiles")
    print(f"  • {json_file} - JSON format for Custom GPT integration")
    print("\nReady for 2025 fantasy draft! ")
    print("\nTo run scoring tests: Set enable_testing_mode=True and run again") 