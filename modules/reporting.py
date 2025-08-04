"""
Reporting Module for Fantasy Football Analyzer

Handles all report generation and file output.
"""

from typing import Dict, List, Optional, Any
import json
from datetime import datetime

from config.constants import POSITIONS, DEFAULT_SCORE


class ReportGenerator:
    """
    Handles all report generation and file output.
    """
    
    def __init__(self, data_loader, scoring_engine, player_analyzer, league_analyzer, value_analyzer):
        """
        Initialize the report generator.
        
        Args:
            data_loader: DataLoader instance
            scoring_engine: ScoringEngine instance
            player_analyzer: PlayerAnalyzer instance
            league_analyzer: LeagueAnalyzer instance
            value_analyzer: ValueAnalyzer instance
        """
        self.data_loader = data_loader
        self.scoring_engine = scoring_engine
        self.player_analyzer = player_analyzer
        self.league_analyzer = league_analyzer
        self.value_analyzer = value_analyzer
    
    def generate_comprehensive_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Complete analysis report as string
        """
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*60)
        
        report_sections = []
        
        # 1. League Analysis
        report_sections.append("LEAGUE ANALYSIS")
        report_sections.append("="*60)
        report_sections.append(self.league_analyzer.analyze_draft_reaches_and_values())
        report_sections.append("\n" + self.league_analyzer.analyze_league_mate_tendencies())
        report_sections.append("\n" + self.league_analyzer.analyze_league_draft_flow())
        report_sections.append("\n" + self.league_analyzer.analyze_roneill19_tendencies())
        
        # 2. Value Analysis
        report_sections.append("\n\nVALUE ANALYSIS")
        report_sections.append("="*60)
        report_sections.append(self.value_analyzer.analyze_player_value_predictions())
        report_sections.append("\n" + self.value_analyzer.analyze_adp_value_mismatches())
        
        # 3. Data Quality Report
        report_sections.append("\n\nDATA QUALITY REPORT")
        report_sections.append("="*60)
        report_sections.append(self._generate_data_quality_report())
        
        # 4. Scoring System Report
        report_sections.append("\n\nSCORING SYSTEM REPORT")
        report_sections.append("="*60)
        report_sections.append(self._generate_scoring_system_report())
        
        return "\n".join(report_sections)
    
    def save_report(self, filename: str = "fantasy_analysis_report.txt") -> str:
        """
        Save the comprehensive report to a file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        report_content = self.generate_comprehensive_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"Comprehensive report saved to: {filename}")
        return filename
    
    def generate_player_profiles(self) -> List[Dict[str, Any]]:
        """
        Generate comprehensive player profiles for all players with data.
        
        Returns:
            List of player profile dictionaries
        """
        print("\n" + "="*60)
        print("GENERATING PLAYER PROFILES")
        print("="*60)
        
        profiles = []
        current_year = max(self.data_loader.years)
        
        for position in POSITIONS:
            print(f"Processing {position} players...")
            
            if position not in self.data_loader.advanced_data or current_year not in self.data_loader.advanced_data[position]:
                continue
            
            df = self.data_loader.advanced_data[position][current_year]
            player_col = self.data_loader.get_player_column(df, f"Advanced {position}")
            
            if not player_col:
                continue
            
            for _, row in df.iterrows():
                player_name = row.get(player_col, '')
                if pd.isna(player_name) or not isinstance(player_name, str):
                    continue
                
                # Generate comprehensive profile
                try:
                    profile = self.player_analyzer.generate_player_profile(player_name, position, current_year)
                    profiles.append(profile)
                except Exception as e:
                    print(f"Error generating profile for {player_name}: {e}")
                    continue
        
        print(f"Generated {len(profiles)} player profiles")
        return profiles
    
    def save_player_profiles(self, profiles: List[Dict[str, Any]], filename: str = "player_profiles.txt") -> str:
        """
        Save player profiles to a text file.
        
        Args:
            profiles: List of player profile dictionaries
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("FANTASY FOOTBALL PLAYER PROFILES\n")
            f.write("="*60 + "\n\n")
            
            for profile in profiles:
                player_name = profile.get('player_name', 'Unknown')
                position = profile.get('position', 'Unknown')
                overall_scores = profile.get('overall_scores', {})
                
                f.write(f"{player_name.upper()} ({position})\n")
                f.write("-" * 40 + "\n")
                
                # Overall scores
                final_score = overall_scores.get('final_score', 0)
                f.write(f"Overall Score: {final_score:.2f}/10\n")
                f.write(f"Historical Score: {overall_scores.get('historical_score', 0):.2f}/10\n")
                f.write(f"Advanced Score: {overall_scores.get('advanced_score', 0):.2f}/10\n")
                f.write(f"ML Score: {overall_scores.get('ml_score', 0):.2f}/10\n")
                f.write(f"Injury Score: {overall_scores.get('injury_score', 0):.2f}/10\n")
                
                # Historical performance
                historical = profile.get('historical_performance', {})
                if historical.get('has_historical_data', False):
                    f.write(f"\nHistorical Performance:\n")
                    f.write(f"  Years of Data: {historical.get('years_of_data', 0)}\n")
                    f.write(f"  ADP Differential: {historical.get('adp_differential', 0):+.1f}\n")
                    f.write(f"  Peak Season: {historical.get('peak_season', 0):.0f} points\n")
                    f.write(f"  Consistency: {historical.get('consistency', 0):.1f}%\n")
                
                # Advanced metrics
                advanced_metrics = profile.get('advanced_metrics', {})
                if advanced_metrics:
                    f.write(f"\nKey Advanced Metrics:\n")
                    for metric, value in list(advanced_metrics.items())[:5]:  # Top 5
                        f.write(f"  {metric}: {value:.2f}\n")
                
                # ML predictions
                ml_predictions = profile.get('ml_predictions', {})
                if ml_predictions:
                    f.write(f"\nML Predictions:\n")
                    for pred_type, score in ml_predictions.items():
                        f.write(f"  {pred_type}: {score:.3f}\n")
                
                # Injury profile
                injury_profile = profile.get('injury_profile', {})
                if injury_profile.get('has_injury_data', False):
                    f.write(f"\nInjury Profile:\n")
                    f.write(f"  Injury Risk: {injury_profile.get('injury_risk', 0):.2f}\n")
                    f.write(f"  Games Missed: {injury_profile.get('games_missed', 0)}\n")
                
                # Trend analysis
                trend_analysis = profile.get('trend_analysis', {})
                if trend_analysis.get('adp_trend'):
                    f.write(f"\nTrend Analysis:\n")
                    f.write(f"  Consistency Score: {trend_analysis.get('consistency_score', 0):.1f}%\n")
                    f.write(f"  Breakout Potential: {trend_analysis.get('breakout_potential', 0):.1f}%\n")
                    risk_factors = trend_analysis.get('risk_factors', [])
                    if risk_factors:
                        f.write(f"  Risk Factors: {', '.join(risk_factors)}\n")
                
                # Similar players
                similar_players = profile.get('similar_players', [])
                if similar_players:
                    f.write(f"\nSimilar Players:\n")
                    for i, similar in enumerate(similar_players[:3]):  # Top 3
                        f.write(f"  {i+1}. {similar['player']} (similarity: {similar['similarity']:.3f})\n")
                
                f.write("\n" + "="*60 + "\n\n")
        
        print(f"Player profiles saved to: {filename}")
        return filename
    
    def save_profiles_json(self, profiles: List[Dict[str, Any]], filename: str = "player_profiles.json") -> str:
        """
        Save player profiles to a JSON file for programmatic access.
        
        Args:
            profiles: List of player profile dictionaries
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        # Convert profiles to JSON-serializable format
        json_profiles = []
        for profile in profiles:
            # Convert numpy types to Python types
            json_profile = {}
            for key, value in profile.items():
                if hasattr(value, 'item'):  # numpy scalar
                    json_profile[key] = value.item()
                elif isinstance(value, dict):
                    json_profile[key] = {}
                    for k, v in value.items():
                        if hasattr(v, 'item'):
                            json_profile[key][k] = v.item()
                        else:
                            json_profile[key][k] = v
                else:
                    json_profile[key] = value
            json_profiles.append(json_profile)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_profiles, f, indent=2, ensure_ascii=False)
        
        print(f"JSON profiles saved to: {filename}")
        return filename
    
    def run_scoring_test(self, num_players: int = 10) -> str:
        """
        Run a scoring system test with sample players.
        
        Args:
            num_players: Number of players to test
            
        Returns:
            Test results as string
        """
        print("\n" + "="*60)
        print("RUNNING SCORING SYSTEM TEST")
        print("="*60)
        
        test_results = []
        test_results.append("SCORING SYSTEM TEST RESULTS")
        test_results.append("="*60)
        
        # Get sample players from each position
        current_year = max(self.data_loader.years)
        test_players = []
        
        for position in POSITIONS:
            if position not in self.data_loader.advanced_data or current_year not in self.data_loader.advanced_data[position]:
                continue
            
            df = self.data_loader.advanced_data[position][current_year]
            player_col = self.data_loader.get_player_column(df, f"Advanced {position}")
            
            if not player_col:
                continue
            
            # Get sample players
            sample_size = max(1, num_players // len(POSITIONS))
            for i, (_, row) in enumerate(df.iterrows()):
                if i >= sample_size:
                    break
                
                player_name = row.get(player_col, '')
                if pd.isna(player_name) or not isinstance(player_name, str):
                    continue
                
                test_players.append((player_name, position, row))
        
        # Test each player
        for i, (player_name, position, player_row) in enumerate(test_players):
            test_results.append(f"\n{'='*60}")
            test_results.append(f"TEST PLAYER #{i+1}: {player_name} ({position})")
            test_results.append(f"{'='*60}")
            
            # Get advanced metrics
            key_metrics = self.data_loader.key_metrics.get(position, [])
            advanced_metrics = {}
            for metric in key_metrics:
                if metric in player_row.index and not pd.isna(player_row[metric]):
                    advanced_metrics[metric] = player_row[metric]
            
            # Get historical performance
            historical_perf = self.player_analyzer.calculate_historical_performance(player_name, position)
            
            # REQUIRE real historical data - no fallbacks
            if not historical_perf.get('has_historical_data', False):
                self.data_loader._log_missing_data(player_name, position, "no_historical_data_for_test", 
                                                 [f"years_of_data: {historical_perf.get('years_of_data', 0)}"])
                continue  # Skip this player if no real data
            
            # Get injury profile
            injury_profile = self.player_analyzer.get_injury_profile(player_name, position)
            
            # Get ML predictions
            features = self.player_analyzer.ml_models._extract_features(player_row, position)
            ml_predictions = {}
            if features is not None:
                ml_predictions = self.player_analyzer.ml_models.generate_predictions(player_name, position, features)
            
            # Calculate overall scores
            overall_scores = self.scoring_engine.calculate_overall_profile_score(
                historical_perf, advanced_metrics, ml_predictions, injury_profile,
                debug_mode=True, position=position, player_row=player_row, player_name=player_name)
            
            # Display results
            test_results.append("\nDETAILED SCORING BREAKDOWN:")
            test_results.append("-" * 40)
            
            # 1. Historical Performance Score - REAL DATA ONLY
            test_results.append(f"\n1. HISTORICAL PERFORMANCE (Weight: 25%):")
            test_results.append(f"   • Data Quality: REAL DATA")
            
            # Show historical scoring structure
            adp_differential = historical_perf.get('adp_differential', 0)
            peak_season = historical_perf.get('peak_season', 0)
            consistency = historical_perf.get('consistency', 0)
            experience = historical_perf.get('years_of_data', 0)
            
            test_results.append(f"   • ADP Differential: {adp_differential:+.1f} (vs ADP)")
            test_results.append(f"   • Peak Season: {peak_season:.0f} points")
            test_results.append(f"   • Consistency: {consistency:.1f}%")
            test_results.append(f"   • Experience: {experience} years")
            
            hist_score = overall_scores.get('historical_score', 0)
            if hist_score > 0:
                test_results.append(f"     Final: {hist_score:.2f}/10 points")
            else:
                hist_score = 0.0  # No score for missing data
                test_results.append(f"   • No historical data available")
                test_results.append(f"   • Score: {hist_score:.2f}/10 (no fallback)")
            
            # 2. Advanced Metrics Score (30% weight)
            test_results.append(f"\n2. ADVANCED METRICS (Weight: 30%):")
            test_results.append(f"   • Available Metrics: {len(advanced_metrics)}")
            
            if advanced_metrics:
                test_results.append(f"   • Key Metrics:")
                for metric, value in list(advanced_metrics.items())[:5]:  # Show top 5
                    test_results.append(f"     - {metric}: {value:.2f}")
                
                adv_score = overall_scores.get('advanced_score', 0)
                test_results.append(f"     Final: {adv_score:.2f}/10 points")
            else:
                adv_score = 0.0
                test_results.append(f"   • No advanced metrics available")
                test_results.append(f"   • Score: {adv_score:.2f}/10 (no fallback)")
            
            # 3. ML Predictions Score (25% weight)
            test_results.append(f"\n3. ML PREDICTIONS (Weight: 25%):")
            test_results.append(f"   • Available Predictions: {len(ml_predictions)}")
            
            if ml_predictions:
                for pred_type, score in ml_predictions.items():
                    test_results.append(f"   • {pred_type}: {score:.3f}")
                
                ml_score = overall_scores.get('ml_score', 0)
                test_results.append(f"     Final: {ml_score:.2f}/10 points")
            else:
                ml_score = 0.0
                test_results.append(f"   • No ML predictions available")
                test_results.append(f"   • Score: {ml_score:.2f}/10 (no fallback)")
            
            # 4. Injury Profile Score (20% weight)
            test_results.append(f"\n4. INJURY PROFILE (Weight: 20%):")
            
            if injury_profile.get('has_injury_data', False):
                injury_risk = injury_profile.get('injury_risk', 0.5)
                games_missed = injury_profile.get('games_missed', 0)
                test_results.append(f"   • Injury Risk: {injury_risk:.2f}")
                test_results.append(f"   • Games Missed: {games_missed}")
                
                injury_score = overall_scores.get('injury_score', 0)
                test_results.append(f"     Final: {injury_score:.2f}/10 points")
            else:
                injury_score = 0.0
                test_results.append(f"   • No injury data available")
                test_results.append(f"   • Score: {injury_score:.2f}/10 (no fallback)")
            
            # Final score
            final_score = overall_scores.get('final_score', 0)
            test_results.append(f"\nFINAL SCORE: {final_score:.2f}/10 ({final_score*10:.0f}/100)")
            
            # Star rating
            star_rating = min(5, max(1, int((final_score / 10) * 5) + 1))
            stars = "*" * star_rating + "-" * (5 - star_rating)
            test_results.append(f"STAR RATING: {stars} ({star_rating}/5)")
        
        # Add missing data report
        test_results.append(f"\n{'='*80}")
        test_results.append("MISSING DATA REPORT")
        test_results.append(f"{'='*80}")
        
        if self.data_loader.missing_data_log:
            test_results.append("Data gaps identified during testing:")
            test_results.append("-" * 40)
            for entry in self.data_loader.missing_data_log:
                test_results.append(f"• Player: {entry['player']} ({entry['position']})")
                test_results.append(f"  Missing: {entry['missing_stat']}")
                test_results.append(f"  Attempted columns: {entry['attempted_columns']}")
                test_results.append(f"  Data source: {entry.get('data_source', 'unknown')}")
                test_results.append(f"  Time: {entry['timestamp']}")
                test_results.append("")
        else:
            test_results.append("OK - No missing data detected during test")
        
        # Add summary
        test_results.append(f"\n{'='*80}")
        test_results.append("TEST SUMMARY")
        test_results.append(f"{'='*80}")
        test_results.append("Scoring system test completed successfully")
        test_results.append("Review the individual component scores above")
        test_results.append("Provide feedback on scoring weights or calculations")
        test_results.append("Run test again with: analyzer.run_scoring_test()")
        
        result_text = "\n".join(test_results)
        print(result_text)
        return result_text
    
    def _generate_data_quality_report(self) -> str:
        """Generate a data quality report."""
        report = []
        report.append("DATA QUALITY REPORT")
        report.append("-" * 40)
        
        # Data availability
        report.append(f"Years Available: {self.data_loader.years}")
        report.append(f"Draft Data: {len(self.data_loader.draft_data)} years")
        report.append(f"ADP Data: {len(self.data_loader.adp_data)} years")
        report.append(f"Results Data: {len(self.data_loader.results_data)} years")
        report.append(f"Advanced Data: {len(self.data_loader.advanced_data)} positions")
        report.append(f"Injury Data: {len(self.data_loader.injury_data)} positions")
        
        # Missing data summary
        if self.data_loader.missing_data_log:
            report.append(f"\nMissing Data Issues: {len(self.data_loader.missing_data_log)}")
            for entry in self.data_loader.missing_data_log[:5]:  # Show first 5
                report.append(f"  • {entry['player']} ({entry['position']}): {entry['missing_stat']}")
        else:
            report.append("\nNo missing data issues detected")
        
        return "\n".join(report)
    
    def _generate_scoring_system_report(self) -> str:
        """Generate a scoring system report."""
        report = []
        report.append("SCORING SYSTEM REPORT")
        report.append("-" * 40)
        
        report.append("Universal Scoring System Components:")
        report.append("• Volume Metrics (20%): Raw production numbers")
        report.append("• Efficiency Metrics (35%): Per-attempt and per-game efficiency")
        report.append("• Explosiveness Metrics (25%): Big-play potential and ceiling")
        report.append("• Opportunity Metrics (15%): Usage and role indicators")
        report.append("• Negative Metrics (-5%): Penalties for turnovers and inefficiency")
        
        report.append("\nProfile Score Weights:")
        report.append("• Historical Performance (25%): Past ADP vs finish performance")
        report.append("• Advanced Metrics (30%): Universal scoring system")
        report.append("• ML Predictions (25%): Machine learning model outputs")
        report.append("• Injury Profile (20%): Injury risk assessment")
        
        report.append("\nData Quality Requirements:")
        report.append("• No fallback data used")
        report.append("• Missing data results in 0.0 scores")
        report.append("• All missing data is logged for debugging")
        
        return "\n".join(report) 