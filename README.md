# Fantasy Football Analyzer

A comprehensive Python-based fantasy football analysis tool that leverages historical draft data, advanced metrics, and machine learning to provide insights for fantasy football draft strategy and league management.

## Features

### Core Analysis Capabilities
- **Historical Performance Analysis**: Analyzes 6 years of draft data, results, and ADP trends
- **Advanced Metrics Integration**: Incorporates advanced QB, RB, WR, and TE metrics for deeper player evaluation
- **Machine Learning Predictions**: Uses Random Forest models for breakout predictions, consistency analysis, and injury risk assessment
- **League-Specific Analysis**: Analyzes league mate tendencies and draft patterns
- **Value Identification**: Identifies draft reaches, values, and ADP mismatches

### Data Sources
- **Draft Data**: 6 years of fantasy league draft results (2019-2024)
- **ADP Data**: Average Draft Position data across multiple platforms
- **Advanced Metrics**: QB, RB, WR, and TE advanced statistics
- **Injury Predictors**: Historical injury data for risk assessment
- **League Results**: Historical fantasy point finishes

### Key Components

#### `fantasy_analyzer.py`
The main analysis engine that:
- Loads and processes all CSV data files
- Performs comprehensive player profiling using the Universal Scoring System
- Generates machine learning predictions
- Creates detailed analysis reports
- Identifies draft values and reaches
- Enforces real data requirements with error logging for missing data

#### `sleeper_extractor.py`
Tool for extracting data from Sleeper fantasy football platform:
- League information and settings
- Draft history and picks
- Roster data
- Transaction history
- Historical league data

#### `scoring.py`
Scoring system analysis and extraction:
- League scoring settings analysis
- Scoring type classification
- Export functionality for scoring data

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd fantasy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Analysis
```python
from fantasy_analyzer import FantasyFootballAnalyzer

# Initialize analyzer with your data directory
analyzer = FantasyFootballAnalyzer('fantasy_data/')

# Generate comprehensive analysis
report = analyzer.generate_comprehensive_report()

# Save report to file
analyzer.save_report('my_analysis_report.txt')
```

### Player Profiling
```python
# Generate detailed player profiles
profiles = analyzer.generate_player_profiles()

# Save profiles to JSON
analyzer.save_profiles_json(profiles, 'player_profiles.json')
```

### League Analysis
```python
# Enable league analysis features
analyzer = FantasyFootballAnalyzer('fantasy_data/', enable_league_analysis=True)

# Analyze league mate tendencies
tendencies = analyzer.analyze_league_mate_tendencies()
```

## Data Structure

The analyzer expects the following CSV files in the `fantasy_data/` directory:

### Required Files
- `draft_YYYY.csv` - Draft results for each year (2019-2024)
- `adp_YYYY.csv` - Average Draft Position data for each year
- `results_YYYY.csv` - Fantasy point finishes for each year
- `advanced_[pos]_YYYY.csv` - Advanced metrics by position (QB, RB, WR, TE)

### Optional Files
- `injury_predictor_[pos].csv` - Injury prediction data by position

## Configuration

### Scoring Systems
The analyzer uses a **Universal Scoring System** that categorizes metrics into:
- **Volume Metrics** (20% weight): Raw production numbers
- **Efficiency Metrics** (35% weight): Per-attempt and per-game efficiency
- **Explosiveness Metrics** (25% weight): Big-play potential and ceiling
- **Opportunity Metrics** (15% weight): Usage and role indicators
- **Negative Metrics** (-5% weight): Penalties for turnovers and inefficiency

All metrics are normalized to a 0-10 scale and require real data - no fallback values are used.

### League Scoring Settings
The analyzer is configured for Half PPR scoring:
- Passing: 1 point per 25 yards, 6 points per TD, -2 for INTs
- Rushing: 1 point per 10 yards, 6 points per TD, -2 for fumbles
- Receiving: 1 point per 10 yards, 6 points per TD, 0.5 points per reception
- Bonuses: +2 points for 40+ yard TDs

### Analysis Thresholds
- Significant reach threshold: 12 picks
- Significant value threshold: -12 picks
- ADP outperform threshold: 15 picks
- Breakout threshold: 20 points
- Regression threshold: -20 points

### Data Quality Requirements
- **No Fallback Data**: The system requires real data for all calculations
- **Error Logging**: Missing data is logged to `data_debug_*.txt` files
- **Zero Scores**: Players with missing data receive 0.0 scores instead of neutral fallbacks

## Output Files

The analyzer generates several output files:
- `player_profiles.txt` - Detailed player analysis
- `player_profiles.json` - Structured player data
- `fantasy_analysis_report.txt` - Comprehensive analysis report
- `data_debug_*.txt` - Debug logs for data processing and missing data errors

## Recent Updates

### Scoring System Standardization (Latest)
- **Unified Scoring**: Replaced legacy scoring system with Universal Scoring System
- **Removed Fallbacks**: Eliminated all fallback data and neutral score defaults
- **Data Quality**: Now requires real data for all calculations
- **Error Logging**: Missing data is logged for debugging and data quality improvement
- **Zero Tolerance**: Players with missing data receive 0.0 scores instead of estimated values

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and entertainment purposes only. Fantasy football involves risk, and past performance does not guarantee future results. Always do your own research and make informed decisions. 