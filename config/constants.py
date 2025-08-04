"""
Constants and configuration for Fantasy Football Analyzer
"""

# Analysis Thresholds
SIGNIFICANT_REACH_THRESHOLD = 12
SIGNIFICANT_VALUE_THRESHOLD = -12
ADP_OUTPERFORM_THRESHOLD = 15
BREAKOUT_THRESHOLD = 20
REGRESSION_THRESHOLD = -20
MIN_PICKS_FOR_ANALYSIS = 10
OPTIMAL_POCKET_TIME = 2.7

# Scoring System Weights (Universal Scoring System)
SCORING_WEIGHTS = {
    'volume': 0.20,      # 20% weight for volume metrics
    'efficiency': 0.35,  # 35% weight for efficiency metrics
    'explosiveness': 0.25,  # 25% weight for explosiveness metrics
    'opportunity': 0.15,    # 15% weight for opportunity metrics
    'negative': -0.05       # -5% weight for negative metrics
}

# Profile Score Weights
PROFILE_WEIGHTS = {
    'historical': 0.25,   # 25% weight for historical performance
    'advanced': 0.30,     # 30% weight for advanced metrics
    'ml_predictions': 0.25,  # 25% weight for ML predictions
    'injury': 0.20        # 20% weight for injury profile
}

# League Scoring Settings (Half PPR)
LEAGUE_SCORING = {
    'passing': {
        'yards_per_point': 25,  # 1 point per 25 yards
        'td_points': 6,         # 6 points per TD
        'int_points': -2        # -2 for interceptions
    },
    'rushing': {
        'yards_per_point': 10,  # 1 point per 10 yards
        'td_points': 6,         # 6 points per TD
        'fumble_points': -2     # -2 for fumbles
    },
    'receiving': {
        'yards_per_point': 10,  # 1 point per 10 yards
        'td_points': 6,         # 6 points per TD
        'reception_points': 0.5 # Half PPR - 0.5 points per reception
    },
    'bonuses': {
        'long_td_bonus': 2      # +2 points for 40+ yard TDs
    }
}

# Key Advanced Metrics by Position
KEY_METRICS = {
    'QB': ['PCT', 'Y/A', 'AIR/A', '10+ YDS', '20+ YDS', 'PKT TIME', 'RTG', 'SACK'],
    'RB': ['Y/ATT', 'YACON/ATT', 'BRKTKL', '10+ YDS', '20+ YDS', 'REC', 'TGT', 'RZ TGT'],
    'WR': ['Y/R', 'YAC/R', 'YACON/R', 'BRKTKL', '% TM', 'CATCHABLE', '10+ YDS', '20+ YDS', 'RZ TGT'],
    'TE': ['Y/R', 'YAC/R', 'YACON/R', 'BRKTKL', '% TM', 'CATCHABLE', '10+ YDS', '20+ YDS', 'RZ TGT']
}

# Metric Categories for Universal Scoring
METRIC_CATEGORIES = {
    'volume': ['YDS', 'ATT', 'REC', 'TGT', 'TD', 'CMP', 'ATT'],
    'efficiency': ['Y/A', 'Y/R', 'PCT', 'RTG', 'YAC/R', 'YACON/R', 'YACON/ATT'],
    'explosiveness': ['10+ YDS', '20+ YDS', '40+ YDS', 'BRKTKL', 'AIR/A'],
    'opportunity': ['TGT', 'RZ TGT', '% TM', 'CATCHABLE', 'PKT TIME'],
    'negative': ['SACK', 'INT', 'FUM', 'DROP']
}

# File Naming Patterns
FILE_PATTERNS = {
    'draft': 'draft_YYYY.csv',
    'adp': 'adp_YYYY.csv',
    'results': 'results_YYYY.csv',
    'advanced': 'advanced_[pos]_YYYY.csv',
    'injury': 'injury_predictor_[pos].csv'
}

# Supported Positions
POSITIONS = ['QB', 'RB', 'WR', 'TE']

# Supported Years (2019-2024)
SUPPORTED_YEARS = list(range(2019, 2025))

# ML Model Types
ML_MODEL_TYPES = ['breakout', 'consistency', 'positional_value', 'injury_risk']

# Default Values
DEFAULT_SCORE = 0.0
DEFAULT_WEIGHT = 1.0
DEFAULT_CONFIDENCE = 0.5 