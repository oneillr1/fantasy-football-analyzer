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
    'efficiency': 0.30,  # 30% weight for efficiency metrics (CHANGED from 0.35)
    'explosiveness': 0.25,  # 25% weight for explosiveness metrics
    'opportunity': 0.20,    # 20% weight for opportunity metrics (CHANGED from 0.15)
    'negative': -0.05       # -5% weight for negative metrics
}

# Profile Score Weights
PROFILE_WEIGHTS = {
    'historical': 0.15,   # 15% weight for historical performance (CHANGED from 0.25)
    'advanced': 0.35,     # 35% weight for advanced metrics (CHANGED from 0.30)
    'ml_predictions': 0.30,  # 30% weight for ML predictions (CHANGED from 0.25)
    'injury': 0.20        # 20% weight for injury profile (CHANGED from 0.20)
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

# Comprehensive Advanced Metrics by Position
KEY_METRICS = {
    'QB': [
        # Efficiency metrics
        'PCT', 'Y/A', 'RTG', 'AIR/A',
        # Volume metrics  
        'COMP', 'ATT', 'YDS', 'AIR',
        # Explosive play metrics
        '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS',
        # Pressure metrics
        'PKT TIME', 'SACK', 'KNCK', 'HRRY', 'BLITZ',
        # Accuracy metrics
        'POOR', 'DROP',
        # Opportunity metrics
        'RZ ATT'
    ],
    'RB': [
        # Efficiency metrics
        'Y/ATT', 'YBCON/ATT', 'YACON/ATT',
        # Volume metrics
        'ATT', 'YDS', 'REC', 'TGT',
        # Explosive play metrics
        '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS', 'LNG TD', 'LNG',
        # Power/elusiveness metrics
        'BRKTKL', 'TK LOSS', 'TK LOSS YDS',
        # Opportunity metrics
        'RZ TGT'
    ],
    'WR': [
        # Efficiency metrics
        'Y/R', 'YBC/R', 'YAC/R', 'YACON/R', 'AIR/R',
        # Volume metrics
        'REC', 'YDS', 'TGT',
        # Explosive play metrics
        '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS', 'LNG',
        # Opportunity metrics
        '% TM', 'CATCHABLE', 'RZ TGT',
        # Playmaking metrics
        'BRKTKL', 'DROP'
    ],
    'TE': [
        # Efficiency metrics
        'Y/R', 'YBC/R', 'YAC/R', 'YACON/R', 'AIR/R',
        # Volume metrics
        'REC', 'YDS', 'TGT',
        # Explosive play metrics
        '10+ YDS', '20+ YDS', '30+ YDS', '40+ YDS', '50+ YDS', 'LNG',
        # Opportunity metrics
        '% TM', 'CATCHABLE', 'RZ TGT',
        # Playmaking metrics
        'BRKTKL', 'DROP'
    ]
}

# Comprehensive Metric Descriptions
METRIC_DESCRIPTIONS = {
    # QB Metrics
    'COMP': 'Completions - Total passes completed',
    'ATT': 'Attempts - Total passes thrown',
    'PCT': 'Completion Percentage - QB accuracy',
    'YDS': 'Passing Yards - Total passing production',
    'Y/A': 'Yards per Attempt - QB efficiency',
    'AIR': 'Air Yards - Downfield passing volume',
    'AIR/A': 'Air Yards per Attempt - QB downfield aggression',
    'RTG': 'Passer Rating - Overall QB efficiency',
    'PKT TIME': 'Pocket Time - QB processing speed/protection',
    'SACK': 'Sacks Taken - QB pressure handling',
    'KNCK': 'Knockdowns - Times hit while throwing',
    'HRRY': 'Hurries - Times pressured',
    'BLITZ': 'Times Blitzed - Pressure frequency',
    'POOR': 'Poor Passes - Inaccurate throws',
    'DROP': 'Drops by Receivers - WR/TE drops',
    'RZ ATT': 'Red Zone Attempts - TD opportunity',
    '10+ YDS': '10+ Yard Passes - Explosive plays',
    '20+ YDS': '20+ Yard Passes - Big plays',
    '30+ YDS': '30+ Yard Passes - Deep plays',
    '40+ YDS': '40+ Yard Passes - Very deep plays',
    '50+ YDS': '50+ Yard Passes - Extremely deep plays',
    
    # RB Metrics
    'ATT': 'Attempts - Rushing volume',
    'YDS': 'Yards - Total rushing production',
    'Y/ATT': 'Yards per Attempt - RB efficiency',
    'YBCON': 'Yards Before Contact - O-line quality/vision',
    'YACON': 'Yards After Contact - Power/elusiveness after hit',
    'YBCON/ATT': 'Yards Before Contact per Attempt - Blocking efficiency',
    'YACON/ATT': 'Yards After Contact per Attempt - RB power/elusiveness',
    'BRKTKL': 'Broken Tackles - Elusiveness and power',
    'TK LOSS': 'Tackled for Loss - Negative plays',
    'TK LOSS YDS': 'Tackled for Loss Yards - Negative yardage',
    'LNG TD': 'Longest Touchdown - Big play ability',
    'LNG': 'Longest Run - Best single play',
    '10+ YDS': '10+ Yard Runs - Explosive plays',
    '20+ YDS': '20+ Yard Runs - Big plays',
    '30+ YDS': '30+ Yard Runs - Deep plays',
    '40+ YDS': '40+ Yard Runs - Very deep plays',
    '50+ YDS': '50+ Yard Runs - Extremely deep plays',
    'REC': 'Receptions - Receiving volume',
    'TGT': 'Targets - Receiving opportunity',
    'RZ TGT': 'Red Zone Targets - TD opportunity',
    
    # WR/TE Metrics
    'REC': 'Receptions - Total catches',
    'YDS': 'Receiving Yards - Total receiving production',
    'Y/R': 'Yards per Reception - Big play ability',
    'YBC': 'Yards Before Catch - Route running/separation',
    'YBC/R': 'Yards Before Catch per Reception - Route efficiency',
    'AIR': 'Air Yards - Downfield target depth',
    'AIR/R': 'Air Yards per Reception - Downfield efficiency',
    'YAC': 'Yards After Catch - Playmaking ability',
    'YAC/R': 'Yards After Catch per Reception - Playmaking efficiency',
    'YACON': 'Yards After Contact - Power after catch',
    'YACON/R': 'Yards After Contact per Reception - Power efficiency',
    'BRKTKL': 'Broken Tackles - Elusiveness after catch',
    'TGT': 'Targets - Receiving opportunity',
    '% TM': 'Team Target Share - Volume opportunity',
    'CATCHABLE': 'Catchable Targets - QB accuracy to player',
    'DROP': 'Drops - Missed catchable passes',
    'RZ TGT': 'Red Zone Targets - TD opportunity',
    '10+ YDS': '10+ Yard Receptions - Explosive plays',
    '20+ YDS': '20+ Yard Receptions - Big plays',
    '30+ YDS': '30+ Yard Receptions - Deep plays',
    '40+ YDS': '40+ Yard Receptions - Very deep plays',
    '50+ YDS': '50+ Yard Receptions - Extremely deep plays',
    'LNG': 'Longest Reception - Best single play'
}

# Metric Categories for Universal Scoring
METRIC_CATEGORIES = {
    'volume': ['YDS', 'ATT', 'REC', 'TGT', 'TD', 'COMP', 'ATT'],
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

# Positional Value Analysis Constants
POSITIONAL_ANALYSIS_CONSTANTS = {
    'min_players_per_position': 5,
    'min_players_per_round': 3,
    'default_league_size': 12,
    'analysis_rounds': 4,  # Analyze first 4 rounds
    'position_tiers': {
        'QB': [(1, 5), (6, 12), (13, 24)],
        'RB': [(1, 5), (6, 12), (13, 24), (25, 36)],
        'WR': [(1, 5), (6, 12), (13, 24), (25, 36)],
        'TE': [(1, 3), (4, 8), (9, 16)]
    }
}

# Default Values
DEFAULT_SCORE = 0.0
DEFAULT_WEIGHT = 1.0
DEFAULT_CONFIDENCE = 0.5 