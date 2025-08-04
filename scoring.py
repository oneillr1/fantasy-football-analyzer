import requests
import json
import time
from datetime import datetime
from typing import Dict, Any

class SleeperScoringExtractor:
    def __init__(self, league_id: str):
        self.league_id = league_id
        self.base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 100ms between requests
        
    def _make_request(self, endpoint: str) -> Any:
        """Make API request with rate limiting"""
        time.sleep(self.rate_limit_delay)
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def get_league_info(self) -> Dict:
        """Get complete league information including scoring settings"""
        print(f"Fetching league info for {self.league_id}...")
        return self._make_request(f"/league/{self.league_id}")
    
    def format_scoring_settings(self, scoring_settings: Dict) -> Dict:
        """Format scoring settings with readable descriptions"""
        
        # Common scoring categories with descriptions
        scoring_descriptions = {
            # Passing
            'pass_yd': 'Passing Yards',
            'pass_td': 'Passing Touchdowns',
            'pass_int': 'Passing Interceptions',
            'pass_2pt': 'Passing 2-Point Conversions',
            'pass_cmp': 'Passing Completions',
            'pass_inc': 'Passing Incompletions',
            'pass_cmp_40p': 'Passing Completions 40+ yards',
            'pass_td_40p': 'Passing TDs 40+ yards',
            'pass_td_50p': 'Passing TDs 50+ yards',
            'pass_sack': 'Times Sacked',
            
            # Rushing  
            'rush_yd': 'Rushing Yards',
            'rush_td': 'Rushing Touchdowns',
            'rush_2pt': 'Rushing 2-Point Conversions',
            'rush_att': 'Rushing Attempts',
            'rush_40p': 'Rushing 40+ yards',
            'rush_td_40p': 'Rushing TDs 40+ yards',
            'rush_td_50p': 'Rushing TDs 50+ yards',
            
            # Receiving
            'rec': 'Receptions (PPR)',
            'rec_yd': 'Receiving Yards', 
            'rec_td': 'Receiving Touchdowns',
            'rec_2pt': 'Receiving 2-Point Conversions',
            'rec_40p': 'Receiving 40+ yards',
            'rec_td_40p': 'Receiving TDs 40+ yards',
            'rec_td_50p': 'Receiving TDs 50+ yards',
            'rec_0_4': 'Receptions 0-4 yards',
            'rec_5_9': 'Receptions 5-9 yards',
            'rec_10_19': 'Receptions 10-19 yards',
            'rec_20_29': 'Receptions 20-29 yards',
            'rec_30_39': 'Receptions 30-39 yards',
            'rec_40p': 'Receptions 40+ yards',
            
            # Kicking
            'fgm': 'Field Goals Made',
            'fgm_0_19': 'Field Goals 0-19 yards',
            'fgm_20_29': 'Field Goals 20-29 yards', 
            'fgm_30_39': 'Field Goals 30-39 yards',
            'fgm_40_49': 'Field Goals 40-49 yards',
            'fgm_50p': 'Field Goals 50+ yards',
            'fgmiss': 'Field Goals Missed',
            'xpm': 'Extra Points Made',
            'xpmiss': 'Extra Points Missed',
            
            # Defense/Special Teams
            'def_td': 'Defensive/ST Touchdowns',
            'def_int': 'Defensive Interceptions',
            'def_fr': 'Fumbles Recovered',
            'def_sack': 'Sacks by Defense',
            'def_safety': 'Safeties',
            'def_block_kick': 'Blocked Kicks',
            'def_pass_def': 'Passes Defended',
            'def_tkl': 'Tackles',
            'def_tkl_loss': 'Tackles for Loss',
            'def_qb_hit': 'QB Hits',
            'def_tkl_solo': 'Solo Tackles',
            'def_tkl_ast': 'Assisted Tackles',
            'def_pts_allowed': 'Points Allowed',
            'def_yds_allowed': 'Yards Allowed',
            
            # Points allowed brackets
            'def_st_td': 'Special Teams TD',
            'def_st_ff': 'Special Teams Forced Fumble',
            'def_st_fum_rec': 'Special Teams Fumble Recovery',
            'def_pr_td': 'Punt Return TD',
            'def_kr_td': 'Kick Return TD',
            
            # Fumbles
            'fum': 'Fumbles',
            'fum_lost': 'Fumbles Lost',
            'fum_rec': 'Fumbles Recovered',
            'fum_rec_td': 'Fumble Recovery TD',
            
            # Bonus points
            'bonus_rec_yd': 'Bonus Receiving Yards',
            'bonus_rush_yd': 'Bonus Rushing Yards',
            'bonus_pass_yd': 'Bonus Passing Yards',
            'bonus_rec_td': 'Bonus Receiving TD',
            'bonus_rush_td': 'Bonus Rushing TD',
            'bonus_pass_td': 'Bonus Passing TD',
            
            # IDP (Individual Defensive Players)
            'idp_tkl': 'IDP Tackles',
            'idp_ast': 'IDP Assisted Tackles',
            'idp_sack': 'IDP Sacks',
            'idp_int': 'IDP Interceptions',
            'idp_fum_rec': 'IDP Fumble Recoveries',
            'idp_def_td': 'IDP Defensive TDs',
            'idp_safety': 'IDP Safeties',
            'idp_blk_kick': 'IDP Blocked Kicks',
            'idp_tkl_loss': 'IDP Tackles for Loss',
            'idp_qb_hit': 'IDP QB Hits',
            'idp_int_td': 'IDP Interception TDs',
            'idp_fum_rec_td': 'IDP Fumble Recovery TDs',
            'idp_blk_kick_td': 'IDP Blocked Kick TDs'
        }
        
        formatted_scoring = {}
        for key, value in scoring_settings.items():
            description = scoring_descriptions.get(key, key.replace('_', ' ').title())
            formatted_scoring[key] = {
                'points': value,
                'description': description
            }
        
        return formatted_scoring
    
    def analyze_scoring_type(self, scoring_settings: Dict) -> Dict:
        """Analyze what type of scoring system this is"""
        analysis = {
            'scoring_type': 'Standard',
            'ppr_value': 0,
            'special_features': [],
            'position_emphasis': {}
        }
        
        # Check for PPR
        if 'rec' in scoring_settings:
            ppr_val = scoring_settings['rec']
            analysis['ppr_value'] = ppr_val
            if ppr_val == 1:
                analysis['scoring_type'] = 'Full PPR'
            elif ppr_val == 0.5:
                analysis['scoring_type'] = 'Half PPR'
            elif ppr_val > 0:
                analysis['scoring_type'] = f'{ppr_val} PPR'
        
        # Check for bonus scoring
        bonus_categories = [k for k in scoring_settings.keys() if 'bonus' in k]
        if bonus_categories:
            analysis['special_features'].append('Bonus Scoring')
        
        # Check for IDP
        idp_categories = [k for k in scoring_settings.keys() if 'idp' in k or 'def_tkl' in k]
        if idp_categories:
            analysis['special_features'].append('Individual Defensive Players (IDP)')
        
        # Check for premium positions
        if scoring_settings.get('pass_td', 0) > 4:
            analysis['special_features'].append('Premium QB Scoring')
        
        if scoring_settings.get('rec_td', 0) != scoring_settings.get('rush_td', 0):
            analysis['special_features'].append('Position-Specific TD Scoring')
        
        # Analyze position emphasis
        qb_emphasis = scoring_settings.get('pass_yd', 0) + (scoring_settings.get('pass_td', 0) * 10)
        rb_emphasis = scoring_settings.get('rush_yd', 0) + (scoring_settings.get('rush_td', 0) * 10)
        wr_emphasis = scoring_settings.get('rec_yd', 0) + (scoring_settings.get('rec_td', 0) * 10) + scoring_settings.get('rec', 0)
        
        analysis['position_emphasis'] = {
            'qb_relative_value': qb_emphasis,
            'rb_relative_value': rb_emphasis, 
            'wr_relative_value': wr_emphasis
        }
        
        return analysis
    
    def extract_scoring_settings(self) -> Dict:
        """Extract and format all scoring-related information"""
        print("Extracting scoring settings...")
        
        # Get league info
        league_info = self.get_league_info()
        if not league_info:
            print("Failed to get league info")
            return {}
        
        # Extract scoring settings
        scoring_settings = league_info.get('scoring_settings', {})
        roster_positions = league_info.get('roster_positions', [])
        league_settings = league_info.get('settings', {})
        
        # Format the data
        formatted_scoring = self.format_scoring_settings(scoring_settings)
        scoring_analysis = self.analyze_scoring_type(scoring_settings)
        
        return {
            'league_basic_info': {
                'league_id': league_info.get('league_id'),
                'name': league_info.get('name'),
                'season': league_info.get('season'),
                'total_rosters': league_info.get('total_rosters'),
                'status': league_info.get('status')
            },
            'roster_settings': {
                'roster_positions': roster_positions,
                'total_roster_spots': len(roster_positions),
                'starting_spots': len([pos for pos in roster_positions if pos != 'BN']),
                'bench_spots': roster_positions.count('BN'),
                'flex_spots': roster_positions.count('FLEX'),
                'superflex_spots': roster_positions.count('SUPER_FLEX')
            },
            'scoring_settings': {
                'raw_settings': scoring_settings,
                'formatted_settings': formatted_scoring,
                'total_categories': len(scoring_settings)
            },
            'scoring_analysis': scoring_analysis,
            'league_settings': {
                'playoff_teams': league_settings.get('playoff_teams'),
                'playoff_weeks': league_settings.get('playoff_week_start'),
                'trade_deadline': league_settings.get('trade_deadline'),
                'waiver_type': league_settings.get('waiver_type'),
                'faab_budget': league_settings.get('waiver_budget'),
                'taxi_slots': league_settings.get('taxi_slots'),
                'reserve_slots': league_settings.get('reserve_slots')
            }
        }
    
    def export_scoring_settings(self, scoring_data: Dict):
        """Export scoring settings to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export comprehensive JSON
        json_filename = f"sleeper_scoring_settings_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(scoring_data, f, indent=2)
        print(f"Exported scoring settings to {json_filename}")
        
        # Export readable text summary
        txt_filename = f"scoring_summary_{timestamp}.txt"
        with open(txt_filename, 'w') as f:
            f.write(f"FANTASY FOOTBALL LEAGUE SCORING SETTINGS\n")
            f.write(f"{'='*50}\n\n")
            
            # Basic info
            basic = scoring_data['league_basic_info']
            f.write(f"League: {basic['name']}\n")
            f.write(f"Season: {basic['season']}\n")
            f.write(f"Teams: {basic['total_rosters']}\n")
            f.write(f"League ID: {basic['league_id']}\n\n")
            
            # Scoring type
            analysis = scoring_data['scoring_analysis']
            f.write(f"SCORING TYPE: {analysis['scoring_type']}\n")
            if analysis['special_features']:
                f.write(f"Special Features: {', '.join(analysis['special_features'])}\n")
            f.write(f"\n")
            
            # Roster settings
            roster = scoring_data['roster_settings']
            f.write(f"ROSTER SETTINGS:\n")
            f.write(f"Starting Spots: {roster['starting_spots']}\n")
            f.write(f"Bench Spots: {roster['bench_spots']}\n")
            f.write(f"Flex Spots: {roster['flex_spots']}\n")
            if roster['superflex_spots'] > 0:
                f.write(f"SuperFlex Spots: {roster['superflex_spots']}\n")
            f.write(f"Roster Positions: {', '.join(roster['roster_positions'])}\n\n")
            
            # Detailed scoring
            f.write(f"DETAILED SCORING SETTINGS:\n")
            f.write(f"{'-'*30}\n")
            
            formatted = scoring_data['scoring_settings']['formatted_settings']
            
            # Group by category
            categories = {
                'Passing': ['pass_'],
                'Rushing': ['rush_'],
                'Receiving': ['rec_'],
                'Kicking': ['fgm', 'xp'],
                'Defense/ST': ['def_'],
                'Fumbles': ['fum'],
                'Bonus': ['bonus_'],
                'IDP': ['idp_']
            }
            
            for category, prefixes in categories.items():
                category_settings = {k: v for k, v in formatted.items() 
                                   if any(k.startswith(prefix) for prefix in prefixes)}
                
                if category_settings:
                    f.write(f"\n{category.upper()}:\n")
                    for key, data in sorted(category_settings.items()):
                        f.write(f"  {data['description']}: {data['points']} pts\n")
            
            # Other settings
            other_settings = {k: v for k, v in formatted.items() 
                            if not any(k.startswith(prefix) for prefixes in categories.values() for prefix in prefixes)}
            
            if other_settings:
                f.write(f"\nOTHER SCORING:\n")
                for key, data in sorted(other_settings.items()):
                    f.write(f"  {data['description']}: {data['points']} pts\n")
        
        print(f"Exported readable summary to {txt_filename}")
        
        return json_filename, txt_filename

def main():
    # Your league ID
    league_id = "1124821239395807232"
    
    # Initialize extractor
    extractor = SleeperScoringExtractor(league_id)
    
    # Extract scoring settings
    scoring_data = extractor.extract_scoring_settings()
    
    if not scoring_data:
        print("Failed to extract scoring settings")
        return
    
    # Export to files
    json_file, txt_file = extractor.export_scoring_settings(scoring_data)
    
    print("\n" + "="*50)
    print("SCORING SETTINGS EXTRACTION COMPLETE!")
    print("="*50)
    
    basic = scoring_data['league_basic_info']
    analysis = scoring_data['scoring_analysis']
    roster = scoring_data['roster_settings']
    
    print(f"League: {basic['name']}")
    print(f"Season: {basic['season']}")
    print(f"Scoring Type: {analysis['scoring_type']}")
    print(f"Total Scoring Categories: {scoring_data['scoring_settings']['total_categories']}")
    print(f"Starting Lineup Spots: {roster['starting_spots']}")
    print(f"Bench Spots: {roster['bench_spots']}")
    
    if analysis['special_features']:
        print(f"Special Features: {', '.join(analysis['special_features'])}")
    
    print(f"\nFiles exported:")
    print(f"  - {json_file} (detailed JSON data)")
    print(f"  - {txt_file} (readable summary)")
    
    print(f"\nUse these files to train your GPT on your league's specific scoring system!")

if __name__ == "__main__":
    main()