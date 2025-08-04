import requests
import json
import time
from datetime import datetime
from typing import Dict, List, Any
import csv

class SleeperDataExtractor:
    def __init__(self, league_id: str):
        self.league_id = league_id
        self.base_url = "https://api.sleeper.app/v1"
        self.session = requests.Session()
        self.rate_limit_delay = 0.1  # 100ms between requests to stay under 1000/minute
        
        # Cache for players data
        self.players_data = None
        
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
    
    def get_players_data(self) -> Dict:
        """Fetch all players data (call once per day max)"""
        if self.players_data is None:
            print("Fetching players data...")
            self.players_data = self._make_request("/players/nfl")
        return self.players_data or {}
    
    def get_league_info(self) -> Dict:
        """Get basic league information"""
        print(f"Fetching league info for {self.league_id}...")
        return self._make_request(f"/league/{self.league_id}")
    
    def get_league_users(self) -> List[Dict]:
        """Get all users in the league"""
        print("Fetching league users...")
        return self._make_request(f"/league/{self.league_id}/users") or []
    
    def get_all_drafts(self) -> List[Dict]:
        """Get all drafts for the league"""
        print("Fetching all drafts...")
        return self._make_request(f"/league/{self.league_id}/drafts") or []
    
    def get_draft_picks(self, draft_id: str) -> List[Dict]:
        """Get all picks for a specific draft"""
        print(f"Fetching picks for draft {draft_id}...")
        return self._make_request(f"/draft/{draft_id}/picks") or []
    
    def get_draft_details(self, draft_id: str) -> Dict:
        """Get detailed draft information"""
        print(f"Fetching draft details for {draft_id}...")
        return self._make_request(f"/draft/{draft_id}")
    
    def get_rosters(self) -> List[Dict]:
        """Get current rosters"""
        print("Fetching current rosters...")
        return self._make_request(f"/league/{self.league_id}/rosters") or []
    
    def get_transactions(self, week: int = 1) -> List[Dict]:
        """Get transactions for a specific week"""
        print(f"Fetching transactions for week {week}...")
        return self._make_request(f"/league/{self.league_id}/transactions/{week}") or []
    
    def get_traded_picks(self) -> List[Dict]:
        """Get all traded picks in the league"""
        print("Fetching traded picks...")
        return self._make_request(f"/league/{self.league_id}/traded_picks") or []
    
    def get_historical_league_ids(self) -> List[str]:
        """Get all historical league IDs by following previous_league_id chain"""
        print("Tracing league history...")
        league_ids = []
        current_league_id = self.league_id
        
        # Follow the chain backwards through previous_league_id
        while current_league_id:
            league_ids.append(current_league_id)
            league_info = self._make_request(f"/league/{current_league_id}")
            if not league_info:
                break
            current_league_id = league_info.get('previous_league_id')
            if current_league_id in league_ids:  # Prevent infinite loops
                break
        
        print(f"Found {len(league_ids)} leagues in history: {league_ids}")
        return league_ids
    
    def extract_all_drafts_data(self) -> Dict:
        """Extract comprehensive draft data for all league drafts across all seasons"""
        print("Starting comprehensive draft data extraction...")
        
        # Get all historical league IDs
        all_league_ids = self.get_historical_league_ids()
        
        # Get players data for player name lookups
        players = self.get_players_data()
        
        # Initialize data structure
        draft_data = {
            'league_info': {},
            'all_users': {},  # Combined users from all seasons
            'drafts': []
        }
        
        # Process each league (season)
        for league_id in all_league_ids:
            print(f"\nProcessing league {league_id}...")
            
            # Get league info
            league_info = self._make_request(f"/league/{league_id}")
            if not league_info:
                print(f"Failed to get info for league {league_id}")
                continue
            
            # Store league info (use most recent as primary)
            if not draft_data['league_info']:
                draft_data['league_info'] = {
                    'current_league_id': self.league_id,
                    'name': league_info.get('name'),
                    'current_season': league_info.get('season'),
                    'total_rosters': league_info.get('total_rosters'),
                    'scoring_settings': league_info.get('scoring_settings'),
                    'roster_positions': league_info.get('roster_positions'),
                    'settings': league_info.get('settings')
                }
            
            # Get users for this league
            users = self._make_request(f"/league/{league_id}/users") or []
            season_user_map = {user['user_id']: user for user in users}
            
            # Add to combined user map
            draft_data['all_users'].update(season_user_map)
            
            print(f"Found {len(users)} users in {league_info.get('season', 'unknown')} season")
            
            # Get all drafts for this league
            drafts = self._make_request(f"/league/{league_id}/drafts") or []
            print(f"Found {len(drafts)} drafts for league {league_id}")
            
            # Process each draft
            for draft in drafts:
                draft_id = draft['draft_id']
                print(f"Processing draft {draft_id} from {draft.get('season', 'unknown')}...")
                
                # Get detailed draft info
                draft_details = self._make_request(f"/draft/{draft_id}")
                if not draft_details:
                    print(f"Failed to get details for draft {draft_id}")
                    continue
                
                # Get all picks for this draft
                picks = self._make_request(f"/draft/{draft_id}/picks") or []
                if not picks:
                    print(f"No picks found for draft {draft_id}")
                    continue
                
                print(f"Found {len(picks)} picks in draft {draft_id}")
                
                # Create roster_id to user mapping for this draft
                roster_to_user = {}
                if 'slot_to_roster_id' in draft_details and 'draft_order' in draft_details:
                    for user_id, slot in draft_details['draft_order'].items():
                        roster_id = draft_details['slot_to_roster_id'].get(str(slot))
                        if roster_id:
                            roster_to_user[str(roster_id)] = user_id
                
                # Enhance picks with player names and user info
                enhanced_picks = []
                for pick in picks:
                    player_id = pick.get('player_id')
                    player_info = players.get(player_id, {}) if players and player_id else {}
                    
                    # Get user info - try multiple methods
                    picked_by_user_id = pick.get('picked_by')
                    roster_id = str(pick.get('roster_id', ''))
                    
                    # If picked_by is empty, try to get from roster mapping
                    if not picked_by_user_id and roster_id in roster_to_user:
                        picked_by_user_id = roster_to_user[roster_id]
                    
                    # Get user info from combined user map
                    user_info = draft_data['all_users'].get(picked_by_user_id, {})
                    
                    # Prioritize display_name over username since username might be empty
                    username = user_info.get('username') or user_info.get('display_name', 'Unknown')
                    display_name = user_info.get('display_name') or user_info.get('username', 'Unknown')
                    
                    enhanced_pick = {
                        'pick_no': pick.get('pick_no'),
                        'round': pick.get('round'),
                        'draft_slot': pick.get('draft_slot'),
                        'player_id': player_id,
                        'player_name': f"{player_info.get('first_name', '')} {player_info.get('last_name', '')}".strip() or 'Unknown Player',
                        'position': player_info.get('position'),
                        'team': player_info.get('team'),
                        'picked_by_user_id': picked_by_user_id,
                        'picked_by_username': username,
                        'picked_by_display_name': display_name,
                        'roster_id': pick.get('roster_id'),
                        'is_keeper': pick.get('is_keeper'),
                        'metadata': pick.get('metadata', {})
                    }
                    enhanced_picks.append(enhanced_pick)
                
                # Sort picks by pick number
                enhanced_picks.sort(key=lambda x: x['pick_no'] or 0)
                
                # Organize picks by user for analysis
                picks_by_user = {}
                picks_by_username = {}
                picks_by_display_name = {}
                
                for pick in enhanced_picks:
                    user_id = pick['picked_by_user_id']
                    username = pick['picked_by_username'] 
                    display_name = pick['picked_by_display_name']
                    
                    # Group by user ID
                    if user_id and user_id != 'Unknown':
                        if user_id not in picks_by_user:
                            picks_by_user[user_id] = []
                        picks_by_user[user_id].append(pick)
                    
                    # Group by username (which now prioritizes display_name)
                    if username and username != 'Unknown':
                        if username not in picks_by_username:
                            picks_by_username[username] = []
                        picks_by_username[username].append(pick)
                    
                    # Also group by display_name as backup
                    if display_name and display_name != 'Unknown':
                        if display_name not in picks_by_display_name:
                            picks_by_display_name[display_name] = []
                        picks_by_display_name[display_name].append(pick)
                
                draft_summary = {
                    'draft_id': draft_id,
                    'league_id': league_id,
                    'season': draft.get('season'),
                    'type': draft.get('type'),  # snake, linear, etc.
                    'status': draft.get('status'),
                    'start_time': draft.get('start_time'),
                    'settings': draft_details.get('settings', {}),
                    'metadata': draft.get('metadata', {}),
                    'draft_order': draft_details.get('draft_order', {}),
                    'slot_to_roster_id': draft_details.get('slot_to_roster_id', {}),
                    'roster_to_user_mapping': roster_to_user,
                    'total_picks': len(enhanced_picks),
                    'all_picks': enhanced_picks,
                    'picks_by_user': picks_by_user,
                    'picks_by_username': picks_by_username,
                    'picks_by_display_name': picks_by_display_name
                }
                
                draft_data['drafts'].append(draft_summary)
                print(f"Successfully processed draft {draft_id} with {len(enhanced_picks)} picks")
        
        # Sort drafts by season (most recent first)
        draft_data['drafts'].sort(key=lambda x: x.get('season', ''), reverse=True)
        
        return draft_data
    
    def analyze_draft_tendencies(self, draft_data: Dict) -> Dict:
        """Analyze draft tendencies for each user across all seasons"""
        print("Analyzing draft tendencies across all seasons...")
        
        analysis = {
            'user_tendencies': {},
            'positional_analysis': {},
            'draft_patterns': {},
            'season_summary': {}
        }
        
        # Process all drafts
        for draft in draft_data.get('drafts', []):
            season = draft.get('season', 'Unknown')
            
            # Track season summary
            if season not in analysis['season_summary']:
                analysis['season_summary'][season] = {
                    'total_picks': 0,
                    'draft_type': draft.get('type'),
                    'participants': set()
                }
            
            # Use picks_by_username (which now prioritizes display_name) 
            # but fall back to picks_by_display_name if needed
            picks_by_username = draft.get('picks_by_username', {})
            if not picks_by_username:  # Fallback to display_name grouping
                picks_by_username = draft.get('picks_by_display_name', {})
            
            for username, picks in picks_by_username.items():
                if username == 'Unknown':
                    continue
                    
                analysis['season_summary'][season]['participants'].add(username)
                analysis['season_summary'][season]['total_picks'] += len(picks)
                
                if username not in analysis['user_tendencies']:
                    analysis['user_tendencies'][username] = {
                        'total_picks': 0,
                        'seasons_participated': set(),
                        'positions_drafted': {},
                        'early_round_preferences': {},  # rounds 1-3
                        'middle_round_preferences': {},  # rounds 4-8
                        'late_round_preferences': {},    # rounds 9+
                        'average_position_by_round': {},
                        'team_preferences': {},
                        'draft_history': []
                    }
                
                user_analysis = analysis['user_tendencies'][username]
                user_analysis['total_picks'] += len(picks)
                user_analysis['seasons_participated'].add(season)
                
                # Track this draft for the user
                draft_summary = {
                    'season': season,
                    'draft_id': draft['draft_id'],
                    'picks': []
                }
                
                for pick in picks:
                    position = pick.get('position', 'Unknown')
                    round_num = pick.get('round', 0)
                    team = pick.get('team', 'Unknown')
                    player_name = pick.get('player_name', 'Unknown')
                    
                    # Add to draft history
                    draft_summary['picks'].append({
                        'round': round_num,
                        'pick_no': pick.get('pick_no'),
                        'player_name': player_name,
                        'position': position,
                        'team': team
                    })
                    
                    # Position counting
                    user_analysis['positions_drafted'][position] = user_analysis['positions_drafted'].get(position, 0) + 1
                    
                    # Team preferences
                    if team and team != 'Unknown':
                        user_analysis['team_preferences'][team] = user_analysis['team_preferences'].get(team, 0) + 1
                    
                    # Round-based preferences
                    if 1 <= round_num <= 3:
                        user_analysis['early_round_preferences'][position] = user_analysis['early_round_preferences'].get(position, 0) + 1
                    elif 4 <= round_num <= 8:
                        user_analysis['middle_round_preferences'][position] = user_analysis['middle_round_preferences'].get(position, 0) + 1
                    elif round_num >= 9:
                        user_analysis['late_round_preferences'][position] = user_analysis['late_round_preferences'].get(position, 0) + 1
                
                user_analysis['draft_history'].append(draft_summary)
        
        # Convert sets to lists for JSON serialization
        for username, data in analysis['user_tendencies'].items():
            data['seasons_participated'] = list(data['seasons_participated'])
        
        for season, data in analysis['season_summary'].items():
            data['participants'] = list(data['participants'])
        
        return analysis
    
    def export_to_files(self, draft_data: Dict, analysis: Dict):
        """Export data to multiple formats for GPT training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export comprehensive JSON
        json_filename = f"sleeper_draft_data_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump({
                'draft_data': draft_data,
                'analysis': analysis,
                'export_timestamp': timestamp
            }, f, indent=2)
        print(f"Exported comprehensive data to {json_filename}")
        
        # Export user tendencies CSV
        csv_filename = f"user_draft_tendencies_{timestamp}.csv"
        with open(csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Username', 'Total_Picks', 'Seasons_Participated', 'Top_Position', 'Early_Round_Preference', 
                'Middle_Round_Preference', 'Late_Round_Preference', 'Favorite_Team'
            ])
            
            for username, data in analysis['user_tendencies'].items():
                top_position = max(data['positions_drafted'], key=data['positions_drafted'].get) if data['positions_drafted'] else 'None'
                early_pref = max(data['early_round_preferences'], key=data['early_round_preferences'].get) if data['early_round_preferences'] else 'None'
                middle_pref = max(data['middle_round_preferences'], key=data['middle_round_preferences'].get) if data['middle_round_preferences'] else 'None'
                late_pref = max(data['late_round_preferences'], key=data['late_round_preferences'].get) if data['late_round_preferences'] else 'None'
                fav_team = max(data['team_preferences'], key=data['team_preferences'].get) if data['team_preferences'] else 'None'
                seasons = ', '.join(sorted(data['seasons_participated']))
                
                writer.writerow([
                    username, data['total_picks'], seasons, top_position, early_pref, 
                    middle_pref, late_pref, fav_team
                ])
        print(f"Exported user tendencies to {csv_filename}")
        
        # Export draft picks summary CSV
        picks_filename = f"all_draft_picks_{timestamp}.csv"
        with open(picks_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Draft_ID', 'Season', 'Pick_No', 'Round', 'Draft_Slot', 
                'Player_Name', 'Position', 'Team', 'Picked_By_Username', 'Is_Keeper'
            ])
            
            for draft in draft_data['drafts']:
                for pick in draft['all_picks']:
                    writer.writerow([
                        draft['draft_id'], draft['season'], pick['pick_no'], 
                        pick['round'], pick['draft_slot'], pick['player_name'],
                        pick['position'], pick['team'], pick['picked_by_username'],
                        pick['is_keeper']
                    ])
        print(f"Exported all picks to {picks_filename}")
        
        # Export season summary CSV
        season_filename = f"season_summary_{timestamp}.csv"
        with open(season_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Season', 'Total_Picks', 'Draft_Type', 'Participants', 'Participant_Count'])
            
            for season, data in analysis['season_summary'].items():
                participants_str = ', '.join(sorted(data['participants']))
                writer.writerow([
                    season, data['total_picks'], data['draft_type'], 
                    participants_str, len(data['participants'])
                ])
        print(f"Exported season summary to {season_filename}")
        
        return json_filename, csv_filename, picks_filename, season_filename

def main():
    # Your league ID from the URL
    league_id = "1124821239395807232"
    
    # Initialize extractor
    extractor = SleeperDataExtractor(league_id)
    
    # Extract all draft data
    draft_data = extractor.extract_all_drafts_data()
    
    if not draft_data:
        print("Failed to extract draft data")
        return
    
    # Analyze draft tendencies
    analysis = extractor.analyze_draft_tendencies(draft_data)
    
    # Export to files
    json_file, csv_file, picks_file, season_file = extractor.export_to_files(draft_data, analysis)
    
    print("\n" + "="*50)
    print("EXTRACTION COMPLETE!")
    print("="*50)
    print(f"League: {draft_data['league_info']['name']}")
    print(f"Current Season: {draft_data['league_info']['current_season']}")
    print(f"Total Drafts: {len(draft_data['drafts'])}")
    print(f"Total Unique Users: {len(draft_data['all_users'])}")
    
    # Show drafts by season
    drafts_by_season = {}
    for draft in draft_data['drafts']:
        season = draft['season']
        if season not in drafts_by_season:
            drafts_by_season[season] = []
        drafts_by_season[season].append(draft)
    
    print(f"\nDrafts by Season:")
    for season in sorted(drafts_by_season.keys(), reverse=True):
        print(f"  {season}: {len(drafts_by_season[season])} draft(s)")
        for draft in drafts_by_season[season]:
            print(f"    - Draft {draft['draft_id']}: {draft['total_picks']} picks ({draft['type']} draft)")
    
    print(f"\nUser Participation Summary:")
    for username, data in analysis['user_tendencies'].items():
        seasons = ', '.join(sorted(data['seasons_participated']))
        print(f"  {username}: {data['total_picks']} total picks across seasons {seasons}")
    
    print(f"\nFiles exported:")
    print(f"  - {json_file} (comprehensive data)")
    print(f"  - {csv_file} (user tendencies)")
    print(f"  - {picks_file} (all draft picks)")
    print(f"  - {season_file} (season summary)")
    
    print("\n" + "="*50)
    print("ADDITIONAL API ENDPOINTS TO CONSIDER:")
    print("="*50)
    print("1. League Transactions: extractor.get_transactions(week)")
    print("2. Current Rosters: extractor.get_rosters()")
    print("3. Traded Picks: extractor.get_traded_picks()")
    print("4. League Matchups: Add matchup analysis for performance patterns")
    print("5. Player Performance: Cross-reference with actual season performance")

if __name__ == "__main__":
    main()