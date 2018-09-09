import requests
import sqlite3
import datetime
import operator
import random
from functools import reduce
from bs4 import BeautifulSoup
import re
import traceback
import time

sleep_time = .1
game_search_base_url = 'https://www.basketball-reference.com/boxscores/index.cgi?month={1}&day={2}&year={0}'
game_base_url = 'https://www.basketball-reference.com{0}'
player_base_url = 'https://www.basketball-reference.com'
path = r'C:\Users\trist\Documents\db_loc\nba/'


def make_request(url):
    max_tries = 3
    time.sleep(sleep_time)
    for i in range(max_tries):
        try:
            s = requests.Session()
            return s.get(url)
        except:
            traceback.print_exc()
            time.sleep(30)

def build_db():
    with sqlite3.connect(path + 'nba.db') as conn:
        conn.execute('create table if not exists player (player_id TEXT UNIQUE, dob date)')
        conn.execute('create table if not exists game (g_id text unique, game_date date, location text)')
        conn.execute('create table if not exists team_game (g_id text, team_name text, date_played date, result int, score int, game_location text, game_type text)')#game from 1 teams percepective
        conn.execute('''create table if not exists player_game_contribution (player_id text, g_id text, team_name text, opponent_name text, date_played date, minutes_played int, fg int, fga int,
                     threeP int, threePA int, FT int, FTA int, ORB int, DRB int, AST int, STL int, BLK int, TOV int, PF int, PTS int, plus_minus int,
                     ts_perc float, three_p_ar float, ftr float, ODR_perc float, DBR_perc float, AST_perc float, STL_perc float, BLK_perc float,
                     TOV_perc flaot, USG_perc float, ortg int, drtg int, result int, home_game int)''')
        conn.execute('''create table if not exists team_total_performance (g_id text, team_name text, opponent_name text, date_played date, minutes_played int, fg int, fga int,
                     threeP int, threePA int, FT int, FTA int, ORB int, DRB int, AST int, STL int, BLK int, TOV int, PF int, PTS int, plus_minus int,
                     ts_perc float, three_p_ar float, ftr float, ODR_perc float, DBR_perc float, AST_perc float, STL_perc float, BLK_perc float,
                     TOV_perc flaot, USG_perc float, ortg int, drtg int, result int, home_game int)''')
        conn.commit()

def get_game_urls_at_date(input_date):
    try:
        year, month, day = input_date.year, input_date.month, input_date.day
        game_urls = set()
        r = make_request(game_search_base_url.format(year, month, day))
        if r is None:
            return set()
        soup = BeautifulSoup(r.text, 'html.parser')
        game_summaries = soup.find('div', {'class':'game_summaries'})
        if game_summaries is not None:
            for i in game_summaries.find_all('div', {'class':'game_summary expanded nohover'}):
                link_tag = i.find('p', recursive = False)
                if link_tag is not None and link_tag.find('a') is not None:
                    game_urls.add(link_tag.find('a')['href'])

        print('{0} game urls recorded at date {1}'.format(len(game_urls), input_date))
        return game_urls
    except:
        traceback.print_exc()
        return set()

def get_date_range():
    d1 = datetime.date(2016, 1, 1)
    # d2 = datetime.date(2016, 1, 12)
    d2 = datetime.datetime.now().date()
    dates = [d1 + datetime.timedelta(days=x) for x in range((d2 - d1).days + 1)]
    random.shuffle(dates)
    return dates

def convert_to_float(input_str):
    try:
        return float(input_str)
    except:
        return None

def read_game_info(game_url):
    try:
        print(game_url)
        player_urls = set()
        r = make_request(game_base_url.format(game_url))
        if r is None:
            return set()
        #game table
        soup = BeautifulSoup(r.text, 'html.parser')
        footer = soup.find('div', {'id':"footer"})
        game_date_span = footer.find('span', {'itemprop':'itemListElement'})
        url = game_date_span.find('a')['href']
        date_parts = [int(i) for i in re.findall(r'\d+', url)]
        game_date = datetime.date(date_parts[2], date_parts[0], date_parts[1])

        #team game,
        location = soup.find('div', {'class':"scorebox_meta"}).find_all('div')[1].text

        ff_table = soup.find('div',{'id':'all_line_score'})
        score_table = soup.find('div',{'id':'all_line_score'})
        score_table_edited = BeautifulSoup(str(score_table).replace('-->', '').replace('<!--', ''), 'html.parser')
        team_1_score_info = score_table_edited.find_all('tr')[2]
        team_2_score_info = score_table_edited.find_all('tr')[3]

        team_1_id = team_1_score_info.find('a')['href'].split('/')[:-1][-1]
        team_2_id = team_2_score_info.find('a')['href'].split('/')[:-1][-1]

        team_1_score = int(team_1_score_info.find_all('td')[-1].text)
        team_2_score = int(team_2_score_info.find_all('td')[-1].text)

        #player_info
        basic_info_tables = soup.find_all('div', {'id':re.compile(r'all_box_[a-zA-Z]{3}_basic')})
        advanced_info_tables = soup.find_all('div', {'id': re.compile(r'all_box_[a-zA-Z]{3}_advanced')})

        player_dict = dict()
        team_totals = dict()
        first_pass = True
        for i in basic_info_tables:
            for j in i.find_all('tr'):
                team = team_1_id if first_pass else team_2_id
                opponent = team_1_id if not first_pass else team_2_id
                columns = j.find_all('td')
                if j.find('a') is not None:
                    player_id = j.find('a')['href']
                    output_dict = dict()
                elif 'Team Totals' in j.text:
                    player_id = 'Team_Totals'
                else:
                    continue

                if len(columns) < 19:
                    continue

                output_dict['min_played'] = int(j.find('td', {'data-stat':'mp'}).text.split(':')[0])
                output_dict['fg'] = int(j.find('td', {'data-stat': 'fg'}).text)
                output_dict['fga'] = int(j.find('td', {'data-stat': 'fga'}).text)
                output_dict['threeP'] = int(j.find('td', {'data-stat': 'fg3'}).text)
                output_dict['threePA'] = int(j.find('td', {'data-stat': 'fg3a'}).text)
                output_dict['FT'] = int(j.find('td', {'data-stat': 'ft'}).text)
                output_dict['FTA'] = int(j.find('td', {'data-stat': 'fta'}).text)
                output_dict['ORB'] = int(j.find('td', {'data-stat': 'orb'}).text)
                output_dict['DRB'] = int(j.find('td', {'data-stat': 'drb'}).text)
                output_dict['AST'] = int(j.find('td', {'data-stat': 'ast'}).text)
                output_dict['STL'] = int(j.find('td', {'data-stat': 'stl'}).text)
                output_dict['BLK'] = int(j.find('td', {'data-stat': 'blk'}).text)
                output_dict['PF'] = int(j.find('td', {'data-stat': 'pf'}).text)
                output_dict['PTS'] = int(j.find('td', {'data-stat': 'pts'}).text)
                try:
                    output_dict['plus_minus'] = int(eval('0' + j.find('td', {'data-stat': 'plus_minus'}).text))
                except:
                    output_dict['plus_minus'] = None
                output_dict['team'] = team
                output_dict['opponent'] = opponent
                if team in game_url:
                    output_dict['home_game'] = 1
                else:
                    output_dict['home_game'] = 0

                if player_id != 'Team_Totals':
                    player_dict.setdefault(player_id, output_dict)
                else:
                    team_totals[team] = output_dict
            first_pass = False

        first_pass = True
        for i in advanced_info_tables:
            for j in i.find_all('tr'):
                team = team_1_id if first_pass else team_2_id
                opponent = team_1_id if not first_pass else team_2_id

                columns = j.find_all('td')
                if j.find('a') is not None:
                    player_id = j.find('a')['href']
                    output_dict = dict()
                elif 'Team Totals' in j.text:
                    player_id = 'Team_Totals'
                else:
                    continue

                if len(columns) < 14:
                    continue

                output_dict['ts_perc'] = convert_to_float(j.find('td', {'data-stat': 'ts_pct'}).text)
                output_dict['three_p_ar'] = convert_to_float(j.find('td', {'data-stat': 'fg3a_per_fga_pct'}).text)
                output_dict['ftr'] = convert_to_float(j.find('td', {'data-stat': 'fta_per_fga_pct'}).text)
                output_dict['ODR_perc'] = convert_to_float(j.find('td', {'data-stat': 'orb_pct'}).text)
                output_dict['DBR_perc'] = convert_to_float(j.find('td', {'data-stat': 'drb_pct'}).text)
                output_dict['AST_perc'] = convert_to_float(j.find('td', {'data-stat': 'ast_pct'}).text)
                output_dict['STL_perc'] = convert_to_float(j.find('td', {'data-stat': 'stl_pct'}).text)
                output_dict['BLK_perc'] = convert_to_float(j.find('td', {'data-stat': 'blk_pct'}).text)
                output_dict['TOV_perc'] = convert_to_float(j.find('td', {'data-stat': 'tov_pct'}).text)
                output_dict['USG_perc'] = convert_to_float(j.find('td', {'data-stat': 'usg_pct'}).text)
                output_dict['ortg'] = convert_to_float(j.find('td', {'data-stat': 'off_rtg'}).text)
                output_dict['drtg'] = convert_to_float(j.find('td', {'data-stat': 'def_rtg'}).text)

                if player_id != 'Team_Totals':
                    player_dict.get(player_id,dict).update(output_dict)
                else:
                    team_totals.get(team, dict).update(output_dict)
            first_pass = False

        with sqlite3.connect(path + 'nba.db') as conn:
            winning_team = team_1_id if team_1_score > team_2_score else team_2_id
            conn.execute('insert into game values (?, ?, ?)', (game_url, game_date, location))
            conn.execute('insert into team_game values (?,?,?,?,?,?,?)', \
                         (game_url, team_1_id, game_date, 1 if team_1_score > team_2_score else 0,team_1_score,location,None))
            conn.execute('insert into team_game values (?,?,?,?,?,?,?)', \
                         (game_url, team_2_id, game_date, 1 if team_2_score > team_1_score else 0,team_2_score,location,None))
            # for i, j in player_dict.items():
            #     conn.execute('insert into player_game_contribution values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?, ?)', \
            #                  (i, game_url, player_dict[i]['team'], player_dict[i]['opponent'], game_date,player_dict[i].get('min_played', 0), player_dict[i].get('fg', None),
            #                   player_dict[i].get('fga', None), player_dict[i].get('threeP', None), player_dict[i].get('threePA', None), player_dict[i].get('FT', None),
            #                   player_dict[i].get('FTA', None), player_dict[i].get('ORB', None), player_dict[i].get('DRB', None),
            #                   player_dict[i].get('AST', None), player_dict[i].get('STL', None), player_dict[i].get('BLK', None),
            #                   player_dict[i].get('TOV_perc', None), player_dict[i].get('PF', None), player_dict[i].get('PTS', None),
            #                   player_dict[i].get('plus_minus', None), player_dict[i].get('ts_perc', None), player_dict[i].get('three_p_ar', None),
            #                   player_dict[i].get('ftr', None), player_dict[i].get('ODR_perc', None), player_dict[i].get('DBR_perc', None),
            #                   player_dict[i].get('AST_perc', None), player_dict[i].get('STL_perc', None), player_dict[i].get('BLK_perc', None),
            #                   player_dict[i].get('TOV_perc', None), player_dict[i].get('USG_perc', None), player_dict[i].get('ortg', None),
            #                   player_dict[i].get('drtg', None), 1 if player_dict[i]['team'] == winning_team else 0, player_dict[i]['home_game']))
            for i, j in team_totals.items():
                conn.execute('insert into team_total_performance values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, ?, ?)', \
                             (game_url, team_totals[i]['team'], team_totals[i]['opponent'], game_date,team_totals[i].get('min_played', 0), team_totals[i].get('fg', None),
                              team_totals[i].get('fga', None), team_totals[i].get('threeP', None), team_totals[i].get('threePA', None),
                              team_totals[i].get('FT', None),team_totals[i].get('FTA', None), team_totals[i].get('ORB', None), team_totals[i].get('DRB', None),
                              team_totals[i].get('AST', None), team_totals[i].get('STL', None), team_totals[i].get('BLK', None),
                              team_totals[i].get('TOV_perc', None), team_totals[i].get('PF', None), team_totals[i].get('PTS', None),
                              team_totals[i].get('plus_minus', None), team_totals[i].get('ts_perc', None), team_totals[i].get('three_p_ar', None),
                              team_totals[i].get('ftr', None), team_totals[i].get('ODR_perc', None), team_totals[i].get('DBR_perc', None),
                              team_totals[i].get('AST_perc', None), team_totals[i].get('STL_perc', None), team_totals[i].get('BLK_perc', None),
                              team_totals[i].get('TOV_perc', None), team_totals[i].get('USG_perc', None), team_totals[i].get('ortg', None),
                              team_totals[i].get('drtg', None), 1 if team_totals[i]['team'] == winning_team else 0, team_totals[i]['home_game']))
            conn.commit()

        return set([i for i,j in player_dict.items()])
    except:
        traceback.print_exc()
        return set()

def read_player_info(player_url):
    print(player_url)
    try:
        r = make_request(player_base_url + player_url)
        if r is None:
            return set()
        soup = BeautifulSoup(r.text, 'html.parser')
        info_tag = soup.find('span',{'itemprop':'birthDate'})
        birth_str = ' '.join(info_tag.text.split('(')[0].replace('Born:', '').split())
        dob = datetime.datetime.strptime(birth_str, '%B %d, %Y').date()

        with sqlite3.connect('nba.db') as conn:
            conn.execute('insert into player values (?, ?)', (player_url, dob))
            conn.commit()
    except:
        traceback.print_exc()
    return set()

def main():
    build_db()
    game_urls = reduce(operator.or_ ,map(get_game_urls_at_date, get_date_range()))
    print(len(game_urls))
    player_urls = reduce(operator.or_ ,map(read_game_info, game_urls))
    print(player_urls)
    print(len(player_urls))
    # reduce(operator.or_, map(read_player_info, player_urls))

if __name__ == '__main__':
    main()