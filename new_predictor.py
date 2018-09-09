import sqlite3
import pandas as pd
from elo import *
import traceback
import time
import tqdm

elo_k_list = [10, 25, 50, 100, 250, 500, 1000]
path = r'C:\Users\trist\Documents\db_loc\nba/'
starting_elo_dict = {k: starting_elo for k in elo_k_list}


def look_up_elo_dict(result_df, team_name, date_played, elo_dict):
    temp_elo = dict()
    try:
        return elo_dict[(team_name, date_played)]
    except KeyError:
        last_match = result_df[(result_df['team_name'] == team_name) & (result_df['date_played']< date_played)].sort_values('date_played', ascending=False).head(1)
        opponent = result_df[(result_df['team_name'] == team_name) & (result_df['date_played'] == date_played)]['opponent_name'].values

        if len(opponent) > 0:
            opponent_name = opponent[0]
            opponent_matches = result_df[(result_df['team_name'] == opponent_name) & (result_df['date_played']< date_played)].sort_values('date_played', ascending=False)
            opponent_last_match = opponent_matches.head(1)
            if len(opponent_last_match) > 0:
                opponent_pre_fight_elo = look_up_elo_dict(result_df, opponent_name, opponent_last_match['date_played'].values[0], elo_dict)
            else:
                opponent_pre_fight_elo = {'pre':starting_elo_dict.copy(), 'post':starting_elo_dict.copy()}

            if len(last_match['date_played'].values) > 0:
                past_fight_elo = look_up_elo_dict(result_df, team_name,
                                                     last_match['date_played'].values[0], elo_dict)
                temp_elo['pre'] = past_fight_elo['post']
            else:
                temp_elo['pre'] = starting_elo_dict.copy()

            current_match = result_df[
                (result_df['team_name'] == team_name) & (result_df['date_played'] == date_played)].head(1)
            if len(current_match) == 0:
                temp_elo['post'] = None
            else:
                if current_match['result'].values[0] == 1:
                    temp_elo['post'] = calculate_different_elos(1, temp_elo['pre'],
                                                                opponent_pre_fight_elo['post'], elo_k_list)
                elif current_match['result'].values[0] == 0:
                    temp_elo['post'] = calculate_different_elos(0, temp_elo['pre'],
                                                                opponent_pre_fight_elo['post'], elo_k_list)
                else:
                    temp_elo['post'] = temp_elo['pre']
            current_result = current_match['result'].values[0]
        else:
            temp_elo = look_up_elo_dict(result_df, team_name, last_match['date_played'], elo_dict)
            current_result = None

        elo_dict[(team_name, date_played)] = temp_elo
    #print('temp', team_name, date_played, temp_elo, current_result)
    return temp_elo


def calculate_elo_df(match_df, input_list = None, name = ''):
    elo_dict = dict()
    match_df = match_df.sample(frac=1)
    start_time = time.time()

    match_df = match_df.set_index(['team_name', 'date_played'], drop = False)

    if input_list:
        for i in input_list:
            try:
                # print(i['team_name'], i['opponent_name'], i['date_played'])
                look_up_elo_dict(match_df, i['team_name'], i['date_played'], elo_dict)
                look_up_elo_dict(match_df, i['opponent_name'], i['date_played'], elo_dict)
            except:
                traceback.print_exc()
    else:
        match_df = match_df.sort_values('date_played')
        print('calculating elos')
        for  i, j in tqdm.tqdm(match_df.iterrows(), total=match_df.shape[0]):
            try:
                temp_values = look_up_elo_dict(match_df, j['team_name'], j['date_played'], elo_dict)
                # print((time.time()-start_time)/(count+1), (time.time()-start_time)/len(elo_dict.keys()), count,
                #       len(elo_dict.keys()), j['team_name'], j['date_played'], temp_values)

                # if count%1000 == 0 and count > 0:
                #     pass
                #     store_elo_dict(elo_dict)
            except:
                traceback.print_exc()

    unrolled_elo_dicts = []
    for i, j in elo_dict.items():
        team_name = i[0]
        date_played = i[1]
        pre_elos = {'elo_pre_{0}_{1}'.format(k1, name):k2 for k1, k2 in j['pre'].items()}
        post_elos = {'elo_post_{0}_{1}'.format(k1, name):k2 for k1, k2 in j['post'].items()}

        temp_dict = dict()
        temp_dict.update({'team_name':team_name})
        temp_dict.update({'date_played': date_played})
        temp_dict.update(pre_elos)
        temp_dict.update(post_elos)
        unrolled_elo_dicts.append(temp_dict)

    df = pd.DataFrame.from_dict(unrolled_elo_dicts)
    return df


def read_data():
    with sqlite3.connect(path + 'nba.db') as conn:
        game_results = pd.read_sql('''select distinct game.g_id, team_game.team_name, team_game.result, team_game.date_played, team_game.game_location
        from game join team_game on game.g_id = team_game.g_id where team_game.date_played > ?''', conn, params=('1985-01-01',))

        team_contribution = pd.read_sql('''select *
                from team_total_performance ''', conn)
        return game_results, team_contribution

def get_features():
    game_results, team_contribution = read_data()
    teams = list(set(team_contribution['team_name'].tolist()))

    print(team_contribution.columns)
    team_contribution = team_contribution.sort_values('date_played')

    dfs = []
    for i in teams:
        df_t = team_contribution[team_contribution['team_name'] == i]
        df_t_r = df_t.rolling(window=2)
        df_t_r_m = df_t_r.mean()
        df_t_r_m_1 = df_t_r_m.shift(-1)
        df_t_r_m_1.columns = ['feature_' + i for i in df_t_r_m_1.columns]
        # print(df_t_r_m_1.columns)
        df_t_r_m_1 = df_t.join(df_t_r_m_1)
        df_t = df_t_r_m_1[[i for i in df_t_r_m_1.columns if (('g_id' in i or 'opponent_name' in i or'team_name' in i or 'date_played' in i or
        'home_game' in i or 'result' in i) and 'feature' not in i) or (('opponent_name' not in i or'g_id' not in i or 'team_name' not in i or 'date_played' not in i or
        'home_game' not in i or 'result' not in i) and 'feature' in i)]]
        dfs.append(df_t)
    df = pd.concat(dfs)
    # game_results = game_results['team_name', 'date_played', 'result']
    # df = df.merge(game_results)

    elo_df = calculate_elo_df(df)
    df = df.merge(elo_df)
    print(teams)

get_features()