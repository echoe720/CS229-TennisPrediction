import pandas as pd
import numpy as np
import chardet
import io

def main():
    with open('charting-m-matches.csv', 'rb') as rawdata:
        result = chardet.detect(rawdata.read(100000))
    print(result)
    men_points = pd.read_csv("charting-m-points_new.csv", encoding='Windows-1252', dtype=str)
    wom_points = pd.read_csv("charting-w-points_new.csv", encoding='Windows-1252', dtype=str)

    all_points = pd.concat([men_points, wom_points])

    all_points = all_points[['match_id', 'Pt', 'Set1', 'Set2', 'Gm1', 'Gm2', 'PtWinner', 'Gender', 'Svr']]

    unique_matches = len(all_points[['match_id']].drop_duplicates())

    mens_points_np = all_points.to_numpy()
    
    winnersPerMatch = {} # stores winners of each match in order.
    # winnerPerMatch = {}
    point_idx = 0
    # Add Y-Values as final column
    for i in range(unique_matches):
        match_id = mens_points_np[point_idx][0]
        point_num = 0
        prev_point_winner = -1

        while (point_idx < mens_points_np.shape[0] and mens_points_np[point_idx][0] == match_id):
            curr_match = mens_points_np[point_idx]
            point_winner = int(curr_match[6])
            point_idx += 1

        prev_point_winner = str(2 - point_winner)
        point_num += 1
        point_idx += 1

        winnersPerMatch[match_id] = prev_point_winner

    start_index = 0
    
    final_data = []

    max_length = 550
    max_points = 0

    for i in range(unique_matches):
        match_id = mens_points_np[start_index][0]
        point_num = 0
        point_history = []
        game_history = []
        prev_game_score = 0
        set_history = []
        prev_set_score = 0
        
        prev_point_winner = -1
        server = -1

        while (start_index < mens_points_np.shape[0] and mens_points_np[start_index][0] == match_id):
            curr_match = mens_points_np[start_index]
            point_winner = int(curr_match[6])
            p1sets = float(curr_match[2])
            p2sets = float(curr_match[3])
            p1games = float(curr_match[4])
            p2games = float(curr_match[5])
            if server == -1:
                server = 2 - int(curr_match[8])
            
            if p1games + p2games != prev_game_score:
                prev_game_score = p1games + p2games
                game_history.append(prev_point_winner)
                if server == 1:
                    point_history.append('5' if prev_point_winner == '1' else '-7')
                else:
                    point_history.append('7' if prev_point_winner == '1' else '-5')
                server = -1

            if p1sets + p2sets != prev_set_score:
                prev_set_score = p1sets + p2sets
                set_history.append(prev_point_winner)
                game_history.append(-1)
                point_history.append('10' if prev_point_winner == '1' else '-10')

            point_history.append(str(2 - point_winner))
            
            if len(point_history) > max_points:
                max_points = len(point_history)
            if point_num % 10 == 0:
                #Add row
                """
                Features needed: 
                    entire point history, padded by max_length + 70
                    1 if p1 won, 0 if p2 won
                    5 if p1 won game, -5 if p2 won game
                    7 if p1 broke, -7 if p2 broke
                    10 if p1 won set, -10 if p2 won set
                """
                new_row = []
                
                for i in range(max_length):
                    if i < len(point_history):
                        new_row.append(point_history[i])
                    else:
                        new_row.append(-1)
                #print(match_id, new_row)
                new_row.append(int(winnersPerMatch[match_id]))
                #print(int(winnersPerMatch[match_id]))
                final_data.append(new_row)

            prev_point_winner = str(2 - point_winner)
            point_num += 1
            start_index += 1

    print(len(final_data))
    print(max_points)
    df = pd.DataFrame(final_data)
    print(df.head)

    df = df.sample(frac=1)
    eighty_percent_split = round(len(final_data) * 0.8)
    ninety_percent_split = round(len(final_data) * 0.9)
    train_set = df.iloc[:eighty_percent_split, :]
    val_set = df.iloc[eighty_percent_split:ninety_percent_split, :]
    test_set = df.iloc[ninety_percent_split:, :]

    train_set.to_csv("lstm_train_serve.csv")
    print(len(train_set))
    val_set.to_csv("lstm_val_serve.csv")
    print(len(test_set))
    test_set.to_csv("lstm_test_serve.csv")
    print(len(test_set))

if __name__ == "__main__":
    main()
