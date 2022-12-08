import pandas as pd
import numpy as np
import io
import csv

def main():
    men_points = pd.read_csv("charting-m-points_new.csv", encoding='Windows-1252', dtype=str)
    wom_points = pd.read_csv("charting-w-points_new.csv", encoding='Windows-1252', dtype=str)

    all_points = pd.concat([men_points, wom_points])
    all_points = all_points[['match_id', 'Pt', 'Svr', 'Set1', 'Set2', 'Gm1', 'Gm2', 'PtWinner', 'Gender']]

    unique_matches = len(all_points[['match_id']].drop_duplicates())

    all_points_np = all_points.to_numpy()

    winnersPerMatch = {} # stores winners of each match in order.
    point_idx = 0
    # Add Y-Values as final column
    for i in range(unique_matches):
        match_id = all_points_np[point_idx][0]
        point_num = 0
        prev_point_winner = -1

        while (point_idx < all_points_np.shape[0] and all_points_np[point_idx][0] == match_id):
            curr_match = all_points_np[point_idx]
            point_winner = int(curr_match[7])
            point_idx += 1

        prev_point_winner = str(2 - point_winner)
        point_num += 1
        point_idx += 1

        winnersPerMatch[match_id] = prev_point_winner

    start_index = 0
    final_data = []

    for i in range(unique_matches):
        match_id = all_points_np[start_index][0]
        point_num = 0
        point_history = []
        game_history = []
        prev_game_score = 0
        set_history = []
        prev_set_score = 0
        prev_point_winner = -1

        while (start_index < all_points_np.shape[0] and all_points_np[start_index][0] == match_id):
            curr_match = all_points_np[start_index]
            point_winner = int(curr_match[7])
            p1sets = float(curr_match[3])
            p2sets = float(curr_match[4])
            p1games = float(curr_match[5])
            p2games = float(curr_match[6])

            point_history.append(str(2 - point_winner))
            
            if p1games + p2games != prev_game_score:
                prev_game_score = p1games + p2games
                game_history.append(prev_point_winner)

            if p1sets + p2sets != prev_set_score:
                prev_set_score = p1sets + p2sets
                set_history.append(prev_point_winner) 
                game_history.append(-1)

            if point_num % 20 == 0:
                #Add row
                """
                Features needed: 
                    number of sets in match
                    Handedness of player 1 (RH 1, LH 0)
                    Handedness of player 2 (RH 1, LH 0)
                    Gender of player 1 (M 0, F 1)                                                  !
                    Gender of player 2 (M 1, F 0)                                                  !
                    Who served first (1 if p1, 0 if p2)                                            !
                    P1 set count                                                                   !
                    P2 set count                                                                   !
                    P1 game count in set                                                           !
                    P2 game count in set                                                           ! 
                    Set history (5 values, 1 if p1 won, 0 if p2 won, -1 if hasn't happened yet)    !
                    Game history (65 values, -1 at end of each 13 point stretch to fill gaps)      ! 
                    Last 5 points                                                                  !

                    Y-value: who won the match (1 if p1 won, 0 if p2 won)                          !
                """
                new_row = []
                new_row.append(curr_match[8]) #Gender of the players
                new_row.append(str(2 - int(all_points_np[0][2]))) # Who served first
                new_row.append(p1sets) #Number of sets player 1 has won
                new_row.append(p2sets) #Number of sets player 2 has won
                new_row.append(p1games) #Number of games player 1 has won in the current set
                new_row.append(p2games) #Number of games player 2 has won in the current set

                #Add set history
                for j in range(5): 
                    if j < len(set_history):
                        new_row.append(set_history[j])
                    else:
                        new_row.append(-1)
                
                #Add game history
                g = 0
                total_added = 0
                index = 0
                while total_added < 65:
                    if index >= len(game_history):
                        new_row.append(-1)
                        total_added += 1
                    elif game_history[index] != -1:
                        new_row.append(game_history[index])         
                        total_added += 1
                        g += 1 
                    else:
                        while g < 13:
                            new_row.append(-1)
                            g += 1
                            total_added += 1
                        g = 0
                    index += 1
                
                #Last 5 points
                if len(point_history) < 5:
                    for j in range(5):
                        if j < len(point_history):
                            new_row.append(point_history[j])
                        else:
                            new_row.append(-1)
                else:
                    for j in range(5):
                        new_row.append(point_history[-j])
                new_row.append(int(winnersPerMatch[match_id]))

                # add all the rows in
                final_data.append(new_row)

            prev_point_winner = str(2 - point_winner)
            point_num += 1
            start_index += 1
        

    print(len(final_data))
    df = pd.DataFrame(final_data)

    df = df.sample(frac=1)
    eighty_percent_split = round(len(final_data) * 0.8)
    ten_percent_split = round(len(final_data) * 0.1)

    train_set = df.iloc[:eighty_percent_split, :]
    eval_set = df.iloc[eighty_percent_split: eighty_percent_split + ten_percent_split, :]
    test_set = df.iloc[eighty_percent_split + ten_percent_split:, :]


    train_set.to_csv("train_points_data.csv")
    test_set.to_csv("val_points_data.csv")
    test_set.to_csv("test_points_data.csv")

if __name__ == "__main__":
    main()