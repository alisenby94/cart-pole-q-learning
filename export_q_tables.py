import json
import os
import pandas as pd
import numpy as np


def export_q_table_as_tables(snapshot_file, output_dir='models'):
    # Open snapshot
    with open(snapshot_file, 'r') as f:
        snapshots = json.load(f)
    
    if not snapshots:
        print("No snapshots found in file!")
        return
    
    # Get first 5 timesteps
    first_episode = [s for s in snapshots if s['episode'] == 0][:5]

    # Get last 5 timesteps
    last_episode_num = max([s['episode'] for s in snapshots])
    last_episode = [s for s in snapshots if s['episode'] == last_episode_num][-5:]
    
    # Create DataFrame for first 5 timesteps
    first_data = []
    for snap in first_episode:
        first_data.append({
            'Episode': snap['episode'],
            'Timestep': snap['timestep'],
            'State': str(snap['state']),
            'Action_Taken': snap['action'],
            'Reward': snap['reward'],
            'Q_Action_0_Left': snap['q_values_for_state'][0],
            'Q_Action_1_Right': snap['q_values_for_state'][1],
            'Epsilon': snap['epsilon']
        })
    
    df_first = pd.DataFrame(first_data)
    
    # Create DataFrame for last 5 timesteps
    last_data = []
    for snap in last_episode:
        last_data.append({
            'Episode': snap['episode'],
            'Timestep': snap['timestep'],
            'State': str(snap['state']),
            'Action_Taken': snap['action'],
            'Reward': snap['reward'],
            'Q_Action_0_Left': snap['q_values_for_state'][0],
            'Q_Action_1_Right': snap['q_values_for_state'][1],
            'Epsilon': snap['epsilon']
        })
    
    df_last = pd.DataFrame(last_data)
    
    # Create base filename
    base_name = os.path.basename(snapshot_file).replace('.json', '')
    
    # Save as CSV
    first_csv = os.path.join(output_dir, f'{base_name}_FIRST_5_table.csv')
    last_csv = os.path.join(output_dir, f'{base_name}_LAST_5_table.csv')
    
    df_first.to_csv(first_csv, index=False, float_format='%.6f')
    df_last.to_csv(last_csv, index=False, float_format='%.6f')
    
    # Summarize
    print(f"CSV tables saved:")
    print(f"  - {first_csv}")
    print(f"  - {last_csv}")
    
if __name__ == "__main__":
    import glob
    
    # Find the most recent snapshot JSON file
    snapshot_files = glob.glob('models/q_table_snapshots_*.json')
    
    if not snapshot_files:
        print("No Q-table snapshots found.")
        print("Please run train.py first.")
    else:
        latest_snapshot = max(snapshot_files, key=os.path.getctime)
        print(f"Using snapshot file: {latest_snapshot}\n")
        export_q_table_as_tables(latest_snapshot)
        print("\nDone.")
