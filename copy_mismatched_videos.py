import os
import shutil
import pandas as pd

def copy_mismatched_videos():
    csv_path = 'scenario_labels/any_mismatches.csv'
    source_dir = 'dataset/train'
    dest_dir = 'mismatched'

    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found. Please run compare_rounds.py first.")
        return

    # Create destination directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'Video Name' not in df.columns:
        print("Error: 'Video Name' column not found in CSV.")
        return

    video_names = df['Video Name'].unique()
    print(f"Found {len(video_names)} unique mismatched videos to copy.")

    copied_count = 0
    missing_count = 0

    for video_name in video_names:
        video_name = str(video_name).strip()
        source_path = os.path.join(source_dir, video_name)
        dest_path = os.path.join(dest_dir, video_name)

        if os.path.exists(source_path):
            try:
                shutil.copy2(source_path, dest_path)
                # print(f"Copied: {video_name}")
                copied_count += 1
            except Exception as e:
                print(f"Error copying {video_name}: {e}")
        else:
            print(f"Warning: Source file not found: {source_path}")
            missing_count += 1

    print("-" * 30)
    print(f"Copy process completed.")
    print(f"Successfully copied: {copied_count}")
    print(f"Missing source files: {missing_count}")
    print(f"Videos are located in: {os.path.abspath(dest_dir)}")

if __name__ == "__main__":
    copy_mismatched_videos()
