Can you take the logic in this python application, and create a standalone Python script in the "downloader" directory.

This new script will:

Stage 1: Read a text file called "youtube-urls.txt" containing youtube urls, and download all the videos into a directory called "downloaded_videos". The filename should be the id of the Youtube video (the "v" querystring).

The script should check if the .mp4 already exists in "downloaded_videos" before downloading. If the video exists, it skips downloading.

Stage 2: Extract the first 2 minutes of frames (as PNG or JPG) for each video in the "downloaded_videos" directory, storing the frame images in the directory "extracted_frames/{youtube_id}/". 

"{youtube_id}" should be the id (the "v" querystring) of the youtube video.

Use Streamlit to display this script, with Streamlit showing its progress in both downloading and which stage (1 or 2) it's at.