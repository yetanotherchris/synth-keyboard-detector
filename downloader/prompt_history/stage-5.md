Can you add a new section to the video_downloader.py (currently there's 2 sections). This section should have the following functionality:

1. From all the videos in the downloaded_videos directory, extracts a single frame as a PNG.
2. This single frame should come from a start point of 2 minutes into the video.
3. The filename of the frame should be in the format "{incremented_number}_{videoid}"
4. {incremented_number} should start at 1 and increment
5. {videoid} is the video id from video
6. Put these PNG files inside the "labelling_images" directory.

can the increment number use "0" as a prefix for the first ten numbers, for directory sorting on Windows. e.g. 01, 02, 03, 04