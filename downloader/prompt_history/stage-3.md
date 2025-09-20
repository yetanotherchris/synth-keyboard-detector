For video_downloader.py:

Change the functionality for the start time for frame extraction so it now takes the video ID.

If the value of this field is "0s" then the field is ignored.

Otherwise the field should be in the format "videoid:time".
For example "ae94835:2m" would extract the video with id "ae94835" starting at the 2 minute mark.

-- Clarification
1. Good question. I've realised the start time functionality should be completely separate. So a brand new section that allows you to extract one video's frames from a certain start point.
2. see 1)

-- Follow up
The "Configuration section" (start time for frame extraction) for bulk processing can be removed, as it isn't needed.