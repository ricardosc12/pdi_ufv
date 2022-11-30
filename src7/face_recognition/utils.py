from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
#  150, 176
ffmpeg_extract_subclip("video.mkv", 150, 176, targetname="video_cutted.mkv")

