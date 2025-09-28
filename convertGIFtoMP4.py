from moviepy import VideoFileClip

# Load the GIF
clip = VideoFileClip("results/phase1_spatial_comparison.gif")

# Export to MP4 (H.264 codec)
clip.write_videofile("results/phase1_spatial_comparison.mp4", codec="libx264")
