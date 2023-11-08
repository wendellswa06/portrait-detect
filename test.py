from ultralytics import YOLO
from moviepy.editor import VideoFileClip
def resize_video(input_path, output_path, new_width, new_height):
    video = VideoFileClip(input_path)
    resized_video = video.resize((new_width, new_height))
    resized_video.write_videofile(output_path, codec="libx264")

# model = YOLO("runs/detect/train/weights/best.pt")
# resize_video("input.mp4", "output.mp4", 1920, 1080)
model = YOLO("best.pt")
model.predict(source="output.mp4", show=True)