import cv2
import numpy as np

# Create a simple demo video with shapes that could be detected as potholes
def create_demo_video():
    # Video parameters
    width, height = 640, 480
    fps = 20
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('/app/demo.mp4', fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create a frame with road-like background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 70  # Dark gray road
        
        # Add some "road lines"
        cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
        # Add moving "potholes" (dark circles that move across the frame)
        circle_x = int((frame_num * 3) % (width + 100))
        circle_y = height // 2 + int(50 * np.sin(frame_num * 0.1))
        
        # Draw a pothole-like circle
        cv2.circle(frame, (circle_x, circle_y), 30, (20, 20, 20), -1)
        cv2.circle(frame, (circle_x, circle_y), 25, (10, 10, 10), -1)
        
        # Add another pothole
        circle2_x = int((frame_num * 2 + 200) % (width + 100))
        circle2_y = height // 3 + int(30 * np.cos(frame_num * 0.15))
        cv2.circle(frame, (circle2_x, circle2_y), 20, (15, 15, 15), -1)
        
        # Add some noise/texture
        noise = np.random.randint(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)
        
        out.write(frame)
    
    out.release()
    print("Demo video created: /app/demo.mp4")

if __name__ == "__main__":
    create_demo_video()