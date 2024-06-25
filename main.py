import cv2
from collections import defaultdict
from ultralytics import YOLO
import multiprocessing
import torch
from util import read_license_plate

# Load YOLO model
model_path = 'models/best_model.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
license_plates = YOLO(model_path).to(device)

# Function to process frames and perform OCR
def process_frames(frame_queue, license_plate_results):
    while True:
        frame, boxes, track_ids = frame_queue.get()
        if frame is None:
            break
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            crop_img = frame[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]
            license_plate_results[track_id] = read_license_plate(crop_img)
        frame_queue.task_done()

def main():
    frame_queue = multiprocessing.JoinableQueue()
    license_plate_results = defaultdict(str)
    num_processes = multiprocessing.cpu_count()
    processes = []
    for _ in range(num_processes):
        p = multiprocessing.Process(target=process_frames, args=(frame_queue, license_plate_results))
        p.start()
        processes.append(p)

    video_path = "video/test.mp4"
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec = cv2.VideoWriter_fourcc(*'XVID')

    output_video_path = "output_video.avi"
    out = cv2.VideoWriter(output_video_path, codec, fps, (width, height))

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = license_plates.track(frame, persist=True)
            if results is None or results[0].boxes.id is None:
                cv2.imshow("Frame", frame)
                out.write(frame)
                print("No objects detected in this frame.")
                continue
            else:
                print(results[0].boxes)
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()

                # Process each detected box separately for OCR
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    crop_img = frame[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]
                    license_plate_results[track_id] = read_license_plate(crop_img)

                # Display annotated results on frame
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    cv2.rectangle(frame, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (255, 0, 255), 3)
                    cv2.putText(frame, f"ID: {track_id}", (int(x - w/2), int(y - h/2) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Read license plate using track_id
                    license_plate = license_plate_results.get(track_id, "")
                    if license_plate:
                        cv2.putText(frame, f"Plate: {license_plate}", (int(x - w/2), int(y + h/2) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                cv2.imshow("Frame", frame)
                out.write(frame)

                # Exit loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    # Stop frame processing processes
    for _ in range(num_processes):
        frame_queue.put((None, None, None))

    for p in processes:
        p.join()

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
