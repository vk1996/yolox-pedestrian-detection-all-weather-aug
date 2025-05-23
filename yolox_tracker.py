from yolox_onnx import YOLOX_ONNX
import cv2
import numpy as np
from sort import  Sort
from collections import defaultdict
from coco_classes import class_names



mot_tracker = Sort(max_age=15, min_hits=3)
yolox_onnx_client=YOLOX_ONNX('models/yolox_nano_pretrained.onnx')


videopath="pedestrian.mp4"
vid = cv2.VideoCapture(videopath)  # or 'path/to/your.avi
tracker_ids=defaultdict(list)
output_file = 'output.mp4'
# Define the codec and create VideoWriter object
frame_width = 1280
frame_height = 720
fps = 30
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For .mp4 output
out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

while True:
    # Capture frame-by-frame
    ret, frame = vid.read()

    # If ret is False, it means we have reached the end of the video
    if not ret:
        break


    boxes,scores,labels=yolox_onnx_client.predict(frame)



    for i,bbox in enumerate(boxes):
        xmin, ymin, xmax, ymax = bbox
        #cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 128, 0), int(0.005 * frame.shape[1]))


    # Filter the detections (e.g., based on confidence threshold)
    # confidence_threshold = 0.5
    dets=[]

    for i, score in enumerate(scores):
        #if True:
        if score > 0.8:
            bbox = boxes[i]
            label_id = labels[i]
            conf = score.item()
            #dets.append([*bbox, 0.99])
            if label_id==0:
                dets.append([*bbox,int(label_id)])

    if len(dets)>0:
        dets=np.array(dets) # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
    else:
        dets=np.empty([0,5])


    trackers = mot_tracker.update(dets)


    for i,d in enumerate(trackers):
        xmin, ymin, xmax, ymax , track_id,class_id= d[0],d[1],d[2],d[3],d[4],d[5]
        class_name = class_names[int(class_id)]
        tracker_ids[class_name].append(track_id)
        curr_track_id=list(np.unique(tracker_ids[class_name])).index(track_id)+1



        cx=int((xmax+xmin)/2)
        cy = int((ymax+ymin) / 2)
        cv2.putText(frame, f"{class_name}:{int(curr_track_id)}", (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255),int(0.00125 * frame.shape[1]))
        ##cv2.putText(frame,f"{class_name}:{int(track_id)}", (cx,cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255),int(0.0025 * frame.shape[1]))
        cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,255), int(0.0025 * frame.shape[1]))


    cv2.imshow('Kalman tracker', frame)
    out.write(frame)

    # Simulate wait for key press to continue, press 'q' to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
out.release()
vid.release()
cv2.destroyAllWindows()
