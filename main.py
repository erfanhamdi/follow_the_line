import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

def find_pattern(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 10])
    upper_green = np.array([90, 255, 255])
    # lower_green = np.array([0, 30, 10])
    # upper_green = np.array([190, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    return frame, contours, hierarchy

def find_pattern_black(frame):
    # Convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Threshold the grayscale image to get a binary image
    ret, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.dilate(thresh, kernel, iterations=3)
    # Find the contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    return frame, contours, hierarchy

def find_laser(hsv_frame):
    hsv_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)
    # lower_red = np.array([172, 50, 120])
    lower_red = np.array([160, 80, 120])
    upper_red = np.array([180, 155, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    laser_contour_ = np.array(np.where(mask == 255) ).T[:, [1, 0]]
    laser_dot_center = np.mean(laser_contour_, 0).astype(np.int32)
    cv2.circle(frame, laser_dot_center, 5, (0, 0, 150), 3)
    return frame, laser_dot_center

def cross_count(dist_list):
    import itertools
    return len(list(itertools.groupby(dist_list, lambda dist_list: dist_list > 0)))-1

def merge_contours(contours):
    cnt = contours[0]
    for cnt_ in contours[1:]:
        cnt = np.concatenate([cnt, cnt_])
    return cnt

def inside_out_algo(contours, hierarchy, pnt):
    # Finding the outermost contours
    outer_main_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    child_1 = hierarchy[0][outer_main_contour_idx][2]
    child_2 = hierarchy[0][child_1][0]
    outer_2nd_contour_idx = hierarchy[0][outer_main_contour_idx][0] 
    if cv2.pointPolygonTest(contours[child_1], pnt, False)>=0:
        return -cv2.pointPolygonTest(contours[child_1], pnt, True)
    if cv2.pointPolygonTest(contours[child_2], pnt, False)>=0:
        return -cv2.pointPolygonTest(contours[child_2], pnt, True)
    elif cv2.pointPolygonTest(contours[outer_main_contour_idx], pnt, False)>=0:
        return cv2.pointPolygonTest(contours[outer_main_contour_idx], pnt, True) 
    if cv2.pointPolygonTest(contours[outer_2nd_contour_idx], pnt, False)>=0: 
        return cv2.pointPolygonTest(contours[outer_2nd_contour_idx], pnt, True)  
    else:
        return cv2.pointPolygonTest(contours[outer_main_contour_idx], pnt, True)

if __name__=="__main__":

    with open("config.yml") as f: 
        config = yaml.safe_load(f)
    
    x1 = config["x1"]
    x2 = config["x2"]
    y1 = config["y1"]
    y2 = config["y2"]

    video_address = config["video_address"]
    file_name = video_address.split("/")[-1].split(".")[0]
    cap = cv2.VideoCapture(video_address)

    ret, frame_1 = cap.read()
    frame_1 = frame_1[y1:y2, x1:x2, :]
    frame, contours, hierarchy = find_pattern(frame_1)
    if len(contours)==0:
        frame, contours, hierarchy = find_pattern_black(frame_1)
    
    frame_shape_y, frame_shape_x, _ = frame.shape
    # Writer encoding
    out = cv2.VideoWriter(f'{file_name}_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'),20, [frame_shape_x, frame_shape_y])
    
    dist_list = []
    trajectory = []
    # Region of Interest 
    roi_contour = np.array([[[0, 0]], [[frame_shape_x, 0]], [[frame_shape_x, frame_shape_y]], [[0, frame_shape_y]], [[0, 0]]])
    cv2.drawContours(frame, [roi_contour], 0, (255,255,0), 3)

    while True:
        ret, frame = cap.read()
        try:
            frame = frame[y1:y2, x1:x2, :]
            if not ret:
                break
            frame, center_pnt = find_laser(frame) 
            center_pnt = tuple([int(round(center_pnt[0]) ), int(round( center_pnt[1] )) ])
            in_roi = cv2.pointPolygonTest(roi_contour, center_pnt, False)
            if in_roi<0:
                continue
            trajectory.append(center_pnt)
            dist = inside_out_algo(contours, hierarchy, center_pnt)/6
            # Writing the penalty value on each frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, str(dist), (0, frame_shape_y-10), font, 0.5, (0, 255, 0), 2,  cv2.LINE_AA)
            
            dist_list.append(dist)
            
            cv2.drawContours(frame, contours, -1, (0,255,0), 3)
            out.write(frame)
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
        except Exception as e:
            print(e)
            break
    
    trajectory = np.array(trajectory)
    dist_list = np.array(dist_list)
    penalty_arr = dist_list[dist_list<0]
    # Finding the biggest contour
    outer_main_contour_idx = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
    # Normalizing by the biggest one
    contour_area = cv2.contourArea(contours[outer_main_contour_idx])
    normalizing_factor = contour_area**0.5
    print(f"Total Number of crossing the boundaries: {cross_count(dist_list)}")
    print(f"The total Penalty is: {penalty_arr.sum()/normalizing_factor}")
    # To plot the Trajectory Uncomment this part
    plt.figure()
    plt.scatter(contours[..., 0], -contours[..., 1])
    plt.scatter(trajectory[:, 0], -trajectory[:, 1], s=2)
    plt.show()
    out.release()
    cap.release()
