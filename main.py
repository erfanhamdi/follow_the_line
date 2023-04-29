import cv2
import numpy as np
import matplotlib.pyplot as plt
import yaml

def find_pattern_zz(frame):
    # Convert the frame to hsv
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # set the lower and upper bound for thresholding,
    # you can plot the hsv frame by uncommenting the next line of code
    # And check the lower and upper color bounds
    # plt.imshow(hsv_frame)
    lower_green = np.array([30, 50, 10])
    upper_green = np.array([90, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    kernel = np.ones((5, 5), np.uint8)
    # In order to help findContour function to find the pattern
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)
    # Find the contours and the hierarchy of them
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # draws the contour on the first frame
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    return frame, contours, hierarchy

def find_pattern_inf(frame):
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

def detect_bg_page(frame):
    # This function crops the frames to the white background page
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to get a binary image
    # Threshold bounds can be found by plotting the gray frame.
    ret, binary = cv2.threshold(gray, 110, 190, cv2.THRESH_BINARY)
    # Find contours in the binary image
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Find the contour with the largest area
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour
            
    # Draw the contour on the original frame for visualization
    cv2.drawContours(frame, [max_contour], -1, (0, 0, 255), 2)
    
    # Get the bounding box of the contour and crop the frame
    x, y, w, h = cv2.boundingRect(max_contour)
    cropped_frame = frame[y:y+h, x:x+w]
    
    # Show the original and cropped frames
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Cropped Frame', cropped_frame)
    cv2.waitKey(1)
    return cropped_frame, x, y, w, h

def find_laser(hsv_frame):
    """
    This function finds the laser pointer in the frames and draws a
    circle around it for visualization.
    """
    hsv_frame = cv2.cvtColor(hsv_frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([160, 80, 120])
    upper_red = np.array([180, 155, 255])
    mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    laser_contour_ = np.array(np.where(mask == 255) ).T[:, [1, 0]]
    laser_dot_center = np.mean(laser_contour_, 0).astype(np.int32)
    cv2.circle(frame, laser_dot_center, 5, (0, 0, 150), 3)
    return frame, laser_dot_center

def cross_count(dist_list):
    """
    This function counts the number of times the laser pointer has crossed
    the pattern boundaries
    """
    import itertools
    return len(list(itertools.groupby(dist_list, lambda dist_list: dist_list > 0)))-1

def merge_contours(contours):
    """
    Merges all of the contours for visualization purpose
    """
    cnt = contours[0]
    for cnt_ in contours[1:]:
        cnt = np.concatenate([cnt, cnt_])
    return cnt

def inside_out_algo(contours, hierarchy, pnt):
    """
    This function uses the hierarchy attribute of the contours
    to find the inner and outer regions of the contours
    """
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

    video_address = config["video_address"]
    file_name = video_address.split("/")[-1].split(".")[0]
    cap = cv2.VideoCapture(video_address)

    ret, frame_1 = cap.read()
    frame_1, x, y, w, h = detect_bg_page(frame_1)
    # You can select the type of the pattern that is to be analysed here
    pattern_type = 0
    if pattern_type:
        frame, contours, hierarchy = find_pattern_inf(frame_1)
    else:
        frame, contours, hierarchy = find_pattern_zz(frame_1)
    
    frame_shape_y, frame_shape_x, _ = frame.shape
    # Writer encoding
    out = cv2.VideoWriter(f'{file_name}_out.mp4', cv2.VideoWriter_fourcc(*'mp4v'),20, [frame_shape_x, frame_shape_y])
    
    dist_list = []
    trajectory = []
    # Region of Interest 
    roi_contour = np.array([[[0, 0]], [[frame_shape_x, 0]], [[frame_shape_x, frame_shape_y]], [[0, frame_shape_y]], [[0, 0]]])
    cv2.drawContours(frame, [roi_contour], 0, (255,255,0), 3)
    
    no_frames = 0
    while True:
        no_frames +=1
        ret, frame = cap.read()
        try:
            frame = frame[y:y+h, x:x+w]
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
    print(no_frames)
    contours = merge_contours(contours)
    plt.figure()
    plt.scatter(contours[..., 0], -contours[..., 1], s=1)
    plt.scatter(trajectory[:, 0], -trajectory[:, 1], s=2)
    plt.show()
    out.release()
    cap.release()
