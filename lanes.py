import cv2
import numpy as np
import matplotlib.pyplot as plt

# Reading the image
image = cv2.imread('lane.jpg')  
lane_image = np.copy(image)  

def canny(image):
    # edge detection
    # convert image to grayscale (black & white)
    gray = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)

    # Noise filtering & image smoothening
    # Using a gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # optional 

    # Use of canny function to calculate change in gradient 
    # use of threshold 0f 1:3
    canny = cv2.Canny(blur, 50, 150)  # 1:3 == 50:150
    return canny

def make_coordinate(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (50,205,50), 10)
    return line_image

def average_slope_intercept(image, lines):
    left_fit = []  # coordinates of left lines
    right_fit = [] # coordinates of right lines
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0 :
            left_fit.append((slope, intercept))            
        else:
            right_fit.append((slope, intercept))
    left_fit_ave = np.average(left_fit, axis=0)
    right_fit_ave = np.average(right_fit, axis=0) 
    left_line = make_coordinate(image, left_fit_ave)
    right_line =make_coordinate(image, right_fit_ave)
    return np.array([left_line, right_line])
    

# Defining points of focus
def region_of_interest(image):
    height = image.shape[0]
    # polygon = np.array([[(250, height), (850, height), (365, 230)]]) # creating a polygon
    polygon = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image) # turn everthing to black
    cv2.fillPoly(mask, polygon, 255) # turn only the polygon to white
    masked_image = cv2.bitwise_and(image, mask) # the mask to the point on the road requiried
    return masked_image

# canny_image = canny('test_road.jpg')
# cropped_image = region_of_interest(canny_image)
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # detecting lines on the image
# ave_lines = average_slope_intercept(lane_image, lines)
# line_image = display_lines(lane_image,ave_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow("lane", combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("road_test.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # detecting lines on the image
    ave_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame,ave_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow("lane", combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

