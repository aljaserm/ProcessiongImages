import cv2 as cv

def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames_1 = cv.absdiff(next_frame, cur_frame)
    diff_frames_2 = cv.absdiff(cur_frame, prev_frame)
    return cv.bitwise_and(diff_frames_1, diff_frames_2)

def get_frame(cap, scaling_factor):
    _, frame = cap.read()
    frame = cv.resize(frame, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv.INTER_AREA)
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    return gray

if __name__=='__main__':
    cap = cv.VideoCapture(0)
    scaling_factor = 0.5
    prev_frame = get_frame(cap, scaling_factor)
    cur_frame = get_frame(cap, scaling_factor)
    next_frame = get_frame(cap, scaling_factor)
    while True:
        cv.imshow('Object Movement', frame_diff(prev_frame, cur_frame, next_frame))
        prev_frame = cur_frame
        cur_frame = next_frame 

        next_frame = get_frame(cap, scaling_factor)

        key = cv.waitKey(10)
        if key == 27:
            break

    cv.destroyAllWindows()
