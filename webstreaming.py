# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
from cv2 import aruco

from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

# termination criteria for the iterative algorithm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# File storage in OpenCV
cv_file = cv2.FileStorage("calib_images/test.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type
# to retrieve otherwise we only get a 'None'
# FileNode object back instead of a matrix
mtx = cv_file.getNode("camera_matrix").mat()
dist = cv_file.getNode("dist_coeff").mat()

cv_file.release()

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion(frameCount):
	# grab global references to the video stream, output frame, and
	# lock variables
	global vs, outputFrame, lock

	times = []

	# loop over frames from the video stream
	while True:
		# Start time
		start = time.time()

		ret, frame = vs.read()

		# operations on the frame
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# set dictionary size depending on the aruco marker selected
		aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

		# detector parameters can be set here (List of detection parameters[3])
		parameters = aruco.DetectorParameters_create()
		parameters.adaptiveThreshConstant = 10

		# lists of ids and the corners belonging to each id
		corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

		# font for displaying text (below)
		font = cv2.FONT_HERSHEY_SIMPLEX

		# check if the ids list is not empty
		# if no check is added the code will crash
		if np.all(ids != None):

			# estimate pose of each marker and return the values
			# rvet and tvec-different from camera coefficients
			rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
			# (rvec-tvec).any() # get rid of that nasty numpy value array error

			for i in range(0, ids.size):
				# draw axis for the aruco markers
				aruco.drawAxis(frame, mtx, dist, rvec[i], tvec[i], 0.1)

			# draw a square around the markers
			aruco.drawDetectedMarkers(frame, corners)

			# code to show ids of the marker found
			strg = ''
			for i in range(0, ids.size):
				strg += str(ids[i][0]) + ', '

			cv2.putText(frame, "Id: " + strg, (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)


		else:
			# code to show 'No Ids' when no markers are found
			cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

		end = time.time()

		times.append(end - start)

		if len(times) > 100:
			times = times[:99]

		cv2.putText(frame, f"FPS: {len(times) / sum(times)}", (0, 100), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

		# acquire the lock, set the output frame, and release the
		# lock
		with lock:
			outputFrame = frame.copy()
		
def generate():
	# grab global references to the output frame and lock variables
	global outputFrame, lock

	# loop over frames from the output stream
	while True:
		# wait until the lock is acquired
		with lock:
			# check if the output frame is available, otherwise skip
			# the iteration of the loop
			if outputFrame is None:
				continue

			# encode the frame in JPEG format
			(flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

			# ensure the frame was successfully encoded
			if not flag:
				continue

		# yield the output frame in the byte format
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--ip", type=str, required=True,
		help="ip address of the device")
	ap.add_argument("-o", "--port", type=int, required=True,
		help="ephemeral port number of the server (1024 to 65535)")
	ap.add_argument("-f", "--frame-count", type=int, default=32,
		help="# of frames used to construct the background model")
	args = vars(ap.parse_args())

	# start a thread that will perform motion detection
	t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],))
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host=args["ip"], port=args["port"], debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()