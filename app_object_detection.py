import sys
import cv2
import numpy as np
from PyQt5 import QtCore as qtc
from PyQt5 import QtGui as qtg
from PyQt5 import QtWidgets as qtw
from PyQt5 import uic

# ---------------------- Main Window Class ---------------------- #
class MainWindow(qtw.QWidget):
	def __init__(self):
		super().__init__()

		# your code goes here
		# Load ui from qtdesigner
		self.ui = uic.loadUi("main_window.ui", self)
		self.setWindowTitle("Intelligent Video Surveillence")
		self.ui.start_stream.clicked.connect(self.start_stream_slot)
		self.ui.stop_stream.clicked.connect(self.stop_stream_slot)


	# Start stream from the camera
	def start_stream_slot(self):
		print("camera stream started********")
		self.worker = Worker()
		self.worker.start()
		self.worker.get_frame.connect(self.display_frame)
		self.worker.get_frame_detected.connect(self.display_frame_detected)
		self.worker.get_threat_area.connect(self.display_threat_area)
		
	# Stop stream from the camera
	def stop_stream_slot(self):
		print("camera released********")
		self.worker.stop()

	# Display opencv frame to pyqt5 mainwindow
	def display_frame(self, img):
		img_to_pixmap = qtg.QPixmap(img)
		self.ui.img_label.setPixmap(img_to_pixmap)

	# Display opencv frame to pyqt5 mainwindow on detected label
	def display_frame_detected(self, img, class_detected):
		img_to_pixmap = qtg.QPixmap(img)
		if class_detected != "":
			self.ui.detected_object.setText("Possible " + class_detected)
		else:
			self.ui.detected_object.setText("No threats detected" + class_detected)
		self.ui.detected_label.setPixmap(img_to_pixmap)
		print(f"Detected type: {class_detected}")

	def display_threat_area(self, img):
		img_to_pixmap = qtg.QPixmap(img)
		self.ui.threat_area_label.setPixmap(img_to_pixmap)





# ---------------------- Worker Thread Class ---------------------- #
class Worker(qtc.QThread):
	
	# Signal to send to main thread
	get_frame = qtc.pyqtSignal(qtg.QImage)
	get_frame_detected = qtc.pyqtSignal(qtg.QImage, str)
	get_threat_area = qtc.pyqtSignal(qtg.QImage)


	# Main run function
	def run(self):
		self.ThreadActive = True

		############################ LOAD MODEL ############################

		# opencv deep neural network
		net = cv2.dnn.readNet('./dnn_model/yolov4-tiny.weights', './dnn_model/yolov4-tiny.cfg')
		model = cv2.dnn_DetectionModel(net)
		model.setInputParams(size=(320, 320), scale=1/255)

		# Load class list
		classes = []

		with open('./dnn_model/classes.txt', 'rt') as f:
			for class_name in f.readlines():
				class_name = class_name.strip()
				classes.append(class_name)

		####################################################################

		# Opencv video capture
		cap = cv2.VideoCapture(0)
		cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
		cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)

		if not cap.isOpened():
			print("Cannot open camera***")
			exit()

		# open cv capture loop
		while self.ThreadActive and cap.isOpened():
			ret, img = cap.read()
			img_detected = np.zeros((200, 200, 3), dtype=np.uint8)
			img_copy = np.zeros((640, 480, 3), dtype=np.uint8)
			if ret:
				img_copy = img.copy()
				############################ OBJECT DETECTION ############################
				
				class_ids, scores, bboxes = model.detect(img)
				
				# reqired outside for loop
				index = class_id = -1
				x = y = h = w = 0
				for class_id, score, bbox in zip(class_ids, scores, bboxes):
					index = class_id
					x, y, w, h = bbox
					print("points:", x, y, w, h)
					img_detected = img[y:y+h, x:x+h]
					img_detected = cv2.resize(img_detected, (200, 200))

				##########################################################################

				# For main frame
				cv2.putText(img, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)
				cv2.rectangle(img, (x, y), (x+w, y+h), (200, 0, 50), 3)

				qimg = qtg.QImage(img.data, img.shape[1], img.shape[0], 3*img.shape[1], qtg.QImage.Format.Format_RGB888).rgbSwapped()
				qimg_detected = qtg.QImage(img_detected.data, img_detected.shape[1], img_detected.shape[0], 3*img_detected.shape[1], qtg.QImage.Format.Format_RGB888).rgbSwapped()


				# Emiting main frame
				self.get_frame.emit(qimg)
				# Emitting sub frame
				if index == -1:
					# self.get_frame_detected.emit(qimg_detected, "")
					img_copy = cv2.resize(img_copy, (200, 200))
					img_copy = qtg.QImage(img_copy.data, img_copy.shape[1], img_copy.shape[0], 3*img_copy.shape[1], qtg.QImage.Format.Format_RGB888).rgbSwapped()
					self.get_frame_detected.emit(img_copy, "")
					self.get_threat_area.emit(img_copy)
				else:
					self.get_frame_detected.emit(qimg_detected, classes[index])
					# Threat area frame
					radius = (h + w) / 4
					img_copy = cv2.circle(img_copy, center=(int(x + radius), int(y + radius)), radius=int(radius), color=(0, 0, 255), thickness=2)
					img_copy = cv2.resize(img_copy, (200, 200))
					img_copy = qtg.QImage(img_copy.data, img_copy.shape[1], img_copy.shape[0], 3*img_copy.shape[1], qtg.QImage.Format.Format_RGB888).rgbSwapped()
					self.get_threat_area.emit(img_copy)

		cap.release()
		cv2.destroyAllWindows()


	# Closing the thread
	def stop(self):
		self.ThreadActive = False



# --------------------------------- main ------------------------------ #
def main():
	app = qtw.QApplication(sys.argv)
	main_window = MainWindow()
	main_window.show()
	sys.exit(app.exec())

if __name__ == "__main__":
	main()