import math
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt
import cv2 as cv
import numpy as np
import concurrent.futures
import time
import matplotlib.pyplot as plt
from threading import Lock


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1300, 525)
        MainWindow.setMouseTracking(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_Original = QtWidgets.QLabel(self.centralwidget,
                                               alignment=Qt.AlignCenter)
        self.label_Original.setMinimumSize(QtCore.QSize(450, 450))
        self.label_Original.setStyleSheet("background-color: rgb(244, 244, 244);\n"
                                          "\n"
                                          "")
        self.label_Original.setText("")
        self.label_Original.setObjectName("label_Original")
        self.horizontalLayout_2.addWidget(self.label_Original)
        self.label_Edit = QtWidgets.QLabel(self.centralwidget,
                                           alignment=Qt.AlignCenter)
        self.label_Edit.setMinimumSize(QtCore.QSize(450, 450))
        self.label_Edit.setStyleSheet("background-color: rgb(244, 244, 244);\n"
                                      "")
        self.label_Edit.setText("")
        self.label_Edit.setObjectName("label_Edit")
        self.horizontalLayout_2.addWidget(self.label_Edit)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton_Dithering = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_Dithering.setMinimumSize(QtCore.QSize(270, 50))
        self.pushButton_Dithering.setMaximumSize(QtCore.QSize(270, 50))
        self.pushButton_Dithering.setStyleSheet(
            "font: 10pt \"MS Shell Dlg 2\";")
        self.pushButton_Dithering.setObjectName("pushButton_Dithering")
        self.verticalLayout.addWidget(self.pushButton_Dithering)
        self.pushButton_Otsu_Binarization_CV = QtWidgets.QPushButton(
            self.centralwidget)
        self.pushButton_Otsu_Binarization_CV.setMinimumSize(
            QtCore.QSize(270, 50))
        self.pushButton_Otsu_Binarization_CV.setMaximumSize(
            QtCore.QSize(270, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Otsu_Binarization_CV.setFont(font)
        self.pushButton_Otsu_Binarization_CV.setObjectName(
            "pushButton_Otsu_Binarization_CV")
        self.verticalLayout.addWidget(self.pushButton_Otsu_Binarization_CV)
        self.pushButton_Otsu_Binarization = QtWidgets.QPushButton(
            self.centralwidget)
        self.pushButton_Otsu_Binarization.setMinimumSize(QtCore.QSize(270, 50))
        self.pushButton_Otsu_Binarization.setMaximumSize(QtCore.QSize(270, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Otsu_Binarization.setFont(font)
        self.pushButton_Otsu_Binarization.setObjectName(
            "pushButton_Otsu_Binarization")
        self.verticalLayout.addWidget(self.pushButton_Otsu_Binarization)
        self.checkBox_Binarization = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.checkBox_Binarization.sizePolicy().hasHeightForWidth())
        self.checkBox_Binarization.setSizePolicy(sizePolicy)
        self.checkBox_Binarization.setMinimumSize(QtCore.QSize(270, 50))
        self.checkBox_Binarization.setMaximumSize(QtCore.QSize(270, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.checkBox_Binarization.setFont(font)
        self.checkBox_Binarization.setObjectName("checkBox_Binarization")
        self.verticalLayout.addWidget(self.checkBox_Binarization)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalSlider_Threshold = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider_Threshold.setEnabled(False)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.horizontalSlider_Threshold.sizePolicy().hasHeightForWidth())
        self.horizontalSlider_Threshold.setSizePolicy(sizePolicy)
        self.horizontalSlider_Threshold.setMinimumSize(QtCore.QSize(210, 15))
        self.horizontalSlider_Threshold.setMaximumSize(QtCore.QSize(210, 30))
        self.horizontalSlider_Threshold.setStyleSheet("")
        self.horizontalSlider_Threshold.setMinimum(1)
        self.horizontalSlider_Threshold.setMaximum(255)
        self.horizontalSlider_Threshold.setTracking(True)
        self.horizontalSlider_Threshold.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Threshold.setObjectName(
            "horizontalSlider_Threshold")
        self.horizontalLayout.addWidget(self.horizontalSlider_Threshold)
        self.spinBox_Threshold = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_Threshold.setMinimumSize(QtCore.QSize(50, 35))
        self.spinBox_Threshold.setMaximumSize(QtCore.QSize(50, 35))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.spinBox_Threshold.setFont(font)
        self.spinBox_Threshold.setMinimum(1)
        self.spinBox_Threshold.setMaximum(255)
        self.spinBox_Threshold.setObjectName("spinBox_Threshold")
        self.spinBox_Threshold.setEnabled(False)
        self.horizontalLayout.addWidget(self.spinBox_Threshold)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.pushButton_Compare_methods = QtWidgets.QPushButton(
            self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.pushButton_Compare_methods.sizePolicy().hasHeightForWidth())
        self.pushButton_Compare_methods.setSizePolicy(sizePolicy)
        self.pushButton_Compare_methods.setMinimumSize(QtCore.QSize(270, 50))
        self.pushButton_Compare_methods.setMaximumSize(QtCore.QSize(270, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.pushButton_Compare_methods.setFont(font)
        self.pushButton_Compare_methods.setObjectName(
            "pushButton_Compare_methods")
        self.verticalLayout.addWidget(self.pushButton_Compare_methods)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1208, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        self.actionOpen.triggered.connect(self.loadImage)
        self.actionSave.triggered.connect(self.savePhoto)
        self.actionExit.triggered.connect(sys.exit)
        self.pushButton_Dithering.clicked.connect(
            self.pushButton_Dithering_Clicked)
        self.pushButton_Otsu_Binarization_CV.clicked.connect(
            self.pushButton_Otsu_Binarization_CV_Clicked)
        self.pushButton_Otsu_Binarization.clicked.connect(
            self.pushButton_Otsu_Binarization_Clicked)
        self.pushButton_Compare_methods.clicked.connect(
            self.pushButton_Compare_methods_Clicked)
        self.checkBox_Binarization.clicked['bool'].connect(
            self.horizontalSlider_Threshold.setEnabled)
        self.checkBox_Binarization.clicked['bool'].connect(
            self.spinBox_Threshold.setEnabled)
        self.checkBox_Binarization.clicked['bool'].connect(
            self.pushButton_Dithering.setDisabled)
        self.checkBox_Binarization.clicked['bool'].connect(
            self.pushButton_Otsu_Binarization_CV.setDisabled)
        self.checkBox_Binarization.clicked['bool'].connect(
            self.pushButton_Otsu_Binarization.setDisabled)
        self.checkBox_Binarization.clicked.connect(
            self.label_Edit.clear)
        self.horizontalSlider_Threshold.valueChanged['int'].connect(
            self.horizontalSlider_Threshold_scrolled)
        self.horizontalSlider_Threshold.valueChanged['int'].connect(
            self.spinBox_Threshold.setValue)
        self.spinBox_Threshold.valueChanged['int'].connect(
            self.horizontalSlider_Threshold.setValue)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # start
        self.filename = None
        self.filename_save = None
        self.result = None

    def pushButton_Compare_methods_Clicked(self):
        try:
            rows = 2
            columns = 2
            fig = plt.figure(figsize=(10, 7))

            fig.add_subplot(rows, columns, 1)

            b, g, r = cv.split(self.image)
            rgb_img = cv.merge([r, g, b])

            plt.imshow(rgb_img)
            plt.axis('off')
            plt.title("Original")

            res = self.dithering(self.image)
            fig.add_subplot(rows, columns, 2)
            plt.imshow(res, cmap='gray')
            plt.axis('off')
            plt.title("Dithering")

            res, threshold = self.otsu_binarization_thread(self.image)
            fig.add_subplot(rows, columns, 3)
            plt.imshow(res, cmap='gray')
            plt.axis('off')
            plt.title("Otsu\nThreshold is {}".format(threshold))

            res, threshold = self.otsu_binarization_cv(self.image)
            fig.add_subplot(rows, columns, 4)
            plt.imshow(res, cmap='gray')
            plt.axis('off')
            plt.title("Otsu OpenCV\nThreshold is {}".format(int(threshold)))

            plt.show()
        except:
            self.checkImage()

    def pushButton_Dithering_Clicked(self):
        try:
            self.result = self.dithering(self.image)
            # self.result = cv.resize(self.image, (1920, 1080), interpolation=cv.INTER_AREA)
            tmp = self.checkSizeImage(self.result)
            self.setPhoto(tmp, 0)
        except:
            self.checkImage()

    def pushButton_Otsu_Binarization_CV_Clicked(self):
        try:
            self.result, threshold = self.otsu_binarization_cv(self.image)
            self.showThreshold(threshold)
            tmp = self.checkSizeImage(self.result)
            self.setPhoto(tmp, 0)
        except:
            self.checkImage()

    def pushButton_Otsu_Binarization_Clicked(self):
        try:
            # self.result, threshold = self.otsu_binarization_thread(self.image)
            self.result, threshold = self.otsu_binarization(self.image)
            self.showThreshold(threshold)
            tmp = self.checkSizeImage(self.result)
            self.setPhoto(tmp, 0)
        except:
            self.checkImage()

    def horizontalSlider_Threshold_scrolled(self, value):
        try:
            self.result = self.binarization(self.image, value)
            tmp = self.checkSizeImage(self.result)
            self.setPhoto(tmp, 0)
        except:
            self.checkImage()

    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        try:
            self.label_Edit.clear()
            self.result = None
            self.filename = QFileDialog.getOpenFileName(
                filter="Image (*.*)")[0]
            str(self.filename)
            f = open(self.filename, "rb")
            chunk = f.read()
            chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
            self.image = cv.imdecode(chunk_arr, cv.IMREAD_COLOR)
            f.close()
            tmp = self.checkSizeImage(self.image)
            self.setPhoto(tmp)
        except:
            None

    def setPhoto(self, image, check=1):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        frame = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = QImage(
            frame,
            frame.shape[1],
            frame.shape[0],
            frame.strides[0],
            QImage.Format_RGB888)
        if check == 1:
            self.label_Original.setPixmap(QtGui.QPixmap.fromImage(image))
        else:
            self.label_Edit.setPixmap(QtGui.QPixmap.fromImage(image))

    def savePhoto(self):
        """ This function will save the result.
        """
        try:
            self.filename_save = QFileDialog.getSaveFileName(
                filter="png Image (*.png)")[0]
            _, im_buf_arr = cv.imencode(".jpg", self.result)
            im_buf_arr.tofile(self.filename_save)
        except:
            self.errorSavePhoto()

    def showThreshold(self, threshold):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Threshold")
        msg.setText("Threshold is {}".format(threshold))
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def errorSavePhoto(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("ERROR")
        msg.setText(
            "You have not edited the image or you have not uploaded the image!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def checkImage(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("ERROR")
        msg.setText("You have not select a file!!!")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def checkSizeImage(self, image):
        """ This function will check the image's size
            and scale the image to fit window
        """
        height, width = image.shape[:2]
        if height >= width:
            k = self.label_Original.height() / height
            height = int(image.shape[0]*k)
            width = int(image.shape[1]*k)
        elif width > height:
            k = self.label_Original.width() / width
            height = int(image.shape[0]*k)
            width = int(image.shape[1]*k)
        math.floor(height)
        math.floor(width)
        return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

    # Dithering
    def dithering(self, img):
        GrayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        Height, Width = GrayImage.shape[:2]

        start = time.perf_counter()

        for y in range(0, Height):
            for x in range(0, Width):
                old_value = GrayImage[y, x]
                new_value = 0
                if (old_value > 128):
                    new_value = 255

                GrayImage[y, x] = new_value

                Error = old_value - new_value

                if (x < Width-1):
                    NewNumber = GrayImage[y, x+1] + Error * 7 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y, x+1] = NewNumber

                if (x > 0 and y < Height-1):
                    NewNumber = GrayImage[y+1, x-1] + Error * 3 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y+1, x-1] = NewNumber

                if (y < Height-1):
                    NewNumber = GrayImage[y+1, x] + Error * 5 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y+1, x] = NewNumber

                if (y < Height-1 and x < Width-1):
                    NewNumber = GrayImage[y+1, x+1] + Error * 1 / 16
                    if (NewNumber > 255):
                        NewNumber = 255
                    elif (NewNumber < 0):
                        NewNumber = 0
                    GrayImage[y+1, x+1] = NewNumber

        print(f'Finished', time.perf_counter()-start)
        # return res

        return GrayImage

    # Otsu Binarization OpenCV
    def otsu_binarization_cv(self, img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        threshold, binary = cv.threshold(
            gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        return binary, threshold

    # Otsu Binarization
    def otsu_binarization(self, img):
        H, W = img.shape[:2]
        # BGR2GRAY
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()
        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        max_sigma = 0
        max_t = 0

        # determine threshold
        # w0, w1 -- Это отношение количества пикселей в двух классах, разделенных порогом, к общему количеству пикселей.
        # m0, m1 -- Это средние значения значений пикселей этих двух категорий.

        start = time.perf_counter()
        for _t in range(1, 256):
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H * W)
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H * W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)

            if sigma > max_sigma:
                max_sigma = sigma
                max_t = _t

        # Binarization
        out[out < max_t] = 0
        out[out >= max_t] = 255

        print(f'Finished', time.perf_counter()-start)

        return out, max_t

    # Otsu Binarization thread
    def otsu_binarization_thread(self, img):
        H, W = img.shape[:2]
        # BGR2GRAY
        b = img[:, :, 0].copy()
        g = img[:, :, 1].copy()
        r = img[:, :, 2].copy()
        # Gray scale
        out = 0.2126 * r + 0.7152 * g + 0.0722 * b
        out = out.astype(np.uint8)

        self.max_sigma = 0
        self.max_t = 0

        l = Lock()

        def _binarization(_t):
            v0 = out[np.where(out < _t)]
            m0 = np.mean(v0) if len(v0) > 0 else 0.
            w0 = len(v0) / (H * W)
            v1 = out[np.where(out >= _t)]
            m1 = np.mean(v1) if len(v1) > 0 else 0.
            w1 = len(v1) / (H * W)
            sigma = w0 * w1 * ((m0 - m1) ** 2)

            with l:
                if sigma > self.max_sigma:
                    self.max_sigma = sigma
                    self.max_t = _t

        start = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for _t in range(1, 256):
                executor.submit(_binarization, _t)

        print(f'Finished', time.perf_counter()-start)

        # Binarization
        out[out < self.max_t] = 0
        out[out >= self.max_t] = 255

        return out, self.max_t

    # Binarization
    def binarization(self, img, a):
        # BGR2GRAY
        out = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        out[out < a] = 0
        out[out >= a] = 255

        return out

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_Dithering.setText(
            _translate("MainWindow", "Dithering"))
        self.pushButton_Otsu_Binarization_CV.setText(
            _translate("MainWindow", "Otsu\'s binarization CV"))
        self.pushButton_Otsu_Binarization.setText(
            _translate("MainWindow", "Otsu\'s binarization"))
        self.checkBox_Binarization.setText(
            _translate("MainWindow", "Binarization"))
        self.pushButton_Compare_methods.setText(
            _translate("MainWindow", "Compare methods"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionOpen.setShortcut(_translate("MainWindow", "Ctrl+O"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionSave.setShortcut(_translate("MainWindow", "Ctrl+S"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Esc"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
