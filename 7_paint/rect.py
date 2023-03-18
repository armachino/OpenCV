import sys
from PyQt5 import QtWidgets, QtCore 
from PyQt5.QtGui import QPainter

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys

class MyWidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(30,30,600,400)
        self.begin = QtCore.QPoint()
        self.end = QtCore.QPoint()
        self.drawing = False
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.brushColor = Qt.black
        self.brushSize = 2
        # making image color to white
        self.image.fill(Qt.white)
        self.UiComponents()
        self.shape="TRIANGLE"
        self.show()
        # button1 = QPushButton(self)
        # button1.setText("Button1")
        # button1.move(64,32)

    def UiComponents(self):
        # creating a push button
        rec_but = QPushButton("Rectangle", self)
        # setting geometry of button
        rec_but.setGeometry(0, 0, 100, 20)
        # adding action to a button
        rec_but.clicked.connect(self.select_rec)
        # creating a push button
        circule_but = QPushButton("Circle", self)
        # setting geometry of button
        circule_but.setGeometry(101, 0, 100, 20)
        # adding action to a button
        circule_but.clicked.connect(self.selec_circl)

    def select_rec(self):
        self.shape="TRIANGLE"
    def selec_circl(self):
        self.shape="CIRCULE"
    
    def clickme(self):
 
        # printing pressed
        print("pressed")        
    def paintEvent(self, event):
        self.drawing = True
        qp = QPainter(self)
        br = QBrush(QColor(100, 10, 10, 40))  
        qp.setBrush(br)
        if self.shape=="TRIANGLE":
            qp.drawRect(QtCore.QRect(self.begin, self.end))       
            canvasPainter = QPainter(self)
            canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        # else:
        #     qp.drawEllipse(self.begin, self.end)       
        #     canvasPainter = QPainter(self)
        #     canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        # painter = QPainter(self.image)
        # painter.setPen(QPen(self.brushColor, self.brushSize,
        #                     Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        # painter.drawRect(QtCore.QRect(self.begin, self.end)) 
        print("HIIIII")
        # draw rectangle  on the canvas

    # def paintEvent(self, event):
    #     qp = QPainter(self)
    #     br = QBrush(QColor(100, 10, 10, 40))  
    #     qp.setBrush(br)   
    #     qp.drawRect(QRect(self.begin, self.end))       
    #     # canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
    #     print("HI")
    
    def mousePressEvent(self, event):
        self.begin = event.pos()
        self.end = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        self.end = event.pos()
        self.update()

    # def mousePressEvent(self, event):
    #     self.drawing = True
    #     self.begin = event.pos()
    #     self.end = event.pos()
    #     self.update()

    # def mouseMoveEvent(self, event):
    #     painter = QPainter(self.image)
    #     painter.setPen(QPen(self.brushColor, self.brushSize,
    #                         Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    #     qp = QPainter(self)
    #     br = QBrush(QColor(100, 10, 10, 40))  
    #     qp.setBrush(br)   
    #     qp.drawRect(QtCore.QRect(self.begin, self.end))
        # painter.drawRect(QtCore.QRect(self.begin, self.end)) 
        # self.drawing = True
        # self.lastPoint = event.pos()

        # #     update
        # self.end = event.pos()
        # self.update()

    def mouseReleaseEvent(self, event):
        # canvasPainter = QPainter(self)
        # canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
        painter = QPainter(self.image)
        painter.setPen(QPen(self.brushColor, self.brushSize,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.shape=="TRIANGLE":
            painter.drawRect(QtCore.QRect(self.begin, self.end)) 
        # else:
        #     painter.drawEllipse(self.begin, self.end)
        # qp = QPainter(self)
        # br = QBrush(QColor(100, 10, 10, 40))  
        # qp.setBrush(br)   
        # qp.drawRect(QtCore.QRect(self.begin, self.end))
        self.drawing = True
        self.lastPoint = event.pos()

        self.begin = event.pos()
        self.end = event.pos()
        
        self.update()

    # def mouseReleaseEvent(self, event):
    #     self.begin = event.pos()
    #     self.end = event.pos()
    #     canvasPainter = QPainter(self)
    #     # draw rectangle  on the canvas
    #     canvasPainter.drawImage(self.rect(), self.image, self.image.rect())
    #     self.drawing = False

    #     self.update()

    # def paintEvent(self, event):
    #     # create a canvas
    #     canvasPainter = QPainter(self)
         
    #     # draw rectangle  on the canvas
    #     canvasPainter.drawImage(self.rect(), self.image, self.image.rect())

App = QtWidgets.QApplication(sys.argv)
 
# create the instance of our Window
window = MyWidget()
 
# showing the window
window.show()
 
# start the app
sys.exit(App.exec())
