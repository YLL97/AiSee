# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_m3.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Module3(object):
    def setupUi(self, Module3):
        Module3.setObjectName("Module3")
        Module3.resize(343, 720)
        self.centralwidget = QtWidgets.QWidget(Module3)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 8, 320, 660))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_scan = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_scan.setGeometry(QtCore.QRect(110, 80, 100, 28))
        self.pushButton_scan.setObjectName("pushButton_scan")
        self.pushButton_import = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_import.setGeometry(QtCore.QRect(40, 140, 100, 28))
        self.pushButton_import.setObjectName("pushButton_import")
        self.textBrowser = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser.setGeometry(QtCore.QRect(10, 200, 300, 400))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_clear = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_clear.setGeometry(QtCore.QRect(180, 140, 100, 28))
        self.pushButton_clear.setObjectName("pushButton_clear")
        Module3.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Module3)
        self.statusbar.setObjectName("statusbar")
        Module3.setStatusBar(self.statusbar)

        self.retranslateUi(Module3)
        QtCore.QMetaObject.connectSlotsByName(Module3)

    def retranslateUi(self, Module3):
        _translate = QtCore.QCoreApplication.translate
        Module3.setWindowTitle(_translate("Module3", "Module3"))
        self.groupBox.setTitle(_translate("Module3", "General Object Detection Module"))
        self.pushButton_scan.setText(_translate("Module3", "Scan"))
        self.pushButton_import.setText(_translate("Module3", "Import"))
        self.pushButton_clear.setText(_translate("Module3", "Clear"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Module3 = QtWidgets.QMainWindow()
    ui = Ui_Module3()
    ui.setupUi(Module3)
    Module3.show()
    sys.exit(app.exec_())
