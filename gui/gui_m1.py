# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui_m1.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Module1(object):
    def setupUi(self, Module1):
        Module1.setObjectName("Module1")
        Module1.resize(343, 720)
        self.centralwidget = QtWidgets.QWidget(Module1)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(10, 8, 320, 660))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_scan = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_scan.setGeometry(QtCore.QRect(110, 50, 100, 28))
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
        self.checkBox = QtWidgets.QCheckBox(self.groupBox)
        self.checkBox.setGeometry(QtCore.QRect(120, 95, 80, 20))
        self.checkBox.setObjectName("checkBox")
        Module1.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(Module1)
        self.statusbar.setObjectName("statusbar")
        Module1.setStatusBar(self.statusbar)

        self.retranslateUi(Module1)
        QtCore.QMetaObject.connectSlotsByName(Module1)

    def retranslateUi(self, Module1):
        _translate = QtCore.QCoreApplication.translate
        Module1.setWindowTitle(_translate("Module1", "Module1"))
        self.groupBox.setTitle(_translate("Module1", "Optical Character Recognition"))
        self.pushButton_scan.setText(_translate("Module1", "Scan"))
        self.pushButton_import.setText(_translate("Module1", "Import"))
        self.pushButton_clear.setText(_translate("Module1", "Clear"))
        self.checkBox.setText(_translate("Module1", "Document"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Module1 = QtWidgets.QMainWindow()
    ui = Ui_Module1()
    ui.setupUi(Module1)
    Module1.show()
    sys.exit(app.exec_())