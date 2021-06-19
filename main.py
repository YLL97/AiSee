"""
Main Program AIA
"""
import os
import sys
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QStackedWidget, QStatusBar, QWidget
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QPixmap, QImage
from gui import gui_main, gui_m1, gui_m2, gui_m3
import time
import numpy as np
import module1
import module2
import module3
from gui import audioio  # Sound Recognition
from queue import Queue
from globalobject import GlobalObject  # Custom Event (Gold!)


class Recogniser(QThread, audioio.MultithreadGetAudio):
    def __init__(self):
        super().__init__()

    def run(self):
        self.get_audio()  # Loop to recognise audio commands


class GlobalDispatcher(QThread):  # Thread to check voice commands
    def __init__(self, object):
        super().__init__()
        self.object = object

    def run(self):
        command_list = ['ok', 'clear', 'switch mode', 'open 1', 'open to', 'open 3', 'exit 1', 'exit 2', 'exit 3']
        dispatch_list = ['ok', 'clear', 'switch_mode', 'open_1', 'open_2', 'open_3', 'exit_1', 'exit_2', 'exit_3']
        while True:
            for command, dispatch in zip(command_list, dispatch_list):
                if self.object.received_text == command:
                    GlobalObject().dispatchEvent(dispatch)
                    self.object.clear_cache()
            time.sleep(0.1)  # Or else it will make ModuleThread runs slow!


class ModuleThread(QThread):
    def __init__(self, module, q_sigscan, q_imgpath=None, q_sigclear=None, q_checked=None):
        super().__init__()
        self.module = module
        self.q_sigscan = q_sigscan
        self.q_imgpath = q_imgpath
        self.q_sigclear = q_sigclear
        self.q_checked = q_checked
        # self.daemon = True
        self.once_flag = True  # Memory for single execution

    setfocus_request = pyqtSignal()
    # https://www.youtube.com/watch?v=G7ffF0U36b0
    showlabel_request = pyqtSignal(str)
    showobj_request = pyqtSignal(str)
    # https://stackoverflow.com/questions/43964766/pyqt-emit-signal-with-dict
    showobjdict_request = pyqtSignal(object)
    showocr_request = pyqtSignal(str)

    def check_scan_clicked(self, object):
        if self.q_sigscan.empty() == False:
            val = self.q_sigscan.get()
            object.get_sigscan()

    def check_import_clicked(self, object):
        if self.q_imgpath.empty() == False:
            val = self.q_imgpath.get()
            object.get_imgpath(val)  # Send path to object to process
            return True
        else: return False

    def check_clear_clicked(self, object):
        if self.q_sigclear.empty() == False:
            val = self.q_sigclear.get()  # Clear the queue
            object.get_sigclear()
            return True
        else: return False

    def check_checkbox_toggled(self, object):  # Applicable in module1 only
        if self.q_checked.empty() == False:
            val = self.q_checked.get()
            object.get_docmode(val)

    def check_quit(self, object):
        if q_quit.empty() == False:
            val = q_quit.get()
            object.quit = val

    def setfocustoside(self):
        if self.once_flag:  # Memory for single execution
            self.setfocus_request.emit()
            self.once_flag = False
        else: pass

    def run(self):  # Override the parent's run method
        if self.module == 1:
            objm1 = module1.main()  # Create Module1 Object
            while objm1.quit == False:
                while objm1.quit == False:  # Mode 1: Streaming (default)
                    self.check_quit(objm1)
                    self.check_checkbox_toggled(objm1)
                    self.check_scan_clicked(objm1)
                    textlist = objm1.run_main(1)  # Launch of cvWindow
                    self.setfocustoside()  # Set activated window back to side MainWindow after cvWindow openns
                    self.showocr_request.emit(textlist)  # self. must be put, DUN ASK WHY
                    if self.check_clear_clicked(objm1):
                        objm1.scanned = False  # reset scanned state!
                    if self.check_import_clicked(objm1):
                        break

                while objm1.quit == False:  # Mode 2: Import
                    self.check_quit(objm1)
                    self.check_checkbox_toggled(objm1)
                    self.check_scan_clicked(objm1)
                    textlist = objm1.run_main(2)
                    self.showocr_request.emit(textlist)
                    self.check_import_clicked(objm1)  # If import again new img path will be updated and replace the previous one
                    if self.check_clear_clicked(objm1):
                        objm1.scanned = False
                        break


        if self.module == 2:
            objm2 = module2.main()
            self.setfocustoside()
            while objm2.quit != True:
                self.check_quit(objm2)
                self.check_scan_clicked(objm2)
                label = objm2.run_main()
                self.showlabel_request.emit(label)


        if self.module == 3:
            objm3 = module3.main()
            while objm3.quit == False:
                while objm3.quit == False:
                    self.check_quit(objm3)
                    self.check_scan_clicked(objm3)
                    obj = objm3.run_main()
                    self.setfocustoside()
                    self.showobj_request.emit(obj)  # self. must be put, DUN ASK WHY
                    if self.check_import_clicked(objm3):   # Send path to object to process
                        objm3.import_infer()  # Infer from the img path given
                        break

                while objm3.quit == False:
                    self.check_quit(objm3)
                    self.check_scan_clicked(objm3)
                    objdict = objm3.run_main()
                    self.showobjdict_request.emit(objdict)
                    if self.check_import_clicked(objm3):
                        objm3.import_infer()
                    if self.check_clear_clicked(objm3):  # if clear is pressed then quit
                        break
                if objm3.quit: break


# https://www.geeksforgeeks.org/python-call-parent-class-method/
class Ui_Main(QMainWindow, gui_main.Ui_MainWindow):
    # Just dont ask why
    def __init__(self):
        super().__init__()  # QtWidgets.QMainWindow.__init__(self) works the same (but works a bit differently on multiple inheritance)

        self.setupUi(self)  # Call Qt Designer generated function, introducing related Qt objects into current scope
        root = os.path.abspath(os.path.join(__file__, ".."))
        os.chdir(root)  # Reset working directory to root because module3 has changed it during import
        self.logo_1.setPixmap(QPixmap("gui/src/logo_ocr.png"))
        self.logo_2.setPixmap(QPixmap("gui/src/logo_rm.png"))
        self.logo_3.setPixmap(QPixmap("gui/src/logo_od.png"))
        self.actionQuit.triggered.connect(self.closeEvent)
        self.pushButton_1.clicked.connect(self.open_module1)
        self.pushButton_2.clicked.connect(self.open_module2)
        self.pushButton_3.clicked.connect(self.open_module3)
        GlobalObject().addEventListener("open_1", self.open_module1)
        GlobalObject().addEventListener("open_2", self.open_module2)
        GlobalObject().addEventListener("open_3", self.open_module3)


    def keyPressEvent(self, event):
        if event.key() == Qt.Key_1:
            self.open_module1()
        if event.key() == Qt.Key_2:
            self.open_module2()
        if event.key() == Qt.Key_3:
            self.open_module3()
        if event.key() == Qt.Key_R:
            listener.get_audio()

    def open_module1(self):
        if not (M1Window.isVisible() or M2Window.isVisible() or M3Window.isVisible()):
            M1Window.launchthread()  # Thread will send show window signal

    def open_module2(self):
        if not (M1Window.isVisible() or M2Window.isVisible() or M3Window.isVisible()):
            M2Window.launchthread()

    def open_module3(self):
        if not (M1Window.isVisible() or M2Window.isVisible() or M3Window.isVisible()):
            M3Window.launchthread()

    def closeEvent(self, event):
        print('AiSee Exit')
        sys.exit()

class Ui_M1(QMainWindow, gui_m1.Ui_Module1):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.move(int(1920 / 2 - W / 2) - 650, int(1080 / 2 - H / 2))
        # Push button clicked signals to events/functions
        self.pushButton_scan.clicked.connect(self.scan_clicked)
        # if import_clicked(), the connected function will be invoked first even before the click/signal!
        self.pushButton_import.clicked.connect(self.import_clicked)
        self.pushButton_clear.clicked.connect(self.clear_clicked)
        self.checkBox.stateChanged.connect(self.checkbox_toggle)
        GlobalObject().addEventListener("ok", self.scan_clicked)
        GlobalObject().addEventListener("clear", self.clear_clicked)
        GlobalObject().addEventListener("switch_mode", self.checkbox_toggle2)
        GlobalObject().addEventListener("exit_1", self.quitall)

        # Queues for GUI to thread interface (Act as a middle man for information transfer)
        self.q_sigscan = Queue()  # GUI scan input
        self.q_imgpath = Queue()  # GUI import input
        self.q_sigclear = Queue()  # GUI clear input
        self.q_checked = Queue()  # GUI checkbox
        self.toggle = True  # For handling checkbox toggle mem


    def launchthread(self):
        # Why need put self infront??
        self.thread = ModuleThread(1, self.q_sigscan, self.q_imgpath, self.q_sigclear, self.q_checked)
        self.thread.start()

        # Signal-slot for thread to GUI interface
        self.thread.setfocus_request.connect(self.setfocus)
        self.thread.showocr_request.connect(self.showocr)
        self.thread.finished.connect(self.threadfinished)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.scan_clicked()
        if event.key() == Qt.Key_X:
            self.import_clicked()
        if event.key() == Qt.Key_C:
            self.clear_clicked()
        if event.key() == Qt.Key_Q:
            self.quitall()
# https://stackoverflow.com/questions/55842175/is-the-a-way-to-check-a-qcheckbox-with-a-key-press
        if event.key() == Qt.Key_Z:
            self.checkBox.nextCheckState()
        if event.key() == Qt.Key_R:
            listener.get_audio()

    def threadfinished(self):
        self.textBrowser.clear()
        self.close()  # Close M3 side window

    def scan_clicked(self):
        if self.isActiveWindow():  # When the window is active then only execute (check this to prevent background execution)
            if self.q_sigscan.empty():
                self.q_sigscan.put(True)

    def showocr(self, text):  # Invoked constantly when in mode1
        if text != '':  # Whenever return value from thread is emitted, the particular loop in the thread it will break, subsequent return will be ''
            self.textBrowser.setFontPointSize(12)
            self.textBrowser.clear()
            self.textBrowser.append(text)

    def import_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Import Image File', r"D:/Users/Leong/Pictures",
                                                   "Image files (*.jpg *.jpeg *.png)")
        if bool(file_name):  # Check if file is not empty (cancel import)
            if self.q_imgpath.empty():
                self.q_imgpath.put(file_name)
                self.textBrowser.clear()  # Clear previous textbrowser data once image is loaded

    def clear_clicked(self):
        self.textBrowser.clear()  # Clear browser info whenever clear is clicked
        self.q_sigclear.put(True)
        # print('q_sigscan: ', list(self.q_sigscan.queue))

    def checkbox_toggle(self):
        if self.q_checked.empty():
            if self.toggle:
                self.q_checked.put(True)
                self.toggle = False
            else:
                self.q_checked.put(False)
                self.toggle = True

    def checkbox_toggle2(self):
        self.checkBox.nextCheckState()

    def quitall(self):
        if q_quit.empty():
            q_quit.put(True)

    def setfocus(self):
        self.show()
        self.activateWindow()  # Turn the window focus to the side MainWindow (instead of cv window)


class Ui_M2(QMainWindow, gui_m2.Ui_Module2):
    def __init__(self):
        super().__init__()  # QtWidgets.QMainWindow.__init__(self) works the same (but works a bit differently on multiple inheritance)
        self.setupUi(self)
        self.move(int(1920/2 - W/2) - 650, int(1080/2 - H/2))
        self.pushButton_scan.clicked.connect(self.scan_clicked)
        GlobalObject().addEventListener("ok", self.scan_clicked)
        GlobalObject().addEventListener("exit_2", self.quitall)

        self.q_sigscan = Queue()
        self.label_rm.setText("")

    def launchthread(self):
        self.thread = ModuleThread(2, self.q_sigscan)
        self.thread.start()

        self.thread.setfocus_request.connect(self.setfocus)
        self.thread.showlabel_request.connect(self.showlabel)
        self.thread.finished.connect(self.threadfinished)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.scan_clicked()
        if event.key() == Qt.Key_Q:
            self.quitall()
        if event.key() == Qt.Key_R:
            listener.get_audio()

    def threadfinished(self):
        self.label_rm.setText("")  # Cleara label before exiting
        self.close()  # Close M2 side window

    def scan_clicked(self):
        if self.isActiveWindow():
            if self.q_sigscan.empty():
                self.q_sigscan.put(True)

    def showlabel(self, label):
        if label != '':  # if there IS label return
            prevlabel = label
            self.label_rm.setText(prevlabel)

    def quitall(self):
        if q_quit.empty():
            q_quit.put(True)
        listener.clear_cache()

    def setfocus(self):
        self.show()
        self.activateWindow()  # Turn the window focus to the side MainWindow (instead of cv window)


class Ui_M3(QMainWindow, gui_m3.Ui_Module3):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.move(int(1920/2 - W/2) - 650, int(1080/2 - H/2))
        self.pushButton_scan.clicked.connect(self.scan_clicked)
        # if import_clicked(), the connected function will be invoked first even before the click/signal!
        self.pushButton_import.clicked.connect(self.import_clicked)
        self.pushButton_clear.clicked.connect(self.clear_clicked)
        GlobalObject().addEventListener("ok", self.scan_clicked)
        GlobalObject().addEventListener("clear", self.clear_clicked)
        GlobalObject().addEventListener("exit_3", self.quitall)

        # Queues for GUI to thread interface (Act as a middle man for information transfer)
        self.q_sigscan = Queue()  # GUI scan input
        self.q_imgpath = Queue()  # GUI import input
        self.q_sigclear = Queue()  # GUI clear input


    def launchthread(self):
        # Why need put self infront??
        self.thread = ModuleThread(3, self.q_sigscan, self.q_imgpath, self.q_sigclear)
        self.thread.start()

        # Signal-slot for thread to GUI interface
        self.thread.setfocus_request.connect(self.setfocus)
        self.thread.showobj_request.connect(self.showobj)  # looping when in mode1
        self.thread.showobjdict_request.connect(self.showobjdict)  # looping when in mode2
        self.thread.finished.connect(self.threadfinished)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.scan_clicked()
        if event.key() == Qt.Key_X:
            self.import_clicked()
        if event.key() == Qt.Key_C:
            self.clear_clicked()
        if event.key() == Qt.Key_Q:
            self.quitall()
        if event.key() == Qt.Key_R:
            listener.get_audio()

    def threadfinished(self):
        self.textBrowser.clear()
        self.close()  # Close M3 side window

    def scan_clicked(self):
        if self.isActiveWindow():
            if self.q_sigscan.empty():
                self.q_sigscan.put(True)

    def showobj(self, obj):  #  Invoked constantly when in mode1
        if obj != '':
            self.textBrowser.setFontPointSize(24)
            self.textBrowser.clear()
            self.textBrowser.append(obj)

    def showobjdict(self, objdict):  # Invoked constantly when in mode2
        self.textBrowser.setFontPointSize(24)
        # self.textBrowser.clear() # Cannot put here due to this function is being called constantly! Will overwrite the newly printed text!
        if objdict != None:
            if bool(objdict):
                self.textBrowser.clear()
                for key in objdict:  # For dictionary, loop only refers key!
                    info = key + ': ' + str(objdict[key])  # Value is referred using key!
                    self.textBrowser.append(info)
            else:
                self.textBrowser.clear()
                self.textBrowser.append("No Object Detected")

    def import_clicked(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Import Image File', r"D:/Users/Leong/Pictures",
                                                   "Image files (*.jpg *.jpeg *.png)")
        if file_name:  # Check if file is not empty (cancel import)
            if self.q_imgpath.empty():
                self.q_imgpath.put(file_name)
                self.textBrowser.clear()  # Clear previous textbrowser data once image is loaded

    def clear_clicked(self):
        self.textBrowser.clear()  # Clear browser info whenever clear is clicked
        self.q_sigclear.put(True)


    def quitall(self):
        if q_quit.empty():
            q_quit.put(True)

    def setfocus(self):
        self.show()
        self.activateWindow()  # Turn the window focus to the side MainWindow (instead of cv window)

def main():
    pass

if __name__ == "__main__":
    W = 340  # Secondary MainWindow width
    H = 845  # Secondary MainWindow height**
    main()
    app = QApplication(sys.argv)
    q_quit = Queue()
    # ScreenManager = QStackedWidget()
    MainWindow = Ui_Main()
    M1Window = Ui_M1()
    M2Window = Ui_M2()
    M3Window = Ui_M3()
    listener = Recogniser()  # Create global audio recognition object
    globaldisipatcher = GlobalDispatcher(listener)  # Checks listener.text_received constantly and dispatch signal if matched
    globaldisipatcher.start()  # Start the looping thread
    # ScreenManager.addWidget(MainWindow)
    # ScreenManager.addWidget((Screen2))
    # ScreenManager.setFixedSize(1280, 720)
    # ScreenManager.show()
    MainWindow.show()
    sys.exit(app.exec_())
