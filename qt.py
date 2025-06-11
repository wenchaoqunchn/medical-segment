from PySide2.QtWidgets import *
from PySide2.QtCore import Qt
from PySide2.QtGui import QFont, QPixmap
import sys
from PySide2.QtCore import QEventLoop, QTimer
from PySide2 import QtCore, QtGui
from PySide2.QtCore import Slot


# 重定向信号
class EmittingStr(QtCore.QObject):
    textWritten = QtCore.Signal(str)  # 定义一个发送str的信号

    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QEventLoop()
        QTimer.singleShot(1000, loop.quit)
        loop.exec_()


def center(mainWindow, title, w, h):
    screen = QDesktopWidget().screenGeometry()
    mainWindow.setWindowTitle(title)
    mainWindow.resize(w, h)
    mainWindow.move((screen.width() - w) / 2, (screen.height() - h) / 2)


class Menu(QMainWindow):
    def __init__(self):
        super().__init__()

        font1 = QFont()
        font1.setFamilies(["Microsoft YaHei UI"])
        font1.setPointSize(16)

        center(self, "操作选择", 800, 600)

        self.pix = QPixmap("test.png")
        size = self.pix.size()

        self.label1 = QLabel(self)
        self.label1.setGeometry(0, 0, size.width(), size.height())
        self.label1.setPixmap(self.pix)

        lab = QLabel("基于Unet的医学图像分割系统")
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("QLabel{font-size:50px;font-family:微软雅黑;}")

        self.startBtn = QPushButton("开始")
        self.startBtn.clicked.connect(self.nextPage)
        self.startBtn.setFont(font1)

        sp = QSpacerItem(1, 150)
        vl = QVBoxLayout()
        vl.addWidget(lab)
        vl.addWidget(self.startBtn)
        vl.addItem(sp)

        w = QWidget()

        w.setLayout(vl)
        w.setContentsMargins(50, 30, 50, 100)

        self.setCentralWidget(w)

    def nextPage(self):
        self.selModel = SelModel()
        self.selModel.show()
        self.close()


class SelModel(QMainWindow):
    def __init__(self):
        super().__init__()
        center(self, "模型选择", 800, 600)

        self.pix = QPixmap("test.png")
        size = self.pix.size()

        self.label1 = QLabel(self)
        self.label1.setGeometry(0, 0, size.width(), size.height())
        self.label1.setPixmap(self.pix)

        lab = QLabel("选择模型")
        lab.setAlignment(Qt.AlignCenter)
        lab.setStyleSheet("QLabel{font-size:30px;font-family:微软雅黑;}")

        font1 = QFont()
        font1.setFamilies(["Microsoft YaHei UI"])
        font1.setPointSize(16)

        self.swinBtn = QPushButton("Swin U-NET")
        self.swinBtn.clicked.connect(self.swinFunc)
        self.swinBtn.setFont(font1)
        self.unetBtn = QPushButton("标准 U-NET")
        self.unetBtn.clicked.connect(self.unetFunc)
        self.unetBtn.setFont(font1)
        vl = QVBoxLayout()
        vl.addWidget(lab)
        vl.addWidget(self.swinBtn)
        vl.addWidget(self.unetBtn)

        w = QWidget()

        w.setLayout(vl)
        w.setContentsMargins(50, 30, 50, 100)

        self.setCentralWidget(w)

    def swinFunc(self):
        _model = "swin"
        self.nextPage()

    def unetFunc(self):
        _model = "unet"
        self.nextPage()

    def nextPage(self):
        self.next = Train()
        self.next.show()
        self.close()


class Train(QMainWindow):
    def __init__(self):
        super().__init__()
        center(self, "实时数据", 1500, 1000)
        lab = QLabel("数据集文件夹:")

        font1 = QFont()
        font1.setFamilies(["Microsoft YaHei UI"])
        font1.setPointSize(12)

        font2 = QFont()
        font2.setFamilies(["Microsoft YaHei UI"])
        font2.setPointSize(12)

        font3 = QFont()
        font3.setFamilies(["Consolas"])
        font3.setPointSize(12)

        hl = QHBoxLayout()
        lab.setFont(font1)
        self.path = QLineEdit()
        self.path.setFocusPolicy(Qt.NoFocus)
        self.path.setFont(font1)

        openBtn = QPushButton("打开")

        openBtn.setFont(font1)

        hl.addWidget(lab)
        hl.addWidget(self.path)
        hl.addWidget(openBtn)

        line = QLabel("注:文件夹内mask图像命名以_mask结尾")
        line.setFont(font1)
        self.startBtn = QPushButton("开始训练")
        self.startBtn.clicked.connect(self.train)
        self.text = QPlainTextEdit()
        self.text.setFont(font3)

        import sys

        sys.stdout = EmittingStr()
        self.text.connect(
            sys.stdout, QtCore.SIGNAL("textWritten(QString)"), self.outputWritten
        )
        sys.stderr = EmittingStr()
        self.text.connect(
            sys.stderr, QtCore.SIGNAL("textWritten(QString)"), self.outputWritten
        )

        font1 = QFont()
        font1.setFamilies(["Microsoft YaHei UI"])
        font1.setPointSize(16)
        self.startBtn.setFont(font1)

        res = QPushButton("查看图像结果")
        res.clicked.connect(self.openRes)
        res.setFont(font1)
        ch = QPushButton("查看分析图表")
        ch.clicked.connect(self.openCh)
        ch.setFont(font1)

        btns = QHBoxLayout()
        btns.addWidget(res)
        btns.addWidget(ch)

        vl = QVBoxLayout()
        vl.addLayout(hl)
        vl.addWidget(line)
        vl.addWidget(self.startBtn)
        vl.addWidget(self.text)
        vl.addLayout(btns)

        w = QWidget()
        w.setLayout(vl)
        w.setContentsMargins(50, 30, 50, 50)

        self.setCentralWidget(w)
        openBtn.clicked.connect(self.openFile)

    @Slot()  # 这个装饰器不加也行，下面edt_log要改成你自己文本框的名字
    def outputWritten(self, text):
        # self.edt_log.clear()
        cursor = self.text.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        self.text.setTextCursor(cursor)
        self.text.ensureCursorVisible()

    def openFile(self):
        filePath = QFileDialog.getExistingDirectory(
            self, "选择数据集所在文件夹", "C:\\Users\\"
        )
        self.path.setText(filePath)

    def openRes(self):
        self.res = QWidget()
        center(self.res, "结果图像", 572, 194)
        self.res.pix = QPixmap("__results___46_1.png")
        size = self.res.pix.size()

        self.res.label1 = QLabel(self.res)
        self.res.label1.setGeometry(0, 0, size.width(), size.height())
        self.res.label1.setPixmap(self.res.pix)
        self.res.show()

    def openCh(self):
        self.w = QWidget()
        center(self.w, "分析图表", 1153, 595)
        self.w.pix = QPixmap("__results___40_0.png")
        size = self.w.pix.size()

        self.w.label1 = QLabel(self.w)
        self.w.label1.setGeometry(0, 0, size.width(), size.height())
        self.w.label1.setPixmap(self.w.pix)
        self.w.show()
        self.ww = QWidget()
        center(self.ww, "分析图表", 1153, 595)
        self.ww.pix = QPixmap("__results___45_0.png")
        size = self.ww.pix.size()

        self.ww.label1 = QLabel(self.ww)
        self.ww.label1.setGeometry(0, 0, size.width(), size.height())
        self.ww.label1.setPixmap(self.ww.pix)
        self.ww.show()

    def train(self):
        if _model == "unet":
            train_main(self.path.text())
        else:
            swin_main(self.path.text())


if __name__ == "__main__":
    app = QApplication([])
    window = Menu()
    window.show()
    sys.exit(app.exec_())
