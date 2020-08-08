# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtCore,QtGui,QtWidgets
from PyQt5.QtWidgets import QApplication,QMainWindow,QMessageBox,QFileDialog,QLabel
import sys
import numpy as np
import os
import gen_tamper_dataset



dataset_root=''
save_root=''


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(819, 647)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(520, 0, 301, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 819, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_3 = QtWidgets.QMenu(self.menubar)
        self.menu_3.setObjectName("menu_3")
        self.menu_4 = QtWidgets.QMenu(self.menubar)
        self.menu_4.setObjectName("menu_4")
        self.menu_5 = QtWidgets.QMenu(self.menubar)
        self.menu_5.setContextMenuPolicy(QtCore.Qt.DefaultContextMenu)
        self.menu_5.setObjectName("menu_5")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_droot = QtWidgets.QAction(MainWindow)
        self.action_droot.setMenuRole(QtWidgets.QAction.ApplicationSpecificRole)
        self.action_droot.setObjectName("action_droot")
        self.action_dsavepath = QtWidgets.QAction(MainWindow)
        self.action_dsavepath.setObjectName("action_dsavepath")
        self.action_COCO = QtWidgets.QAction(MainWindow)
        self.action_COCO.setCheckable(True)
        self.action_COCO.setChecked(True)
        self.action_COCO.setObjectName("action_COCO")
        self.action_double_edge = QtWidgets.QAction(MainWindow)
        self.action_double_edge.setCheckable(True)
        self.action_double_edge.setChecked(True)
        self.action_double_edge.setObjectName("action_double_edge")
        self.action_src = QtWidgets.QAction(MainWindow)
        self.action_src.setCheckable(True)
        self.action_src.setObjectName("action_src")
        self.action_tamper = QtWidgets.QAction(MainWindow)
        self.action_tamper.setCheckable(True)
        self.action_tamper.setChecked(True)
        self.action_tamper.setObjectName("action_tamper")
        self.action_poisson = QtWidgets.QAction(MainWindow)
        self.action_poisson.setCheckable(True)
        self.action_poisson.setChecked(True)
        self.action_poisson.setObjectName("action_poisson")
        self.action_GT = QtWidgets.QAction(MainWindow)
        self.action_GT.setCheckable(True)
        self.action_GT.setChecked(True)
        self.action_GT.setObjectName("action_GT")

        self.action_mask = QtWidgets.QAction(MainWindow)
        self.action_mask.setCheckable(True)
        self.action_mask.setChecked(True)
        self.action_mask.setObjectName("action_mask")

        self.action_optimize = QtWidgets.QAction(MainWindow)
        self.action_optimize.setCheckable(True)
        self.action_optimize.setChecked(True)
        self.action_optimize.setObjectName("action_optimize")

        self.action_start = QtWidgets.QAction(MainWindow)
        self.action_start.setObjectName("action_start")
        self.action_start.setText('开始程序')

        self.menu.addAction(self.action_droot)
        self.menu.addAction(self.action_dsavepath)
        self.menu_2.addAction(self.action_double_edge)
        self.menu_2.addAction(self.action_src)
        self.menu_2.addAction(self.action_tamper)
        self.menu_2.addAction(self.action_poisson)
        self.menu_2.addAction(self.action_GT)
        self.menu_3.addAction(self.action_COCO)
        self.menu_4.addAction(self.action_mask)
        self.menu_4.addAction(self.action_optimize)
        self.menu_5.addAction(self.action_start)

        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_3.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_4.menuAction())
        self.menubar.addAction(self.menu_5.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.menu.setTitle(_translate("MainWindow", "文件"))
        self.menu_2.setTitle(_translate("MainWindow", "保存内容"))
        self.menu_3.setTitle(_translate("MainWindow", "选择数据集"))
        self.menu_4.setTitle(_translate("MainWindow", "方法"))
        self.menu_5.setStatusTip(_translate("MainWindow", "Ready"))
        self.menu_5.setTitle(_translate("MainWindow", "开始"))
        self.action_droot.setText(_translate("MainWindow", "选择数据集根目录"))
        self.action_dsavepath.setText(_translate("MainWindow", "选择数据集保存目录"))
        self.action_COCO.setText(_translate("MainWindow", "COCO"))
        self.action_double_edge.setText(_translate("MainWindow", "保存双边缘图"))
        self.action_src.setText(_translate("MainWindow", "保存原图"))
        self.action_tamper.setText(_translate("MainWindow", "保存直接篡改图"))
        self.action_poisson.setText(_translate("MainWindow", "保存poisson融合篡改图"))
        self.action_GT.setText(_translate("MainWindow", "保存GT"))
        self.action_mask.setText(_translate("MainWindow", "mask面积约束"))
        self.action_optimize.setText(_translate("MainWindow", "较优区域寻找"))



class My_Main(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(My_Main, self).__init__()
        self.setupUi(self)
        self.action_droot.triggered.connect(self.chooseRootPath)
        self.action_dsavepath.triggered.connect(self.chooseSavePath)
        self.action_start.triggered.connect(self.start)

    def chooseRootPath(self):
        print('we are in chooseRootPath')
        global dataset_root
        dataset_root = QFileDialog.getExistingDirectory(self,"请选择数据集所在的文件夹",'D:\\')
        print(dataset_root)



    def chooseSavePath(self):
        print('we are in chooseSavePath')
        global save_root
        save_root = QFileDialog.getExistingDirectory(self,"请选择保存生成数据的文件夹",'D:\\实验室\\图像篡改检测\\数据集\\COCO')
        content = ['double_edge_result', 'src_result','tamper_result','tamper_poisson_result','ground_truth_result']
        print(save_root)
        if os.listdir(save_root):
            print('文件夹不为空，请创建一个空的文件夹')

        else:
            for item in content:
                print(os.path.join(save_root,item))
                os.mkdir(os.path.join(save_root,item))

        print('创建文件成功')
    def start(self):
        print('we are in start')
        flag = QMessageBox.information(self,"提示","你确认开始生成数据集吗？", QMessageBox.Yes|QMessageBox.No,QMessageBox.Yes)
        if flag == QMessageBox.Yes:
            #self.label_foreground = QLabel
            # self.label_foreground.setText('foreground')
            # bmp = QtGui.QPixmap("D:\\Image_Tamper_Project\\Lab_project_code\\2020-7-23\make_tamper_dataset_from_coco\\filter_tamper2\\Tp_18575_70048_317.65_162.93_637.06_470.23_bowl.bmp")


            # 种类共80类别，传一个列表，有两个值cat_range = [1,80]
            # 面积约束标志, area_constraint

            # 较优求解标志, optimize_constraint
            # 每个类别数量, num_per_cat
            # 保存根目录地址, save_root_path

            gen_tamper_dataset.main(save_root_path=save_root,dataset_root=dataset_root)

            pass
        elif flag == QMessageBox.No:
            pass
if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = My_Main()
    myWin.show()
    sys.exit(app.exec_())