import tkinter
import tkinter.filedialog
import os
from PIL import ImageGrab
from time import sleep
from tkinter import *
from tkinter import messagebox

from PIL import Image,ImageTk
from tkinter.filedialog import askopenfilename
import codecs

import numpy as np
import paddle
import matplotlib.pyplot as plt
# import ../sim_cnn/test
# import sys
# sys.path.append("../sim_cnn")
# import test # not work



class MyMenu():
    '''菜单类'''
    def __init__(self, root):
        '''初始化菜单'''
        self.menubar = tkinter.Menu(root) # 创建菜单栏
        # 创建“文件”下拉菜单
        filemenu = tkinter.Menu(self.menubar, tearoff=0)
        filemenu.add_command(label="打开", command=self.file_open)
        filemenu.add_command(label="新建", command=self.file_new)
        filemenu.add_command(label="保存", command=self.file_save)
        filemenu.add_separator()
        filemenu.add_command(label="退出", command=root.quit)
        # 创建“编辑”下拉菜单
        editmenu = tkinter.Menu(self.menubar, tearoff=0)
        editmenu.add_command(label="剪切", command=self.edit_cut)
        editmenu.add_command(label="复制", command=self.edit_copy)
        editmenu.add_command(label="粘贴", command=self.edit_paste)
        # 创建“帮助”下拉菜单
        helpmenu = tkinter.Menu(self.menubar, tearoff=0)
        helpmenu.add_command(label="关于", command=self.help_about)
        # 将前面三个菜单加到菜单栏
        self.menubar.add_cascade(label="文件", menu=filemenu)
        self.menubar.add_cascade(label="编辑", menu=editmenu)
        self.menubar.add_cascade(label="帮助", menu=helpmenu)
        # 最后再将菜单栏整个加到窗口 root
        root.config(menu=self.menubar)

    def file_open(self):
        choosepic()
        # self.filename = filedialog.askopenfilename()
        # messagebox.showinfo('打开', self.filename)

    def file_new(self):
        messagebox.showinfo('新建','文件-新建！')  # 消息提示框
        pass

    def file_save(self):
        messagebox.showinfo('保存', '文件-保存！')  # 消息提示框
        pass

    def edit_cut(self):
        messagebox.showinfo('剪切', '编辑-剪切！')  # 消息提示框
        pass

    def edit_copy(self):
        messagebox.showinfo('复制', '编辑-复制！')  # 消息提示框
        pass

    def edit_paste(self):
        messagebox.showinfo('粘贴','编辑-粘贴！')  # 消息提示框
        pass

    def help_about(self):
        messagebox.showinfo('关于', '作者：Eric \n verion 1.0  \n myh912579@gmail.com \n 感谢您的使用！')  # 弹出消息提示框


img_path = ''
imgs_path = []
imgs_path_idx = 0
img_idx = 0


# 打开图片文件并显示
def choosepic():
    path_ = askopenfilename()
    path.set(path_)
    global img_path
    img_path = file_entry.get() # 具体一张图像的路径 
    # print(img_path)
    global imgs_path
    imgs_path = os.listdir(os.path.split(img_path)[0]) 
    # print(imgs_path)
    # 如果并非得到理想的文件名顺序
    # imgs_path.sort(key=lambda x:int(x[:-4]))
    global imgs_path_idx 
    imgs_path_idx = [i for i in range(len(imgs_path))]
    global img_idx
    img_idx = imgs_path.index(os.path.split(img_path)[1])
    # print(img_idx)

    img_open = Image.open(img_path).resize((400,400))
    img = ImageTk.PhotoImage(img_open)
    image_label.config(image=img)
    image_label.image = img  # keep a reference

# 上一张
def next_up():
    global img_path
    global imgs_path
    global imgs_path_idx
    global img_idx

    if img_idx == 0:
        messagebox.showinfo('提醒', '已是第一张！')
        return
    img_idx -= 1
    # print(img_idx)
    img_path = img_path.replace(os.path.split(img_path)[1], imgs_path[img_idx])
    path.set(img_path)
    img_open = Image.open(img_path)
    img = ImageTk.PhotoImage(img_open)
    image_label.config(image=img)
    image_label.image = img  # keep a reference


# 下一张
def next_down():
    global img_path
    global imgs_path
    global imgs_path_idx
    global img_idx

    if img_idx == (len(imgs_path)-1):
        messagebox.showinfo('提醒', '已是最后一张！')
        return
    img_idx += 1
    # print(img_idx)
    img_path = img_path.replace(os.path.split(img_path)[1], imgs_path[img_idx])
    path.set(img_path)
    img_open = Image.open(img_path)
    img = ImageTk.PhotoImage(img_open)
    image_label.config(image=img)
    image_label.image = img  # keep a reference


def inference_click():
    os.system('python ' + '../U-Net/predict.py ' + img_path)

def show_click():
    os.system('python ' + '../../Detection/yolo.py ' + img_path)

    # TODO
    # model = test.built_model(input_shape=(46, 30, 3), is_training=False)
    # x_test = test_data(img_path)
    # predict_digit(model, x_test, visual_layer_num=1)



if __name__ == '__main__':
    # 创建tkinter主窗口
    root = tkinter.Tk()
    root.iconbitmap('buf.ico')
    root.title('CNN_Visualizer')
    # 指定主窗口位置与大小
    root.geometry('640x480+100+50')   # width x height + widthoffset + heightoffset
    # 不允许改变窗口大小
    root.resizable(False, False)
    root.focusmodel()
    # 创建主菜单栏
    mymenu = MyMenu(root)

    path = StringVar() # 属于Tkinter下的对象，会自动跟踪变量值的变化

    image_label = Label(root, bg='gray')
    image_label.place(x=0, y=0,width = 400, height = 400)

    tkinter.Button(root, text='打开图像', bg='DarkKhaki', font=('Arial', 12), relief='raised',command=choosepic).place(x=30, y=410,w=150, h=40)
    tkinter.Button(root, text='上一张',  bg='DarkKhaki', font=('Arial', 12), command=next_up).place(x=250, y=410, w=150, h=40)
    tkinter.Button(root, text='下一张',  bg='DarkKhaki', font=('Arial', 12), command=next_down).place(x=450, y=410, w=150, h=40)

    dir_label = Label(root, text='当前图像路径：').place(x=420, y=0)
    file_entry = Entry(root, state='readonly', text=path)
    # file_entry.pack(side='right', expand='yes', anchor='ne')
    file_entry.place(x=420, y=20, width = 200, height = 20)

    model_label = Label(root, text='选择模型：').place(x=420, y=50)
    model_valueText = tkinter.Entry(root,width=12)
    model_valueText.place(x=420 , y=70, width=100, height=30, anchor=NW)

    layer_label = Label(root, text='选择层：').place(x=420, y=110)
    layer_valueText = tkinter.Entry(root,width=12)
    layer_valueText.place(x=420 , y=130, width=100, height=30, anchor=NW)

    buttonSegmentation = tkinter.Button(root, text='Inference', bg='DarkKhaki', command=inference_click)
    buttonSegmentation.place(x=440, y=210, width=150, height=40)

    buttonSegmentation = tkinter.Button(root, text='Show', bg='DarkKhaki', command=show_click)
    buttonSegmentation.place(x=440, y=260, width=150, height=40)

    paddle.enable_static()
    # program = paddle.load('./models/inference_model.pdmodel')

    paddle.disable_static()
    net = paddle.jit.load('./models/dyn_model')
    net.eval()
    print(net.full_name())
    # print(net.parameters())
    paddle.summary(net, input_size=(None, 1, 28, 28))
    
    data = np.load('data/mnist.npz')
    test_imgs = data['x_test']
    pic_id = int(input('Input pic id: '))
    plt.imshow(test_imgs[pic_id])
    plt.show()
    input = paddle.to_tensor(test_imgs[pic_id]/255., dtype='float32').reshape((1, 1, 28, 28))
    print(net(input))

    # 启动消息主循环
    root.update()
    root.mainloop()