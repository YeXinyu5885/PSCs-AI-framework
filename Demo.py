from tkinter import messagebox
import tkinter as tk
from tkinter.ttk import Combobox
import pandas as pd
from tkinter import filedialog
import plotly.express as px
import windnd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from functools import partial
from category_encoders import TargetEncoder
import joblib
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import shap
from PIL import Image, ImageTk
import pickle



class window:
    def __init__(self,name,label = tk.Tk,parent = None):
        self.name = name
        self.window_ = label()
        self.path = {'load':None,
                      'save':None}
        self.feature = {}
    def rest(self):
        self.window_.title(self.name)  # 设置窗口标题
        self.k,self.g = self.window_.maxsize() #获取分辨率
        #self.window_.geometry(f'{self.k//2}x{self.g//2}+{self.k//4}+{self.g//4}') #宽乘高,窗口大小，+后面为打开位置，x，y
        self.window_.geometry(f'{self.k}x{self.g}') #宽乘高,窗口大小，+后面为打开位置，x，y
        self.window_.focus_get()#对焦
    
    def add_button(self,text,size,row,column,command = '',return_botten = False,in_type = 'g'):
        button = tk.Button(self.window_, text=text, command=command,font = ('楷体',size))
        if in_type == 'g':
            button.grid(row = row,column = column)
        else :
            button.place(x = row,y =column )
        if return_botten:
            return button
    def add_label(self,text,size,row,column,return_label = False):
        label = tk.Label(self.window_,text = text,font = ('楷体',size))
        label.grid(row = row,column = column)
        if return_label:
            return label  

    def add_enttry(self,text,size,row,column,save_name,return_entry = False):
        get_str = tk.StringVar()
        entry = tk.Entry(self.window_,text = text,textvariable = get_str,font = ('楷体',size))
        entry.grid(row = row,column = column)
        self.path[save_name] = get_str
        if return_entry:
            return entry 
        
    def dragged_files(self,constraint,load_tip,size,row,column,path_name,files):
        msg = '\n'.join((item.decode('gbk') for item in files))
        queren = messagebox.showinfo("Drag and drop file path",msg,parent = self.window_)
        put_load = self.add_label(f'{load_tip}:{msg}',size,row+1,column,True)
        if queren:
            if constraint == '':
                if not os.path.isdir(msg):
                    messagebox.showerror('error','data is not floder',parent = self.window_)
                else :
                    self.path[path_name] = msg
            else:
                if msg.split('.')[-1] != constraint:
                    messagebox.showerror('error','data is not .csv',parent = self.window_)
                else:
                    self.path[path_name] = msg
        return msg
    
    def add_loadpath(self,tip_text,size,row,column,constraint,load_tip,path_name):
        #实现拖放事件
        def drag_enter(event):
            _label.config(bg="lightblue")
        def drag_leave(event):
            _label.config(bg="white")

        _label = self.add_label(tip_text,size,row,column,True)
        load_path = windnd.hook_dropfiles(_label , func=partial(self.dragged_files,constraint,load_tip,size,row,column,path_name))
        _label.bind("<Enter>", drag_enter)
        _label.bind("<Leave>", drag_leave)
        
        put_load = self.add_label(f'{load_tip}:{load_path}',size,row+1,column,True)
        #_label.bind("<Drop>", drag_drop)
        
        return load_path

        '''#self.window_.title("You can drag files in here,It must be a. csv file")
        
        # 允许拖放
        _label.drop_target_register(tk.DND_FILES)
        _label.dnd_bind('<<Drop>>', loadpath)'''
    
    def add_radio(self,text_list,row,column,size = 16):
        get_str = tk.StringVar(value = 'FF')
        for i,text in enumerate(text_list):
            tk.Radiobutton(self.window_,text = text,font = ('楷体',size),variable = get_str,
                           value = text).place(x = self.k*(row+i)//20,y = self.g//20*column)
        self.path['feature'] = get_str
        return get_str.get()
    def add_combox(self,row,column,width = 10,height=10,state = None,number = None):
        get_str = tk.StringVar()
        valus = pd.read_csv(fr'{self.path["data_path"]}', encoding='ISO-8859-1').columns[:-1].tolist()
        return_str = Combobox(self.window_,state = state,width = width,textvariable = get_str)
        return_str.grid(row = row,column = column)
        return_str['values']=valus  
        return_str.current(1) 
        self.feature[number] = get_str
        return return_str
    def close_(self):
        c = messagebox.askokcancel('close','Do you want to close the current window',parent = self.window_)
        if c:
            self.window_.destroy()
    




def run_yc(input_file_path,model_file_path,train_file_path,output_file_path,target):

    # 目标特征列

    # 读取输入文件 (测试集)，测试集没有目标特征列
    try:
        test_data = pd.read_csv(input_file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            test_data = pd.read_csv(input_file_path, encoding='gbk')
        except UnicodeDecodeError:
            test_data = pd.read_csv(input_file_path, encoding='utf-16')

    # 加载之前训练时使用的模型
    model = joblib.load(model_file_path)

    # 初始化目标编码器
    encoder = TargetEncoder()

    # 加载训练数据（仅用于重新拟合目标编码器）
    try:
        train_data = pd.read_csv(train_file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            train_data = pd.read_csv(train_file_path, encoding='gbk', low_memory=False)
        except UnicodeDecodeError:
            train_data = pd.read_csv(train_file_path, encoding='utf-16', low_memory=False)

    # 处理目标特征中的缺失值
    train_data = train_data.dropna(subset=[target])

    # 只对特定的非数值列进行目标编码
    categorical_columns = [col for col in test_data.columns if test_data[col].dtype == object]

    # 遍历特定的列，对测试集的非数值特征进行目标编码
    for column in categorical_columns:
        if column in train_data.columns:  # 确保训练集中有此列
            # 拟合编码器
            encoder.fit(train_data[column], train_data[target])
            # 对测试集进行目标编码
            encoded_col = encoder.transform(test_data[column])
            # 用训练数据的全局均值填充未知特征值
            global_mean = encoder.transform(train_data[column]).mean()
            test_data[column] = encoded_col.fillna(global_mean)

    # 确保测试集的特征与模型训练时的特征一致
    model_features = model.get_booster().feature_names
    test_data = test_data[model_features]

    # 使用加载的模型进行预测
    predictions = model.predict(test_data)

    # 输出预测结果到控制台
    print("预测结果：")
    print(predictions)

    # 将预测结果保存到CSV文件
    pd.DataFrame(predictions, columns=[target]).to_csv(output_file_path, index=False)

    print(f"预测结果已保存到: {output_file_path}")
    return predictions
def get_yc(parent_window):
    
    input_file_path = parent_window.path['input']
    model_file_path = parent_window.path['model']
    train_file_path = parent_window.path['train']
    output_file_path = parent_window.path['output'] + '\\' +parent_window.path['save_name'].get()
    #return_path = parent_window.path['save'] + '\\' +parent_window.path['save_name'].get()
    str_ = parent_window.path['feature'].get()
    target = f'JV_default_{str_}'
    print(input_file_path,model_file_path,train_file_path,output_file_path)
    #if f'JV_default_{str_}' in pd.read_csv(load_path).columns:
    yc_wind = window('output',tk.Toplevel)
    yc_wind.rest()
    yc_wind.add_label(f'The data you input is {input_file_path}',16,1,0)
    yc_wind.add_label(f'The choose method is {output_file_path}',16,2,0)
    yc_wind.add_label(f'The program is running',16,3,0)
    #fun_ = fun_dict[f'{str_}_main']
    fun_ = run_yc
    predictions  = fun_(input_file_path,model_file_path,train_file_path,output_file_path,target)
    yc_wind.add_label(f'Forecast results:{predictions}' ,16,4,0)
    yc_wind.add_label(f'The predicted results have been saved to:{output_file_path}' ,16,5,0)
    ''''colsample_bytree': 0.5, 'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 500, 'subsample': 0.9'''
    '''yc_wind.add_label(f'Best Parameters from Grid Search colsample_bytree:{colsample_bytree}' ,16,4,0)
        yc_wind.add_label(f'Best Parameters from Grid Search learning_rate:{learning_rate}' ,16,5,0)
        yc_wind.add_label(f'Best Parameters from Grid Search max_depth:{max_depth}' ,16,6,0)
        yc_wind.add_label(f'Best Parameters from Grid Search n_estimators:{n_estimators}' ,16,7,0)
        yc_wind.add_label(f'Best Parameters from Grid Search subsample:{subsample}' ,16,8,0)
        yc_wind.add_label(f'Root Mean Squared Error (RMSE):{rmse}' ,16,9,0)
        yc_wind.add_label(f'Mean Absolute Error (MAE):{mae}' ,16,10,0)
        yc_wind.add_label(f'R^2 (Coefficient of Determination):{r2}' ,16,11,0)'''
    #yc_wind.add_button('保存',20,12,0)
    yc_wind.add_button('Enter new data',20,12,0)
    yc_wind.add_button('Over',20,13,0)
    #else :
    #    messagebox.showerror('error',f'JV_default_{str_} not in data',parent = parent_window.window_)
def yc():
    yc_wind = window('forecast',tk.Toplevel)
    yc_wind.rest()
    load_label = yc_wind.add_label('Please select input data',16,0,0,True)
    load_path = yc_wind.add_loadpath('You can drag files in here,It must be a .csv file',10,0,1,'csv','Already submitted data','input')

    save_label = yc_wind.add_label('Please select the location to save the input file',16,6,0,True)
    return_path = yc_wind.add_loadpath('You can drag files in here,It must be a folder',10,6,1,'','Already Location of submitted files','output')
    
    save_label = yc_wind.add_label('Please enter a save file name',16,8,0,True)
    return_path = yc_wind.add_enttry('save_data.csv',12,8,1,'save_name')
    
    load_label = yc_wind.add_label('Please select a predictive model',16,2,0,True)
    load_path = yc_wind.add_loadpath('You can drag files in here,It must be a .pkl file',10,2,1,'pkl','Already submitted data','model')

    load_label = yc_wind.add_label('Please select model training data',16,4,0,True)
    load_path = yc_wind.add_loadpath('You can drag files in here,It must be a .csv file',10,4,1,'csv','Already submitted data','train')
 
    yc_wind.add_label('',16,9,0)
    yc_wind.add_label('Please select feature',16,10,0)
    retrun_str = yc_wind.add_radio(['FF','Jsc','PCE','Voc'],4.2,5.3)

    yc_wind.add_button('submit',20,yc_wind.k//20*4.5,yc_wind.g//20*7 ,partial(get_yc,yc_wind),True,'p')
    
def add_button(now_wind,i,number = None,_botten = None):
    if _botten is not None:
        _botten.destroy()
    now_wind.add_label(f'Please select the {i}rd feature',16,1+i,0)
    now_wind.add_combox(1+i,1,number = 3)
    #return_botten = now_wind.add_button('添加指标',10,i+1,2,partial(add_button,now_wind,i+1),True)
    now_wind.add_button('analysis',10,i+2,1,partial(get_yh,now_wind))
    #return return_botten
"""
'''
def run_fx(now_wind,i):
    '''if i == 1:
        messagebox.showerror('选择指标数错误','选择指标数应大于等于2')
    else:'''
    fx_pic_wind = window('分析_pic',tk.Toplevel)
    fx_pic_wind.rest()
    return_pic_title = {
        0:'热力图',
        1:'雷达图'
    }
    fx_pic_wind.add_label(f'根据选择的指标数量返回{return_pic_title[i>=3]}',16,1,0)'''"""
    

'''def run_fx(parent_window,num):
    

    
    for i in range(num):
        now_wind = window(f'Features {i}',tk.Toplevel)
        
        # 使用Image和ImageTk模块来加载图片
        image = Image.open(image_path + '\\' + f'Features {i}.png')
        image = ImageTk.PhotoImage(image)
        # 创建标签，并将图片设置为背景
        label = now_wind.add_label('',16,0,0,True)
        #tk.Label(root, image=image)
        label.image = image  # 防止图片被垃圾回收
        now_wind.window_.mainloop()
    '''

import tkinter as tk
from tkinter.messagebox import showinfo
 
def get_fx(parent_window,num):
    
    image_path = parent_window.path['save_path']
    mother_data_path = parent_window.path['mother_data_path']
    sub_data_path = parent_window.path['sub_data_path']
    run_fx(mother_data_path,sub_data_path,image_path)
    
    for i in range(num):
        # 创建Toplevel窗口
        top = tk.Toplevel()
        # 设置窗口标题
        top.title(fr"Features {i}")
        # 创建PhotoImage对象，并插入图片
        image = tk.PhotoImage(file=fr"{image_path}\Features {i}.png")
        # 创建Label小部件，并设置图片
        label = tk.Label(top, image=image)
        label.image = image
        label.grid(row = 0,column = 0)
        with open(fr"{image_path}\Features {i}.txt", 'r') as file:
            content = file.read()
        contents = content.split('/n')
        for n,now_content in  enumerate(contents):
            label = tk.Label(top,text = now_content,font = ('楷体',16))
            label.grid(row = n+1,column = 0)
    
def fx():
    fx_wind = window('optimize',tk.Toplevel)
    fx_wind.rest()
    load_label = fx_wind.add_label('Please select mother_data',16,0,0,True)
    load_path = fx_wind.add_loadpath('You can drag files in here,It must be a .csv file',10,0,1,'csv','Submitted data','mother_data_path')
    
    load_label = fx_wind.add_label('Please select sub_data',16,2,0,True)
    load_path = fx_wind.add_loadpath('You can drag files in here,It must be a .csv file',10,2,1,'csv','Submitted data','sub_data_path')
 
    save_label = fx_wind.add_label('Please select the location to save the input file',16,4,0,True)
    return_path = fx_wind.add_loadpath('You can drag files in here,It must be a folder',10,4,1,'','Location of submitted files','save_path')
    
    
    fx_wind.add_button('submit',20,fx_wind.k//20*3.5,fx_wind.g//20*7 ,partial(get_fx,fx_wind,8),True,'p')
    
def get_yh(parent_window):
    mother_data_path = parent_window.path['data_path']
    columns = [parent_window.feature[i].get() for i in parent_window.feature]
    user_excluded_features = ','.join(columns)
    print(columns)
    
    run_yh(mother_data_path,user_excluded_features)
    './Feature/'
    str_dict = {
        2:'Heatmap',
        3:'Radarmap'
    }
    num = len(columns)
    for i in range(6):
        # 创建Toplevel窗口
        top = tk.Toplevel()
        # 设置窗口标题
        top.title(fr"Cluster {i}")
        # 创建PhotoImage对象，并插入图片
        image = tk.PhotoImage(file=f"./Feature/Cluster_{i}_{str_dict[num]}.png")
        # 创建Label小部件，并设置图片
        label = tk.Label(top, image=image)
        label.image = image
        label.grid(row = 0,column = 0)
'''
def reset_run():
    c = messagebox.askokcancel('选择','是否重新训练模型')
    if c:
        pass
    
    yh_wind = window('优化返回',tk.Toplevel)
    yh_wind.rest()
    yh_wind.add_label('需要重新写一个函数先将添加的数据和原始数据合并',16,1,0)
    yh_wind.add_label('运行HTL_analysis_and_visualization返回特征重要性图',16,2,0)
    
    yh_wind.add_label('运行PSC_optimization返回优化方案',16,3,0)
'''

def yh_(yh_wind):
    yh_wind.add_label('Please select the 1st feature',16,2,0)
    yh_wind.add_combox(2,1,number = 1)
    yh_wind.add_label('Please select the 2nd feature',16,3,0)
    yh_wind.add_combox(3,1,number = 2)
    _botten = yh_wind.add_button('Add Feature',10,3,2,partial(add_button,yh_wind,3),True)

    yh_wind.add_button('SHAP analysis',10,4,1,partial(get_yh,yh_wind))
    
    

def yh():
    yh_wind = window('analysis',tk.Toplevel)
    yh_wind.rest()
    load_label = yh_wind.add_label('Please select data',16,0,0,True)
    load_path = yh_wind.add_loadpath('You can drag files in here,It must be a .csv file',10,0,1,'csv','Submitted data','data_path')
    
    yh_wind.add_button('input data to model',20,yh_wind.k//20*3.5,yh_wind.g//20*7 ,partial(yh_,yh_wind),True,'p')
    
'''    yh_wind.add_label('请选择第1个指标',16,2,0)
    yh_wind.add_combox(2,1)
    yh_wind.add_label('请选择第2个指标',16,3,0)
    yh_wind.add_combox(3,1)
    _botten = yh_wind.add_button('添加指标',10,3,2,partial(add_button,yh_wind,3),True)
    
    yh_wind.add_button('SHAP分析',10,4,1,reset_run)
'''



def run_fx(mother_data_path,sub_data_path,__save_path):
    output_data_path = "OPT_PSC.csv" # 输出文件路径
    encoded_sub_data_path = "encodeOPT_PSC.csv"  # 输出的编码后的子数据文件路径
    encoder_file_path = "target_PCEencoder.joblib"
    file_path = output_data_path
    encoded_file_path = "encodeOPT_PSC.csv"
    mapping_file_path = "target_PCEencoder_mapping.joblib"
    model_save_path = "OPT_PSC.joblib"

    model_path = model_save_path
    data_path = encoded_sub_data_path
    mapping_path = mapping_file_path


    # 加载母数据和子数据，并指定编码
    mother_data = pd.read_csv(mother_data_path, encoding='ISO-8859-1')  # 或使用 encoding='cp1252'
    sub_data = pd.read_csv(sub_data_path, encoding='ISO-8859-1')
    print("Data loaded successfully.")

    # 检查母数据和子数据的列差异
    missing_columns = [col for col in mother_data.columns if col not in sub_data.columns]
    print("Missing columns in sub data:", missing_columns)

    # 为子数据中缺少的列填充数据
    for column in missing_columns:
        # 根据母数据的列类型来选择填充值
        if pd.api.types.is_numeric_dtype(mother_data[column]):
            # 数值列：使用母数据的平均值
            fill_value = mother_data[column].mean()
        else:
            # 非数值列：使用母数据的众数
            fill_value = mother_data[column].mode()[0]

        # 在子数据中添加缺少的列，并填充相应的值
        sub_data[column] = fill_value
        print(f"Filled missing column '{column}' with {'mean' if pd.api.types.is_numeric_dtype(mother_data[column]) else 'mode'} value:", fill_value)

    # 重新排列子数据的列顺序以匹配母数据
    sub_data = sub_data[mother_data.columns]

    # 合并数据
    combined_data = pd.concat([mother_data, sub_data], ignore_index=True)
    print("Data combined successfully.")

    # 保存合并后的数据
    combined_data.to_csv(output_data_path, index=False)
    print(f"Combined data saved to '{output_data_path}'.")




    # 数据路径定义


    # 加载母数据和子数据，并指定编码
    mother_data = pd.read_csv(mother_data_path, encoding='ISO-8859-1')
    sub_data = pd.read_csv(sub_data_path, encoding='ISO-8859-1')
    print("Data loaded successfully.")

    # 检查母数据和子数据的列差异
    missing_columns = [col for col in mother_data.columns if col not in sub_data.columns]
    print("Missing columns in sub data:", missing_columns)

    # 为子数据中缺少的列填充数据
    for column in missing_columns:
        # 根据母数据的列类型来选择填充值
        if pd.api.types.is_numeric_dtype(mother_data[column]):
            # 数值列：使用母数据的平均值
            fill_value = mother_data[column].mean()
        else:
            # 非数值列：使用母数据的众数
            fill_value = mother_data[column].mode()[0]

        # 在子数据中添加缺少的列，并填充相应的值
        sub_data[column] = fill_value
        print(f"Filled missing column '{column}' with {'mean' if pd.api.types.is_numeric_dtype(mother_data[column]) else 'mode'} value:", fill_value)

    # 重新排列子数据的列顺序以匹配母数据
    sub_data = sub_data[mother_data.columns]

    # 合并数据
    combined_data = pd.concat([mother_data, sub_data], ignore_index=True)
    print("Data combined successfully.")

    # 保存合并后的数据
    combined_data.to_csv(output_data_path, index=False)
    print(f"Combined data saved to '{output_data_path}'.")

    # 目标编码步骤
    target = 'JV_default_PCE'
    features_to_encode = [column for column in combined_data.columns if combined_data[column].dtype == object and column != target]

    # 初始化目标编码器并对合并数据进行编码
    encoder = TargetEncoder()
    encoded_combined_data = combined_data.copy()
    encoded_combined_data[features_to_encode] = encoder.fit_transform(combined_data[features_to_encode], combined_data[target])

    # 提取编码后的子数据部分（即最初的8个测试数据）并保存
    encoded_sub_data = encoded_combined_data.iloc[-len(sub_data):]  # 获取合并数据中的最后N行
    encoded_sub_data.to_csv(encoded_sub_data_path, index=False)
    print(f"Encoded subset data saved to {encoded_sub_data_path}")

    # 保存目标编码器状态
    joblib.dump(encoder, encoder_file_path)
    print(f"Encoder saved to {encoder_file_path}")

    # 创建并保存编码映射
    mapping = {feature: dict(zip(combined_data[feature], encoded_combined_data[feature])) for feature in features_to_encode}
    joblib.dump(mapping, mapping_file_path)
    print(f"Mapping saved to {mapping_file_path}")

    '''
    import pandas as pd
    from category_encoders import TargetEncoder
    import joblib

    # 文件路径

    try:
        data = pd.read_csv(file_path, low_memory=False)
    except UnicodeDecodeError:
        try:
            data = pd.read_csv(file_path, encoding='latin1')
        except UnicodeDecodeError:
            data = pd.read_csv(file_path, encoding='utf-16')

    # 目标特征列
    target = 'JV_default_PCE'

    # 确定需要进行目标编码的列
    features_to_encode = [column for column in data.columns if data[column].dtype == object and column != target]

    # 初始化目标编码器并对指定的特征列进行编码
    encoder = TargetEncoder()
    data[features_to_encode] = encoder.fit_transform(data[features_to_encode], data[target])

    # 保存处理后的数据
    data.to_csv(encoded_file_path, index=False)
    print(f"Encoded data saved to {encoded_file_path}")

    # 保存目标编码器状态
    joblib.dump(encoder, encoder_file_path)
    print(f"Encoder saved to {encoder_file_path}")

    # 为了创建映射，我们需要原始数据和编码后的数据
    original_data = pd.read_csv(file_path, low_memory=False, encoding='latin1')
    encoded_data = pd.read_csv(encoded_file_path)

    # 创建映射关系
    category_mapping = {}
    for feature in features_to_encode:
        category_mapping[feature] = original_data[feature].astype(str).unique().tolist()

    # 保存映射关系
    joblib.dump(category_mapping, mapping_file_path)
    print(f"Mapping saved to {mapping_file_path}")

    '''



    # 自定义评估函数，添加打印语句以监控进度
    def custom_r2_scorer(y_true, y_pred):
        score = r2_score(y_true, y_pred)
        print(f"Evaluating model... Score: {score:.4f} - Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return score

    # 为确保随机性，设置numpy的随机种子
    np.random.seed(7)

    # 从 CSV 文件中读取数据
    data = pd.read_csv(encoded_file_path)
    # 删除所有值都是NaN的列
    data = data.dropna(axis=1, how='all')

    # 将包含NaN值的其他列的NaN值替换为整列的平均值
    data = data.fillna(data.mean())
    features = data.columns[:-1].tolist()
    X = data[features]  # 特征
    y = data["JV_default_PCE"]  # 目标标签

    # 将数据集分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=7)

    # 初始化XGBoost模型，使用找到的最佳参数
    xgb_model = XGBRegressor(
        n_estimators=802,
        max_depth=9,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.5,
    )

    # 训练模型
    xgb_model.fit(X_train, y_train)

    # 进行预测
    y_pred = xgb_model.predict(X_test)

    # 评估模型性能
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print('Root Mean Squared Error (RMSE):', rmse)
    print('Mean Absolute Error (MAE):', mae)
    print('R^2 (Coefficient of Determination):', r2)
    # 输出测试集性能参数


    # 模型保存路径

    # 保存训练好的XGBoost模型
    joblib.dump(xgb_model, model_save_path)
    print(f"Model saved successfully to {model_save_path}")


    # 模型和数据集的文件路径

    # 加载训练好的模型
    print("Loading model from", model_path)
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # 加载编码映射关系
    print("Loading mapping from", mapping_path)
    mapping = joblib.load(mapping_path)
    print("Mapping loaded successfully with structure:", type(mapping))

    # 构建反转映射，确保小数点精度为4位
    reversed_mapping = {}
    for feature, feature_mapping in mapping.items():
        if isinstance(feature_mapping, dict):
            # 反转字典并将浮点数统一为4位小数
            reversed_mapping[feature] = {round(float(v), 4) if isinstance(v, float) else v: k for k, v in feature_mapping.items()}
        elif isinstance(feature_mapping, list):
            # 处理映射为列表的情况
            reversed_mapping[feature] = {}
            for item in feature_mapping:
                if isinstance(item, dict):
                    for v, k in item.items():
                        rounded_v = round(float(v), 4) if isinstance(v, float) else v
                        reversed_mapping[feature][rounded_v] = k
        else:
            print(f"Unexpected mapping format for feature '{feature}': {type(feature_mapping)}")

    print("Reversed mapping constructed successfully.")

    # 加载数据集
    print("Loading data from", data_path)
    data = pd.read_csv(data_path)
    print("Data loaded successfully.")

    # 获取模型的特征
    model_features = list(model.get_booster().feature_names)

    # 确保数据集包含模型训练时使用的所有特征
    if not all(feature in data.columns for feature in model_features):
        raise ValueError("Not all model features are present in the dataset.")

    # 对每个样本进行SHAP分析和优化建议
    print("\nPerforming SHAP analysis and generating optimization suggestions for each sample.")
    for index, row in data.iterrows():
        save_text = ''
        print(f"\nAnalyzing sample index {index}")

        # 使用每个样本进行SHAP分析
        explainer = shap.Explainer(model)
        shap_values = explainer(pd.DataFrame([row[model_features]]))
        shap_sum = np.abs(shap_values.values).mean(axis=0)

        # 特征重要性排序
        importance_df = pd.DataFrame([model_features, shap_sum.tolist()]).T
        importance_df.columns = ['feature', 'shap_importance']
        importance_df = importance_df.sort_values('shap_importance', ascending=False)
        filtered_importance_df = importance_df[~importance_df['feature'].isin(['Cell_stack_sequence', 'Perovskite_composition_long_form'])]
        top_features_df = filtered_importance_df.head(5)

        # 绘制特征重要性柱状图
        fig = plt.figure(figsize=(10, 4))
        plt.bar(top_features_df['feature'], top_features_df['shap_importance'], width=0.4)
        plt.ylabel('SHAP Importance')
        plt.title(f"Top 5 Important Features for Sample index {index} (Filtered)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fr'{__save_path}/Features {index}.png')

        # 优化建议
        print(f"Optimization suggestions for sample index {index}:")
        save_text += f"Optimization suggestions for sample index {index}:"
        for feature in top_features_df['feature']:
            category_values = data[feature].unique()
            best_score = -np.inf
            best_category = None

            # 遍历特征的每个类别值，计算最优值
            for category in category_values:
                modified_row = row.copy()
                modified_row[feature] = category
                prediction = model.predict([modified_row[model_features]])[0]

                if prediction > best_score:
                    best_score = prediction
                    best_category = category

            # 获取原始和最佳类别值的映射
            original_category = row[feature]
            # 将数值按4位小数点舍入，确保匹配成功
            rounded_original_category = round(float(original_category), 4) if pd.api.types.is_numeric_dtype(data[feature]) else str(original_category)
            rounded_best_category = round(float(best_category), 4) if pd.api.types.is_numeric_dtype(data[feature]) else str(best_category)

            # 使用reversed_mapping查找并转换为原始值
            if feature in reversed_mapping:
                original_mapped_value = reversed_mapping[feature].get(rounded_original_category, f"No mapping for {rounded_original_category}")
                best_mapped_value = reversed_mapping[feature].get(rounded_best_category, f"No mapping for {rounded_best_category}")
            else:
                original_mapped_value = f"No encoding map for feature {feature}"
                best_mapped_value = f"No encoding map for feature {feature}"

            print(f"  - {feature}: Optimal value = {best_mapped_value} (Original: {original_mapped_value})")
            save_text += f"/n  - {feature}: Optimal value = {best_mapped_value} (Original: {original_mapped_value})"
            with open(fr'{__save_path}/Features {index}.txt','w',encoding='utf-8') as file:
                file.write(save_text) # 将字符串写入文件
    print("Analysis completed.")

    
def run_yh(mother_data_path,user_excluded_features):
    print(user_excluded_features)
    output_data_path = "Analysis_encoded_Supplementary_Data_1_PCE.csv"  # 输出的编码后的数据路径
    target = 'JV_default_PCE'
    encoder_file_path = "Analysis_target_PCEencoder.joblib"
    mapping_file_path = "Analysis_target_PCEencoder_mapping.joblib"

    file_path = output_data_path


    output_dir = './Feature/'




    # 加载数据并指定编码
    mother_data = pd.read_csv(fr"{mother_data_path}", encoding='ISO-8859-1')
    print("Mother data loaded successfully.")

    # 填充缺失值：数值特征用均值填充，分类特征用众数填充
    for column in mother_data.columns:
        if mother_data[column].dtype in ['float64', 'int64']:
            # 数值特征使用均值填充
            mother_data[column].fillna(mother_data[column].mean(), inplace=True)
        else:
            # 分类特征使用众数填充
            mother_data[column].fillna(mother_data[column].mode()[0], inplace=True)

    print("Missing values filled successfully.")

    # 目标特征列


    # 确定需要进行目标编码的列（排除目标列）
    features_to_encode = [column for column in mother_data.columns if mother_data[column].dtype == object and column != target]

    # 初始化目标编码器并对数据进行编码
    encoder = TargetEncoder()
    encoded_mother_data = mother_data.copy()
    encoded_mother_data[features_to_encode] = encoder.fit_transform(mother_data[features_to_encode], mother_data[target])

    # 保存编码后的数据
    encoded_mother_data.to_csv(output_data_path, index=False)
    print(f"Encoded mother data saved to '{output_data_path}'.")

    # 保存目标编码器状态
    joblib.dump(encoder, encoder_file_path)
    print(f"Encoder saved to {encoder_file_path}")

    # 创建并保存编码映射
    mapping = {feature: dict(zip(mother_data[feature], encoded_mother_data[feature])) for feature in features_to_encode}
    joblib.dump(mapping, mapping_file_path)
    print(f"Mapping saved to {mapping_file_path}")




    # 加载数据集
    data = pd.read_csv(file_path)

    # 定义分类时需要排除的特征，这些特征就是要研究的特征，限定2到3个组合
    default_excluded_features = ['HTL_stack_sequence', 'HTL_thickness_list', 'HTL_deposition_procedure']
    if user_excluded_features:
        excluded_features = ['JV_default_PCE'] + [feature.strip() for feature in user_excluded_features.split(',')]
    else:
        excluded_features = ['JV_default_PCE'] + default_excluded_features

    # 提取需要保留的特征以便后续使用
    data_excluded = data[excluded_features]

    # 删除不参与聚类的特征列
    data_for_clustering = data.drop(excluded_features, axis=1)

    # 列出包含 NaN 或 inf 值的特征并删除这些列
    nan_inf_columns = [
        'Backcontact_surface_treatment_before_next_deposition_step',
        'Add_lay_front_additives_compounds', 'Add_lay_front_additives_concentrations',
        'Add_lay_front_deposition_synthesis_atmosphere_pressure_total',
        'Add_lay_front_deposition_synthesis_atmosphere_pressure_partial',
        'Add_lay_front_deposition_synthesis_atmosphere_relative_humidity',
        'Add_lay_front_deposition_solvents_mixing_ratios',
        'Add_lay_front_deposition_reaction_solutions_compounds',
        'Add_lay_front_deposition_reaction_solutions_concentrations',
        'Add_lay_front_storage_relative_humidity',
        'Add_lay_front_surface_treatment_before_next_deposition_step',
        'Add_lay_back_additives_compounds', 'Add_lay_back_additives_concentrations'
    ]
    data_for_clustering = data_for_clustering.drop(columns=nan_inf_columns, errors='ignore')
    print("已删除包含NaN或inf值的特征列。")


    k = 6  # 聚类簇数

    
    # 标准化数据
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering.select_dtypes(include=['float64', 'int64', 'bool']))
    
    # 检查标准化后的数据是否包含异常值
    if np.isnan(data_scaled).any() or np.isinf(data_scaled).any():
        print("标准化后仍包含NaN或inf值，请检查数据清洗过程。")
    else:
        print("标准化后的数据无NaN或inf值。")

    # 使用t-SNE进行降维
    tsne = TSNE(n_components=2, random_state=0)
    data_tsne = tsne.fit_transform(data_scaled)

    # 选择聚类数量并应用K-Means算法进行聚类
    k = 6  # 聚类簇数
    kmeans = KMeans(n_clusters=k, random_state=0)
    clusters = kmeans.fit_predict(data_tsne)

    # 可视化聚类结果
    plt.figure(figsize=(8, 6))
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=clusters, cmap='viridis')
    plt.title("t-SNE")
    plt.xlabel("Dimention 1")
    plt.ylabel("Dimention 2")
    plt.colorbar()  # 显示颜色条，用于表示不同的簇标签
    plt.show()

    # 为每个样本添加聚类标签
    data['cluster_label'] = clusters
    
    # 创建输出目录
    #output_dir = '/personal/Feature/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 按簇标签保存每个聚类的样本到单独的CSV文件
    for i in range(k):
        print(f"正在处理簇 {i}")
        cluster_data = data[data['cluster_label'] == i]
        
        filename = f'{output_dir}/cluster_{i}_samples.csv'
        print(f"保存到 {filename}")
        cluster_data.to_csv(filename, index=False)
    









    # 根据特征数量生成不同图表
    os.makedirs(output_dir, exist_ok=True)

    # 加载编码映射文件
    mapping = joblib.load(mapping_file_path)
    print("Loaded encoding mapping.")

    # 定义解码函数
    def decode_value(value, mapping, feature):
        decoded_mapping = {round(v, 4): k for k, v in mapping[feature].items()}
        return decoded_mapping.get(round(value, 4), value)

    for cluster_number in range(k):
        file_path = f'{output_dir}/cluster_{cluster_number}_samples.csv'

        if not os.path.exists(file_path):
            print(f"文件 {file_path} 不存在，跳过。")
            continue

        data_cluster = pd.read_csv(file_path)
        selected_features = excluded_features[1:]

        if len(selected_features) == 3:
            # 雷达图
            grouped_data = data_cluster.groupby(selected_features)['JV_default_PCE'].mean().reset_index()
            top_combinations = grouped_data.nlargest(5, 'JV_default_PCE')
            categories = selected_features + ['JV_default_PCE']
            N = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            radar_center = 14
            num_circles = 8
            circle_values = [radar_center + i for i in range(num_circles + 1)]
            ax.set_ylim(radar_center, radar_center + num_circles)
            ax.set_yticks(circle_values)

            for index, row in top_combinations.iterrows():
                # 先获取未解码的特征值
                values = row[selected_features].tolist() + [row['JV_default_PCE']]
                values += values[:1]

                # 绘制图形
                ax.plot(angles, values, 'o-', linewidth=2)
                ax.fill(angles, values, alpha=0.4)

                # 解码特征值并创建标签
                decoded_values = [
                    decode_value(row[feature], mapping, feature) if feature in mapping else row[feature]
                    for feature in selected_features
                ]
                decoded_values += [row['JV_default_PCE']]
                label = f"Combination {index + 1}: " + ', '.join(f"{cat}: {value}" for cat, value in zip(categories, decoded_values))
                ax.plot(angles, values, 'o-', linewidth=2, label=label)

            ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], categories)
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 0.9), frameon=False)
            plt.title(f'Top 5 Combinations Radar Chart for Cluster {cluster_number}')
            heatmap_path = f"{output_dir}/Cluster_{cluster_number}_Radarmap.png"
            plt.savefig(heatmap_path)
            #plt.show()

        elif len(selected_features) == 2:
            # 热图
            data_cluster['JV_default_PCE'] = data_cluster['JV_default_PCE'].round(2)
            pivot_data = data_cluster.pivot_table(
                index=selected_features[0],
                columns=selected_features[1],
                values='JV_default_PCE'
            ).dropna(axis=0, how='all').dropna(axis=1, how='all')

            fig = px.imshow(
                pivot_data,
                labels=dict(
                    x=selected_features[1],
                    y=selected_features[0],
                    color='JV Default PCE'
                ),
                aspect='auto'
            )

            fig.update_layout(
                xaxis_nticks=len(pivot_data.columns),
                yaxis_nticks=len(pivot_data.index),
                width=1100,
                height=700
            )

            heatmap_path = f"{output_dir}/Cluster_{cluster_number}_Heatmap.png"
            fig.write_image(heatmap_path)
            fig.show()



main_wind = window('main')
main_wind.window_.protocol('WM_DELETE_WINDOW',main_wind.close_)
main_wind.rest()
main_wind.add_button('prediction',20,1,2,yc)
main_wind.add_button('optimize',20,2,2,fx)
main_wind.add_button('analysis',20,3,2,yh)

main_wind.window_.mainloop()