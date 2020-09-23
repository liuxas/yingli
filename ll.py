from flask import Flask, request
from gevent.pywsgi import WSGIServer
from gevent import monkey
import os
import math
import keras
import tensorflow as tf
from keras import backend as k
import pandas as pd
import json
from sklearn.preprocessing import StandardScaler
import numpy as np
import time
import requests
from ipdb import set_trace

print(keras.__version__,tf.__version__)


# "异步加载"
# monkey.patch_all()

"""@lyy"""
step = 10  # 步长
sensor_point = "/temp/python_code/cloud/Flask_cloud/point_conf/sensor_point.csv"
normal_point = "/temp/python_code/cloud/Flask_cloud/point_conf/normal_point.csv"
sensor_df = pd.read_csv(sensor_point, index_col=0)
normal_df = pd.read_csv(normal_point, index_col=0)
"""禁用GPU"""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


"""选点算法"""
class choise(object):
    def __init__(self):
        self.name='ee'

    def choise_point(self, data):
        #data = np.load(data_path, allow_pickle=True)
        count_max_data = data[0]  #存储数据组中的第一组数据用来找最大值索引
        #将找最大值数组中的数全部取绝对值
        for i in range(len(count_max_data)):
            count_max_data[i] = abs(count_max_data[i])
        data_f = pd.DataFrame(data, dtype='float32')
        start_time = time.time()
        print("counting------")
        correlation_matrics = data_f.head(1000).corr()

        #记录选择点的索引列表
        index_choise = []
        #记录相关性点的索引列表
        index_correlation = []
        index_left = np.arange(0, 30897)
        while index_left!=[]:
            #循环遍历数据里面的每一个元素求最大值
            for i in range(len(count_max_data)):
                #记录最大值索引，并选择
                max_index = np.argmax(count_max_data)
                if count_max_data[max_index] == 0:
                    break
              #  print("max_index:", max_index)
                index_choise.append(max_index)
                #并将最大值清0，防止下次再检索到
                count_max_data[max_index] = -1
                index_left = list(set(index_left) - set([max_index]))
                for j in index_left:
                   # print("正在计算第", max_index, "个数与第", j, "个数的相关性")

                    if correlation_matrics[max_index][j] < -0.95 or correlation_matrics[max_index][j] > 0.95:
                        index_correlation.append(j)
                        count_max_data[j] = -1
                index_left = list(set(index_left) - set(index_correlation))
                index_correlation = []
               # print("----------------index_left长度:", len(index_left))
                #print("--------------------", index_left)
          #  print(index_choise)
            break
        return index_choise


"""选点算法"""
def s1_s2_choisepoint(path):
    data = pd.ExcelFile(path)
    array_sum1 = []
    array_sum2 = []
    for i, name in enumerate(data.sheet_names):
        if i < 30:
            table = data.parse(sheet_name=name, header=None)
            table1 = table.iloc[:, 1].values
            table2 = table.iloc[:, 2].values
            table1 = table1.tolist()
            table2 = table2.tolist()
            array_sum1.append(table1)
            array_sum2.append(table2)
        else:
            break
    array_sum1 = np.asarray(array_sum1)
    array_sum2 = np.asarray(array_sum2)
    cp = choise()
    index_s1 = cp.choise_point(array_sum1)
    index_s2 = cp.choise_point(array_sum2)
    index_res = index_s1[:15]
    for index in index_s2:
        if index not in index_res:
            index_res.append(index)
        if len(index_res) == 30:
            break
    return index_res


"""模型算法
1.df 列名可能有误
"""

#
global sess
global graph
sess = tf.Session()
graph = tf.get_default_graph()
k.set_session(sess)


# yty
class YtY(object):
    def __init__(self):
        self.name = "liux"

    def load_model(self,model_path):
        model = keras.models.load_model(model_path)
        return model

    def pre_processing(self, selected_index):
        node = np.load(selected_index)
        total_index = [i for i in range(30897)]
        node_index = node[0:20]
        other_node_index = node[20:]
        pre_node_index = list(set(total_index) - set(node_index))
        return node, total_index, node_index, other_node_index, pre_node_index

    # 得到标准化参数
    def data_processing(self, path, node_index, pre_node_index):
        data_array = np.load(path, allow_pickle=True)
        scaler1 = StandardScaler()
        scaler2 = StandardScaler()
        data_x = scaler1.fit_transform(data_array[:, node_index])
        data_y = scaler2.fit_transform(data_array[:, pre_node_index])
        x_train = data_x[0:70]
        x_test = data_x[70:99]
        y_train = data_y[0:70]
        y_test = data_y[70:99]
        # data_tuple = (predict_node_lenth,data_array_x_train,data_array_x_test,data_array_y_train,data_array_y_testp.
        return scaler1, scaler2

    def file_processing(self, node_values):
        node_values_s1 = node_values[:, 0].reshape(31, 1)
        node_values_s2 = node_values[:, 2].reshape(31, 1)
        node_values_s12 = node_values[:, 4].reshape(31, 1)
        node_values_total = node_values[:, 6].reshape(31, 1)
        return node_values_s1, node_values_s2,node_values_s12,node_values_total

    # 预测
    def predict_and_save(self, node_values, model, node_index, pre_node_index, other_node_index, scaler1, scaler2):
        # 预测
        # node_values = pd.read_csv(pre_file,index_col=0).values
        with graph.as_default():
            k.set_session(sess)
            detected_node = node_values[0:20].T
            detected_node_inverse = scaler1.transform(detected_node)
            pre = model.predict(detected_node_inverse)

        pre = scaler2.inverse_transform(pre)
        # 将预测值组织起来
        pre_array = np.empty((1, 30897), dtype=float)
        pre_array[:, node_index] = detected_node
        pre_array[:, pre_node_index] = pre
        # 将真实值组织起来
        real_array = np.empty((1, 30897), dtype=float)
        real_array[:, node_index] = detected_node
        real_array[:, pre_node_index] = pre
        real_array[:, other_node_index] = node_values[20:].T
        result_dict = {"real": real_array[0], "predict": pre_array[0]}
        result = pd.DataFrame.from_dict(result_dict)
        # 更新模型的数据预处理
        update_data = real_array[:, pre_node_index]
        update_y = scaler2.transform(update_data)
        return result, detected_node_inverse, update_y

    def save_result(self, result_s1, result_s2,result_s12,result_total):
        result_s1.columns = ["real_s1", "predict_s1"]  #
        result_s2.columns = ["real_s2", "predict_s2"]  #
        result_s12.columns = ["real_s12", "predict_s12"]
        result_total.columns = ["real_total", "predict_total"]

        result = pd.concat([result_s1, result_s2,result_s12,result_total], axis=1)
        return result

    def update_model(self, detected_node_inverse, update_y, model_path, model):
        # with graph.as_default():
        #     k.set_session(sess)
        model.fit(detected_node_inverse, update_y, epochs=10)
        model.save(model_path)
        print("The model update success!")
        return model


# ZtY
class ZtY(object):
    def __init__(self):
        self.name = "liux"

    def load_model(self, path):
        model = keras.models.load_model(path)
        return model

    def predict_and_save(self, inc, inc_next, model, node, node_values):
        # inc = inc.reshape(1,)
        # inc_next = inc_next_reshape(1,)
        with graph.as_default():
            k.set_session(sess)
            yl_predict = model.predict(inc)
            yl_real = yl_predict.copy()
            yl_real[:, node] = node_values
            yl_predict_next = model.predict(inc_next)
        return yl_real, yl_predict, yl_predict_next

    def save_result(self, s1_real, s1_pre, s1_pre_next, s2_real, s2_pre, s2_pre_next,
                    s12_real, s12_pre, s12_pre_next,total_real, total_pre, total_pre_next):
        result_s1 = {"s1_real": s1_real[0], "s1_predict": s1_pre[0]}
        result_s2 = {"s2_real": s2_real[0], "s2_predict": s2_pre[0]}
        result_s12 = {"s12_real": s12_real[0], "s12_predict": s12_pre[0]}
        result_total = {"total_real": s2_real[0], "total_predict": total_pre[0]}

        result_s1 = pd.DataFrame.from_dict(result_s1)
        result_s2 = pd.DataFrame.from_dict(result_s2)
        result_s12 = pd.DataFrame.from_dict(result_s12)
        result_total = pd.DataFrame.from_dict(result_total)
        result = pd.concat([result_s1, result_s2,result_s12,result_total], axis=1)
        result_next = {"s1_pre_next": s1_pre_next[0], "s2_pre_next": s2_pre_next[0],
                       "s12_pre_next": s12_pre_next[0],"total_pre_next": total_pre_next[0]}
        result_next = pd.DataFrame.from_dict(result_next)
        return result, result_next

    def update_model(self, model, inc, real,model_path):
        model.fit(inc, real,epochs=10)
        model.save(model_path)
        return model


# YtZ
class YtZ(object):
    def __init__(self):
        self.name = "liux"

    def load_model(self,model_path):
        model = keras.models.load_model(model_path)
        return model

    def predict_and_save(self, model, node_values, inc):
        with graph.as_default():
            k.set_session(sess)
            zh_pre = model.predict(node_values[0:20, 0].reshape(1, -1))
        return zh_pre



"""模型算法"""

#path
model_path = ["/temp/model/new", "/temp/model/old"]
yty_model = ["/temp/model/new/yty_s1_model.h5", "/temp/model/new/yty_s2_model.h5",
             "/temp/model/new/yty_s12_model.h5", "/temp/model/new/yty_total_model.h5"]
zty_model = ["/temp/model/new/zty_s1_model.h5", "/temp/model/new/zty_s2_model.h5",
             "/temp/model/new/zty_s12_model.h5", "/temp/model/new/zty_total_model.h5"]
ytz_model = ["/temp/model/new/ytz_total_model.h5"]
data_path_list = ["/temp/python_code/cloud/Flask_cloud/conf/data_s1.npy",
                  "/temp/python_code/cloud/Flask_cloud/conf/data_s2.npy",
                  "/temp/python_code/cloud/Flask_cloud/conf/data_s12.npy",
                  "/temp/python_code/cloud/Flask_cloud/conf/data_total.npy"]
selected_index = ["/temp/python_code/cloud/Flask_cloud/conf/selected_node.npy"]

#YtY
yty_obj = YtY()
yty_model_s1 = yty_obj.load_model(yty_model[0])
yty_model_s2 = yty_obj.load_model(yty_model[1])
yty_model_s12 = yty_obj.load_model(yty_model[2])
yty_model_total = yty_obj.load_model(yty_model[3])
# yty_model_s1.predict(np.random.random((1,20)))
# yty_model_s2.predict(np.random.random((1,20)))
node,total_index,node_index,other_node_index,pre_node_index= yty_obj.pre_processing(selected_index[0])
scaler1_s1,scaler2_s1= yty_obj.data_processing(data_path_list[0],node_index,pre_node_index)
scaler1_s2,scaler2_s2= yty_obj.data_processing(data_path_list[1],node_index,pre_node_index)
scaler1_s12,scaler2_s12= yty_obj.data_processing(data_path_list[2],node_index,pre_node_index)
scaler1_total,scaler2_total= yty_obj.data_processing(data_path_list[3],node_index,pre_node_index)
#ZtY
zty_obj = ZtY()
zty_model_s1 = zty_obj.load_model(zty_model[0])
zty_model_s2 = zty_obj.load_model(zty_model[1])
zty_model_s12 = zty_obj.load_model(zty_model[2])
zty_model_total = zty_obj.load_model(zty_model[3])
# zty_model_s1.predict(np.random.random((1,1)))
# zty_model_s2.predict(np.random.random((1,1)))


#YtZ
ytz_obj = YtZ()
ytz_model_total = ytz_obj.load_model(ytz_model[0])
# ytz_model_s1.predict(np.random.random((1,20)))

app = Flask(__name__)


@app.route("/set_step", methods=['POST'])
def set_step():
    try:
        step_msg = int(request.form.get("step"))
        global step
        step = step_msg
        return json.dumps({"code": 200, "msg": "success"})
    except:
        return json.dumps({"code": 500, "msg": "参数不合法"})



global num
num=1


@app.route("/predict", methods=['POST'])
def get_predict():
    # 全局变量
    global yty_model_s1,yty_model_s2,yty_model_s12,yty_model_total,\
        zty_model_s1,zty_model_s2,zty_model_s12,zty_model_total,ytz_model_total
    
    num+=1
    # 读数据
    inc = request.form.get("inc")  # 字符串类型"50" 需要转为int
    file_path = request.form.get("file_path")  # 先伪造一份,路径格式　如下"./data/2020-08-27/1597763554.610886/upload/real.csv"
    save_path = file_path.rsplit("/", 2)[0]


    yty_save_path = os.path.join(save_path, "predict")
    ytz_save_path = yty_save_path
    zty_save_path = os.path.join(save_path, "predict_f")
    node_values = pd.read_csv(file_path, index_col=0).values
    # model1
    """1.应力预测应力用于画云图
        落入磁盘路径file_path路径的upload改为predict,注意存储格式csv"""
    # 预测并保存结果
    node_values_s1, node_values_s2,node_values_s12,node_values_total = yty_obj.file_processing(node_values)
    result_s1, detected_node_inverse_s1, update_y_s1 = yty_obj.predict_and_save(node_values_s1, yty_model_s1,
                                                                                node_index, pre_node_index,
                                                                                other_node_index, scaler1_s1,
                                                                                scaler2_s1)

    result_s2, detected_node_inverse_s2, update_y_s2 = yty_obj.predict_and_save(node_values_s2, yty_model_s2,
                                                                                node_index, pre_node_index,
                                                                                other_node_index, scaler1_s2,
                                                                                scaler2_s2)
    result_s12, detected_node_inverse_s12, update_y_s12 = yty_obj.predict_and_save(node_values_s12, yty_model_s12,
                                                                                node_index, pre_node_index,
                                                                                other_node_index, scaler1_s12,
                                                                                scaler2_s12)
    result_total, detected_node_inverse_total, update_y_total = yty_obj.predict_and_save(node_values_total, yty_model_total,
                                                                                node_index, pre_node_index,
                                                                                other_node_index, scaler1_total,
                                                                                scaler2_total)
    result = yty_obj.save_result(result_s1, result_s2,result_s12,result_total)
    result.index = result.index + 1  # 索引加1
    """报警文件生成lyy"""
    sensor_mises = result.loc[sensor_df.index,].apply(lambda x: math.sqrt(
        0.5 * ((x[0] - x[2]) ** 2 + x[0] ** 2 + x[2] ** 2 + 6 * ((0.5 * (x[0] + x[2])) ** 2))), axis=1)  #合力
    normal_mises = result.loc[normal_df.index,].apply(lambda x: math.sqrt(
        0.5 * ((x[0] - x[2]) ** 2 + x[0] ** 2 + x[2] ** 2 + 6 * ((0.5 * (x[0] + x[2])) ** 2))), axis=1)

    a = np.where(sensor_mises > sensor_df.threshold1, 1, 0)
    b = np.where(sensor_mises > sensor_df.threshold2, 2, 0)
    c = np.where(sensor_mises > sensor_df.threshold3, 3, 0)

    a1 = np.where(normal_mises > normal_df.threshold1, 1, 0)
    b1 = np.where(normal_mises > normal_df.threshold2, 2, 0)
    c1 = np.where(normal_mises > normal_df.threshold3, 3, 0)


    sensor_result = pd.DataFrame(index=sensor_mises.index, data=zip(a, b, c)).apply(lambda x: max(x), axis=1)
    normal_result = pd.DataFrame(index=normal_mises.index, data=zip(a1, b1, c1)).apply(lambda x: max(x), axis=1)
    logging_warning = os.path.join(save_path, "logging_warning")  # loging文件路径
    sensor_result.to_csv(os.path.join(logging_warning, "sensor_warning.csv"), header=['level'])
    normal_result.to_csv(os.path.join(logging_warning, "normal_warning.csv"), header=['level'])
    ## //


   # result.to_csv(os.path.join(yty_save_path, "yty_result.csv"))
    # 移动模型到old
    os.system("cp %s %s" %(yty_model[0],model_path[1]))
    os.system("cp %s %s" %(yty_model[1],model_path[1]))
    # model2
    """应力+步长预测载荷用于前端更新
    s = {"code": 200, "predict_inc":predict_inc}
    data = json.dump(s)
    将data返回给我"""
    predict_inc = ytz_obj.predict_and_save(ytz_model_total, node_values, inc)
    predict_inc = predict_inc[0, 0].tolist()

    "保持模型及其预测载荷"
    result.to_csv(os.path.join(yty_save_path, "yty_result{}.csv".format(predict_inc)))

    s = {"code": 200, "predict_inc": predict_inc}
    data = json.dumps(s)

    res = requests.post("http://0.0.0.0:5555/web/set_predict_inc", data={"predict_inc": predict_inc}, timeout=5)
    if res.status_code != 200:
        return json.dumps({"code": 200, "msg": "端口错误请检查(5555,6666)"})
    # model3
    """载荷预测应力用于画曲线图
        落入磁盘路径file_path路径的upload改为predict_f,注意存储格式
    """
    # 预测并保存结果
    inc = np.array([inc], dtype="float32")
    inc_next = inc + step
    s1_real, s1_pre, s1_pre_next = zty_obj.predict_and_save(inc, inc_next, zty_model_s1, node, node_values[:, 0])
    s2_real, s2_pre, s2_pre_next = zty_obj.predict_and_save(inc, inc_next, zty_model_s2, node, node_values[:, 2])
    s12_real, s12_pre, s12_pre_next = zty_obj.predict_and_save(inc, inc_next, zty_model_s12, node, node_values[:, 4])
    total_real, total_pre, total_pre_next = zty_obj.predict_and_save(inc, inc_next, zty_model_total, node, node_values[:, 6])
    result, result_next = zty_obj.save_result(s1_real, s1_pre, s1_pre_next, s2_real, s2_pre, s2_pre_next,
                                              s12_real, s12_pre, s12_pre_next,total_real, total_pre, total_pre_next)
    result.index = result.index + 1
    result_next.index = result_next.index + 1
    #result.to_csv(os.path.join(zty_save_path, "zty_result.csv"))
    result_next.to_csv(os.path.join(zty_save_path, "zty_result_next{}.csv".format(inc_next[0])))
    # 移动模型
    os.system("cp %s %s" %(zty_model[0], model_path[1]))
    os.system("cp %s %s" %(zty_model[1], model_path[1]))
    # 更新模型
    # YtY
    if num%10==0:
        yty_model_s1 = yty_obj.update_model(detected_node_inverse_s1, update_y_s1, yty_model[0], yty_model_s1)
        yty_model_s2 = yty_obj.update_model(detected_node_inverse_s2, update_y_s2, yty_model[1], yty_model_s2)
        yty_model_s12 = yty_obj.update_model(detected_node_inverse_s12, update_y_s12, yty_model[2], yty_model_s12)
        yty_model_total = yty_obj.update_model(detected_node_inverse_total, update_y_total, yty_model[3], yty_model_total)
        # ZtY
        zty_model_s1 = zty_obj.update_model(zty_model_s1, inc, s1_real, zty_model[0])
        zty_model_s2 = zty_obj.update_model(zty_model_s2, inc, s2_real, zty_model[1])
        zty_model_s12 = zty_obj.update_model(zty_model_s12, inc, s12_real, zty_model[2])
        zty_model_total = zty_obj.update_model(zty_model_total, inc, total_real, zty_model[3])
    return data


@app.route("/choce_point", methods=['POST'])
def choice_point():
    """文件格式excel"""
    print("调用接口")
    file_path = request.form.get("file_path")  # 先伪造一份,路径格式　如下"./data/2020-08-27/1597763554.610886/upload/real.csv"
    print(file_path)
    index_res = s1_s2_choisepoint(file_path)
    index_res = map(lambda x: int(x)+1, index_res)  # 索引+1


    return json.dumps({"code": 200, "result": list(index_res)})

if __name__ == '__main__':
    # server = WSGIServer(("0.0.0.0", 7777
    #                      ), app)
    # server.serve_forever()
    app.run("0.0.0.0", 7777, threaded=True)

