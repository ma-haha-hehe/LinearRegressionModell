import torch
import torch.nn as nn
import numpy as np

#构建输入数据x和其对应的标签y
x_values = [i for i in range(11)] #生成从0到10的整数列表
x_train = np.array(x_values,dtype=np.float32) #x_value转成numpy
x_train = x_train.reshape(-1,1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values,dtype=np.float32)
y_train = x_train.reshape(-1,1)

#线性回归模型
class LinearRegressionModel(nn.Module): #继承nn模块中Model模块
    def __init__(self,input_dim,output_dim): #初始化
        super(LinearRegressionModel,self).__init__() #调用父类构造函数
        self.linear = nn.Linear(input_dim,output_dim) #创建一个线性层

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim,output_dim)

#指定好参数和损失函数
epochs = 1000        #训练的次数
learning_rate = 0.01 #学习率
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate) #随机梯度下降（SGD）作为优化方法
criterion = nn.MSELoss() #均方误差损失函数

#训练模型
for epoch in range(epochs):
    epoch += 1 #记录当前训练轮次

    #转换为tensor
    input = torch.from_numpy(x_train) #将Numpy数组转换为Pytorch张量
    labels = torch.from_numpy(y_train)

    #梯度要清零
    optimizer.zero_grad()

    #前向传播
    outputs = model(input)

    #计算损失
    loss = criterion(outputs, labels)

    #反向传播
    loss.backward()

    #更新权重参数
    optimizer.step()
    if epoch % 100 == 0:
        print(f'epoch {epoch}, loss {loss.item():.4f}')

#模型预测结果
predicted = model(torch.from_numpy(x_train).requires_grad_()).data.numpy()
print(predicted)

#模型的保存和读取
torch.save(model.state_dict(),'model.pkl') #保存 模型的权重参数
model.load_state_dict(torch.load('model.pkl')) #读取



