import numpy as np

def test_function(x):
    return np.sin(1.1*np.pi*x)+4.5*np.cos(1.4*np.pi*x)

def generate_data(n_samples=100):
    x = np.random.rand(n_samples)
    y=test_function(x)
    return x, y

def relu(x):
    return np.maximum(0,x)

#relu的导数
def relu_derivative(x):
    return np.where(x>0,1,0)


class FunctionFitModel:
    def __init__(self,input_dim,hidden_dim,output_dim,learning_rate=0.01):
        self.learning_rate=learning_rate
        self.W1=np.random.randn(input_dim,hidden_dim)*0.1
        self.b1=np.zeros((1,hidden_dim))
        self.W2=np.random.randn(hidden_dim,output_dim)*0.1
        self.b2=np.zeros((1,output_dim))

    def forward(self,x):
        self.z1=x@self.W1+self.b1#第一层输出
        self.a1=relu(self.z1)#第一层激活
        self.z2=self.a1@self.W2+self.b2#第二层输出
        return self.z2

    def get_loss(self,y_pred,y_true):
        return np.mean(np.square(y_pred-y_true))#均方误差

    def backward(self,x,y_true,y_pred):
        m=x.shape[0]
        #进行反向传播，计算梯度
        #第二层
        dz2=(y_pred-y_true)/m
        dW2=self.a1.T@dz2#
        db2=np.sum(dz2,axis=0,keepdims=True)
        #第一层
        da1=dz2@self.W2.T
        dz1=da1*relu_derivative(self.z1)
        dW1=x.T@dz1
        db1=np.sum(dz1,axis=0,keepdims=True)

        #更新参数
        self.W2-=self.learning_rate*dW2
        self.b2-=self.learning_rate*db2
        self.W1-=self.learning_rate*dW1
        self.b1-=self.learning_rate*db1

    def train(self,x,y_true,epochs,loss_threshold=-1.):
        for i in range(epochs):
            y_pred=self.forward(x)#前向传播
            loss=self.get_loss(y_pred,y_true)#计算损失
            self.backward(x,y_true,y_pred)#反向传播
            if i%100==0:
                print(f'Epoch {i}, Loss: {loss}')
            if loss<loss_threshold:
                break

if __name__ == '__main__':
    x,y=generate_data(1000)
    model=FunctionFitModel(1,64,1,0.01)
    model.train(x[:,None],y[:,None],100000,0.005)
    x_test,y_test=generate_data(100)
    y_pred=model.forward(x_test[:,None])
    loss=model.get_loss(y_pred,y_test[:,None])
    print(f'Test Loss: {loss}')
    import matplotlib.pyplot as plt
    plt.scatter(x_test,y_test,label='True Data')
    plt.scatter(x_test,y_pred,label='Predictions')
    plt.legend()
    plt.show()


