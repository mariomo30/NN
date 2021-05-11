import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles 
from matplotlib.gridspec import GridSpec

n = 500
p = 2

#Dataset
X, Y = make_circles(n_samples=n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]
print(X)
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")
plt.axis("equal")
plt.show()

#Calse que representa la capa de la red nueronal
class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)*2 - 1
        self.W = np.random.rand(n_conn, n_neur)*2 - 1

#Funciones de activación
sigm = (lambda x: 1 / (1 + np.e**(-x)), #Funcion sigmoide
        lambda x: x * (1 - x))

#Construccion de la Ren Nueronal
def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))
    return nn

topology = [p, 4, 8, 1]
neural_net = create_nn(topology, sigm)

#Entrenamiento
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr)**2), #Funcion de costo definida como el error cuadratico medio
            lambda Yp, Yr: (Yp - Yr))

def train(neural_net, X, Y, l2_cost, lr=0.5, train=True):
    #Paso hacia adelante
    out = [(None, X)]
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)

        out.append((z, a))
    #print("Prediccion(Error):", l2_cost[0](out[-1][1], Y)*100, "%")
    
    if train:
        _W = None
        #Pasio hacia atras y decenso del gradiente
        deltas = []

        for l in reversed(range(0, len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]
            if l == len(neural_net) - 1:
                #Calcular delta de la ultima capa    
                deltas.insert(0, l2_cost[1](a, Y) * neural_net[l].act_f[1](a))
                print("a", neural_net[l].act_f[1](a).shape)
            else:
                #Calcular delta anterior
                deltas.insert(0, deltas[0] @ _W.T * neural_net[l].act_f[1](a))

            _W = neural_net[l].W
            #decenso del gradiente
            neural_net[l].b = neural_net[l].b - np.mean(deltas[0], axis=0, keepdims=True) * lr
            neural_net[l].W = neural_net[l].W - out[l][1].T @ deltas[0] * lr

    return out[-1][1]

#Implementacion
    
neural_n = create_nn(topology, sigm)
loss = []
pY = train(neural_n, X, Y, l2_cost, lr=0.05)
'''

for i in range(2500):
    #Entrenamiento
    pY = train(neural_n, X, Y, l2_cost, lr=0.05)

    if i % 25 == 0:
        loss.append(l2_cost[0](pY, Y))
        res = 50

        _x0 = np.linspace(-1.7, 1.7, res)
        _x1 = np.linspace(-1.7, 1.7, res)

        _Y = np.zeros((res, res))

        for i0, x0 in enumerate(_x0):
            for i1, x1 in enumerate(_x1):
                _Y[i0, i1] = train(neural_n, np.array([[x0, x1]]), Y, l2_cost, train=False)[0][0]
                
        
        fig=plt.figure()
        gs=GridSpec(1, 2)
        
        ax = fig.add_subplot(gs[0,0])
        ay = fig.add_subplot(gs[0,1], )
        
        ax.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
        ax.axis("equal")
        ax.scatter(X[Y[:,0] == 0, 0], X[Y[:, 0] == 0, 1], c="skyblue")
        ax.scatter(X[Y[:,0] == 1, 0], X[Y[:, 0] == 1, 1], c="salmon")

        ay.grid()
        ay.plot(range(len(loss)), loss)
        fig.savefig("T" + str(i) + ".png")

print(loss[-1]*100)
'''