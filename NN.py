import pandas
import re
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
import matplotlib.pyplot as plt

# Leer nuestro data set
df = pandas.read_csv('movie_review.csv')

x = df['text'] # Mensaje
y = df['tag'] # Clasificación

# Limpiamos datos
documents = []

stemmer = WordNetLemmatizer()

for i in range(0, len(x)):
    # Remover los caracteres especiales
    document = re.sub(r'\W', ' ', str(x[i]))

    # Removemos los caracteres solos
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

    # Sustituimos los multiples espacios seguidos con uno solo
    document = re.sub(r'\s+', ' ', document)

    # Convertimos todo a minusculas
    document = document.lower()

    # Lemantización
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

# Convertir el texto a números

Y = np.zeros((len(x),1))
for i in range(len(y)):
    if i == 'pos':
        Y[i] = 1.0
    else:
        Y[i] = 0.0

# Convertimos los mensajes a números con el modelo de "Bolsa de palabaras"
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vect.fit_transform(documents).toarray()

# Normalizamos los valores para que queden entre 0 y 1
from sklearn.feature_extraction.text import TfidfTransformer
t = TfidfTransformer()
X = t.fit_transform(X).toarray()

print("Data ready")

# Nuestro modelo de red neuronal

n = len(x)
p = 1500

class neural_layer():
    def __init__(self, n_conn, n_neur, act_f):
        self.act_f = act_f
        self.b = np.random.rand(1, n_neur)*2 - 1
        self.W = np.random.rand(n_conn, n_neur)*2 - 1

# Función de activación

sigm = (lambda x: 1 / (1 + np.e**(-x)),
        lambda x: x * (1 - x))

# Construcción de la red neuronal

def create_nn(topology, act_f):
    nn = []
    for l, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(topology[l], topology[l+1], act_f))
    return nn

topology = [p, 4, 2, 1]

#Entrenamiento
l2_cost = (lambda Yp, Yr: np.mean((Yp - Yr)**2), #Funcion de costo definida como el error cuadratico medio
            lambda Yp, Yr: (Yp - Yr))

def train(neural_net, X, Y, l2_cost, lr, train=True):
    # Paso hacia adelante
    out = [(None, X)]
    for l, layer in enumerate(neural_net):
        z = out[-1][1] @ neural_net[l].W + neural_net[l].b
        a = neural_net[l].act_f[0](z)
        out.append((z, a))

    if train:
        _W = None

        #Pasio hacia atras y decenso del gradiente
        deltas = []

        for l in reversed(range(0, len(neural_net))):
            z = out[l+1][0]
            a = out[l+1][1]

            if l == len(neural_net) - 1:
                #Calcular delta de la ultima capa    
                deltas.insert(0, (a - Y) * neural_net[l].act_f[1](a))
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
print("Trining data")
for i in range(1000):
    #Entrenamiento
    pY = train(neural_n, X, Y, l2_cost, lr=0.05)

    if i % 25 == 0:
        loss.append(l2_cost[0](pY, Y))
        plt.plot(range(len(loss)), loss)
        plt.grid()
        plt.savefig("T" + str(i) + ".png")

#Resultado del entrenamiento
print("Error: ", loss[-1]*100, "%")
print("Red Neuronal Entrenada")

while True:
    print("Ingrese una frase para probar")
    s = input()
    print("Tu msj es:", s)
    # Predicción
    documents2 = []

    stemmer2 = WordNetLemmatizer()
    x2 = x.to_list()
    x2.append(s)
    for i in range(0, len(x)):
        # Remover los caracteres especiales
        document = re.sub(r'\W', ' ', str(x[i]))

        # Removemos los caracteres solos
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Sustituimos los multiples espacios seguidos con uno solo
        document = re.sub(r'\s+', ' ', document)

        # Convertimos todo a minusculas
        document = document.lower()

        # Lemantización
        document = document.split()

        document = [stemmer2.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents2.append(document)

    # Convertimos los mensajes a números con el modelo de "Bolsa de palabaras"
    vect2 = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    X2 = vect2.fit_transform(documents2).toarray()

    # Normalizamos los valores para que queden entre 0 y 1
    t2 = TfidfTransformer()
    X2 = t2.fit_transform(X2).toarray()
    print(X2[-1])
    res = train(neural_n, np.array([X2[-1]]), Y, l2_cost, lr=0.05,train=False)[0][0]

    if res > 0.5:
        print("Es una buena reseña")
    else:
        print("Es una mala reseña")

    print(res)

    print("Fue una buena reseña? Y/N")
    op = input()

    pred = 1.0
    if(op == 'N'):
        pred = 0.0

    train(neural_n, np.array([X2[-1]]), np.array([pred]), l2_cost, lr=0.05)

    print("Realizar otra predicción? Y/N")
    op = input()
    if(op == 'N'):
        break

print("Fin de rutina")