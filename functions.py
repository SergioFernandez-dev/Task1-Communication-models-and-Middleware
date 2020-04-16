#!/bin/env python

import numpy as np
import pickle
import time as t
import pywren_ibm_cloud as pywren
from backend import COSBackend

# Llista de paràmetres necessaris per a poder executar el codi
# [Files mat1, cols mat1 i files mat2, cols mat2, rang de generació de nums, num. workers]

lista = [1000, 1000, 1000, 100, 5]

def intialize_matrix(n, m, l, rang, num):
    cos = COSBackend()
    iterdata = []
    aux = num

    # Comprovem si el numero de workers es major a la mida de les matrius
    # Si no hi ha tants workers com posicions ( 1 multiplicació per worker )
    # Adaptem el num. workers a numero de la matriu més petita
    # Garantint que cada worker té minim una fila sencera

    if (num > n or num > l) and num != n * l:
        num = min(n, l)
        num2 = num

    # Si el numero de workers es el mateix que posicions
    # Cada worker fa només una multiplicació

    elif num == n * l:
        num = n
        num2 = l
    else:
        num2 = num

    # Generem les matrius A i B amb nombres aleatoris dins del rang definit
    # Seguidament les partim en tantes submatrius com workers tinguem

    matriz = [[(np.random.randint(rang)) for i in range(m)] for j in range(n)]
    matrizB = [[(np.random.randint(rang)) for i in range(l)] for j in range(m)]
    array = np.array_split(matriz, num)
    array2 = np.array_split(np.transpose(matrizB), num2)

    # Un cop subdividides les pujem al cloud

    for i in range(num):
        name = "fil" + str(i)
        cos.put_object('deposito-willy', name,
                       pickle.dumps(array[i], pickle.HIGHEST_PROTOCOL))

    for j in range(num2):
        name = "col" + str(j)
        cos.put_object('deposito-willy', name,
                       pickle.dumps(np.transpose(array2[j]), pickle.HIGHEST_PROTOCOL))

    # Ademés creem l'iterdata per a la funció del map_reduce
    # En el cas d'una multiplicació per worker, guardem només una tupla a cada
    # posició de l'iterdata

    if aux > n:
        for i in range(num):
            for j in range(num2):
                array = []
                array.append("fil" + str(i))
                array.append("col" + str(j))
                iterdata.append([array])

    # En l'altre cas guardem la llista de tuples pertinent a cada posició

    else:
        for i in range(num):
            array = []
            for j in range(num2):
                array.append("fil" + str(i))
                array.append("col" + str(j))
            iterdata.append([array])
    return iterdata


def map_func(array):
    res = []
    cos = COSBackend()

    # Per a cada tupla de la posició de l'iterdata que rebem baixem la submatriu
    # del cloud corresponent, fem la multiplicació i la concatenem en una llista
    # de tots els resultats de les multiplicacions que ha fet el worker

    for i in range(len(array)):
        if (i % 2) == 0:
            aux = cos.get_object('deposito-willy', array[i])
            matrix1 = pickle.loads(aux)
            aux2 = cos.get_object('deposito-willy', array[i + 1])
            matrix2 = pickle.loads(aux2)
            res = np.append(res, np.dot(matrix1, matrix2))
    return res


def reduce_func(results):
    array1d = []
    cos = COSBackend()

    # Per últim concatenem tots els resultats de la funció map en una llista
    # i la formatem a la mida de la matriu pertinent per a finalment pujar-la
    # al cloud

    for map_result in results:
        array1d = np.append(array1d, map_result)
    array2d = np.reshape(array1d, (lista[0], lista[2]))
    cos.put_object('deposito-willy', 'matrizFinal',
                   pickle.dumps(array2d, pickle.HIGHEST_PROTOCOL))
    return array2d



if __name__ == "__main__":

    # Creem una variable a l'inici per al temps inicial
    # Fem la crida asíncrona a la funció d'inicialitzar
    # I seguidament cridem al map_reduce i esperem que acabi l'execució

    pw = pywren.ibm_cf_executor()
    pw.call_async(intialize_matrix, lista)
    iterdata = pw.get_result()
    start_time = t.time()
    futures = pw.map_reduce(map_func, iterdata,reduce_func)
    pw.wait(futures)

    # Un cop acabada l'execució calculem el temps

    elapsed_time = t.time() - start_time
    print("Time: {0:.2f} secs. ".format(elapsed_time))
    print(pw.get_result())
