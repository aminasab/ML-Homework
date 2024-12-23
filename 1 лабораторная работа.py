import numpy as np
from typing import List

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

class Neuron:
    """
    Класс, реализующий нейрон.
    """
    def __init__(self, a: float, b: float, c: float, prob_output: bool = True):
        """
        a, b, c -- коэффициенты (веса) нейрона
        prob_output -- если True, то на выходе -- вероятности, если False -- логит
        """
        self.a = a
        self.b = b
        self.c = c
        self.prob_output = prob_output

    def calculate_logit(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов. 
             Первый столбец -- признак x1, второй -- x2.

        Данный метод должен возвращать logit = a*x1 + b*x2 + c
        """  
        assert np.ndim(x) == 2 and x.shape[1] == 2
        logit = self.a * x[:, 0] + self.b * x[:, 1] + self.c
        return logit

    def call(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов. 
             Первый столбец -- признак x1, второй -- x2.

        Данный метод должен возвращать logit(x), если self.prob_output=False,
        и sigmoid(logit(x)), иначе
        """  
        assert np.ndim(x) == 2 and x.shape[1] == 2
        logit = self.calculate_logit(x)
        
        if self.prob_output:
            output = sigmoid(logit)
        else:
            output = logit
            
        assert output.shape == (x.shape[0],), f"Output size must have following shape: {[x.shape[0],]}. Received: {output.shape}"
        return output
    
    def predict_class(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов. 
             Первый столбец -- признак x1, второй -- x2.

        Данный метод должен возвращать предсказанный класс для 
        каждого из N объектов -- 0 или 1.
        """
        logit = self.calculate_logit(x)
        predicted_classes = (logit > 0.0).astype(np.int32)

        assert predicted_classes.shape == (x.shape[0],), f"Output size must have following shape: {[x.shape[0],]}. Received: {predicted_classes.shape}"
        return predicted_classes

    def __repr__(self):
        return f"Neuron description. Weights: a={self.a}, b={self.b}. Bias: c={self.c}."


class ThreeNeuronsNeuralNet(Neuron):
    """
    Нейронная сеть из трех нейронов.
    """
    def __init__(self, first_neuron_params: List[float],
                 second_neuron_params: List[float],  
                 third_neuron_params: List[float]):
        """
        Для конструирования нейронной сети нам потребуются параметры трех нейронов,
        которые передаются в трех списках.

        Мы наследуемся от класса Neuron, т.к. нам нужно переопределить только 
        пересчет логитов. Предсказания классов и вероятностей уже реализованы.
        """
        self.prob_output = True  # фиксируем вероятностный выход
        self.neuron1 = Neuron(*first_neuron_params, prob_output=True)  # конструируем первый нейрон
        self.neuron2 = Neuron(*second_neuron_params, prob_output=True)  # конструируем второй нейрон
        self.neuron3 = Neuron(*third_neuron_params, prob_output=self.prob_output)  # конструируем третий нейрон

    def calculate_logit(self, x: np.ndarray) -> np.ndarray:
        """
        x -- массив размера (N, 2), где N -- количество объектов. 
             Первый столбец -- признак x1, второй -- x2.
             Важно! Это исходные координаты!

        Этот метод должен вернуть логит предсказанный всей сетью.
        Это можно сделать в 4 шага:
        1) Получить вероятности синего класса для исходных данных первым 
           нейроном: вектор длины N -- z1
        2) Получить вероятности синего класса для исходных данных вторым
           нейроном: вектор длины N -- z2
        3) Склеить полученные вероятности: массив размера (N, 2) -- z1z2
           * вам может быть полезна функция np.vstack
        4) Получить логит(!), calculate_logit третьего нейрона, примененного к z1z2 -- logit
        """
        
        # Шаг 1: Получаем вероятности от первого нейрона
        z1 = self.neuron1.call(x)  # Метод call возвращает вероятности

        # Шаг 2: Получаем вероятности от второго нейрона
        z2 = self.neuron2.call(x)  # То же самое для второго нейрона

        # Шаг 3: Склеиваем полученные вероятности по вертикали
        z1z2 = np.vstack((z1, z2)).T  # Объединяем по строкам и транспонируем для получения нужной формы

        # Шаг 4: Получаем логит от третьего нейрона
        logit = self.neuron3.calculate_logit(z1z2)  # Вычисляем логит третьего нейрона

        return logit
