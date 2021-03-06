# Calculation of π on GPU with Monte-Carlo method using CUDA

В ходе лабораторной работы реализованы алгоритмы вычисления числа π методом Монте-Карло.
Оба алгоритма написаны на языке C++, один использует CPU, другой GPU с применением CUDA.

Результаты сравнения времени выполнения вычисления π на CPU и GPU.
С увеличением числа точек N алгоритм с использованием CUDA вычисляет π быстрее, чем алгоритм на CPU:
| N    | CPU time, msec |  CPU π  | GPU time, msec |  GPU π  |  
| -----|----------------|---------|----------------|---------|
| 32   | 6 msec         | 3.12427 | 17.4889 msec   | 3.13428 |
| 100  | 32 msec        | 3.13758 | 18.817 msec    | 3.13586 |
| 500  | 136 msec       | 3.14146 | 24.7398 msec   | 3.13974 |
| 1000 | 316 msec       | 3.13898 | 34.944 msec    | 3.14198 |
| 2000 | 471 msec       | 3.13948 | 52.01 msec     | 3.14213 |
