# CS3450 
##### Mitchell Johnstone
##### 28 May 2023 
##### Lab 1: Predicting Runtimes
----

## Introduction

The ROSIE supercomputer at MSOE includes a variety of hardware, and this 
lab aims to help the students in comparing the different performances. 
This will include comparing the relative performance specs of an NVIDIA
V100 and the T4 nodes that are available to students.
The comparison will be done but theoretically by comparing the 
hardware specs, and experimentally by testing some jobs on each.


## Predict Relative Runtime from GPU Specs

TFLOP -> single-precision floating point operation

NVIDIA V100: 16.4 TFLOPS
T4: 8.1 TFLOPS

Since the NVIDIA can do more floating point operations per second, I'd
think it would be almost double the speed / efficiency of the T4.


## Experimentally determine runtime from running one epoch
| Node used | number of CPUS | Total Runtime (s) |
|-|-|-|
| T4 | 2 | 1444.9 |
| T4 | 16 | 531.1 |
| T4 & GPU | 8 | 65.0 |
| DGX & GPU | 8 | 37.4 |

Damn, the DGX is fast.


## Predict Runtime for 20 epochs
Well, since we're running for 20 epochs, I'd expect each to take 20 times as long.
| Node used | number of CPUS | Total Expected Runtime (s) |
|-|-|-|
| T4 | 2 | 1444.9*20 = 28898s = 8.027h |
| T4 | 16 | 531.1*20 = 10622s = 2.95h |
| T4 & GPU | 8 | 65.0*20 = 1300s = 21m |
| DGX & GPU | 8 | 37.4*20 = 748s = 12m |


## Experimentally determine runtime from running 20 epochs
I ran these using TMUX with multiple windows to enhance the coding 
process and become more familiar with tmux.
Dang, these guys were slow :(
| Node used | number of CPUS | Total Runtime (s) |
|-|-|-|
| T4 | 2 | 28138.8s |
| T4 | 16 | 10658.7s |
| T4 & GPU | 8 | 1347s |
| DGX & GPU | 8 |  756s |

In general, the results were fairly what I expected, only off by a few minutes at the most. 




