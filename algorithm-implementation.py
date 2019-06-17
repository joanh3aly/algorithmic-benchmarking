'''
G00007918
Benchmarking Algorithms Project Code
'''

import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas import plotting
import numpy as np
import random
import seaborn as sns
from scipy import stats

''' Create class to benchmark algorithms '''
class BenchmarkClass():

  def randomListGen(self,inputSize):
    return np.random.randint(inputSize, size=inputSize)

  def bubbleSort(self,randomList): 
    outer_start = randomList.size - 1
    for passnum in range(outer_start,0,-1): 
      for i in range(passnum): 
        if randomList[i]>randomList[i+1]:
          temp = randomList[i]
          randomList[i] = randomList[i+1]
          randomList[i+1] = temp

          
  def selectionSort(self,randomList):
    print('selectionSort input list length ',len(randomList))
    outer_start = randomList.size - 1
    for fillslot in range(outer_start,0,-1):
      positionOfMax=0
      for location in range(1,fillslot+1):
          if randomList[location]>randomList[positionOfMax]:
              positionOfMax = location
      temp = randomList[fillslot]
      randomList[fillslot] = randomList[positionOfMax]
      randomList[positionOfMax] = temp
   
   
  def insertionSort(self,randomList):
    print('insertionsort input list length ',len(randomList))
    outer_start = randomList.size 
    
    for index in range(1,outer_start):
      currentvalue = randomList[index]
      position = index

      while position>0 and randomList[position-1]>currentvalue:
        randomList[position]=randomList[position-1]
        position = position-1
        randomList[position]=currentvalue
    


  def countingSort(self, randomList): 
    n = len(randomList)
    randomList = randomList.tolist() 
    
    # The output array elements that will have sorted arr 
    output = [0] * (n) 
    # initialize count array as 0 
    count = [0] * (n) 
    resultList = [0] * (n) 
    resultList2 = []
    
    # Store count of occurrences in count[] 
    for i in range(0, n): 
      index = (randomList[i]) 
      count[ index ] += 1
    # Change count[i] so that count[i] now contains actual 
    #  position of this digit in output array 
    for i in range(1,n): 
      count[i] += count[i-1] 
  
    # Build the output array 
    for i in range(len(randomList)):  
      index = (randomList[i]) 
      output[ count[ index ] - 1] = randomList[i] 
      count[ index ] -= 1
      
    # Copying the output array to arr[], 
    # so that arr now contains sorted numbers 
    i = 0
    for i in range(0,len(randomList)): 
      resultList[i] = output[i] 
      
    
  def quickSort(self,randomList):
    self.quickSortHelper(randomList,0,len(randomList)-1)

  def quickSortHelper(self, randomList,first,last):
    if first<last:

      splitpoint = self.partition(randomList,first,last)

      self.quickSortHelper(randomList,first,splitpoint-1)
      self.quickSortHelper(randomList,splitpoint+1,last)


  def partition(self,randomList,first,last):
    pivotvalue = randomList[first]

    leftmark = first+1
    rightmark = last

    done = False
    while not done:

      while leftmark <= rightmark and randomList[leftmark] <= pivotvalue:
        leftmark = leftmark + 1

      while randomList[rightmark] >= pivotvalue and rightmark >= leftmark:
        rightmark = rightmark -1

      if rightmark < leftmark:
        done = True
      else:
        temp = randomList[leftmark]
        randomList[leftmark] = randomList[rightmark]
        randomList[rightmark] = temp

    temp = randomList[first]
    randomList[first] = randomList[rightmark]
    randomList[rightmark] = temp
    
    return rightmark


  def chooseAlgo(self,algoType,randomList): 
    if algoType == 'bubble_sort':
      print('chosen algorithm: ', algoType)
      return self.bubbleSort(randomList)
    elif algoType == 'selection_sort':
      print('chosen algorithm: ', algoType)
      return self.selectionSort(randomList)
    elif algoType == 'insertion_sort':
      print('chosen algorithm: ', algoType)
      return self.insertionSort(randomList)    
    elif algoType == 'count_sort':
      print('chosen algorithm: ', algoType)
      return self.countingSort(randomList)
    elif algoType == 'quick_sort':
      print('chosen algorithm: ', algoType)
      return self.quickSort(randomList)  
    

  def benchmark(self,numRuns,randomListLength,algoType):
    outputAggregate = 0
    generatedRandomList = []

    for run in range(numRuns):
      print(' ')
      print(' ')
      generatedRandomList = self.randomListGen(randomListLength)
      print('Run number: ', run)
      print('random list length in benchmark: ', len(generatedRandomList))
      print('random list benchmark() : ', generatedRandomList[:10])
      # Get start time in seconds #
      startTime = time.time()

      self.chooseAlgo(algoType,generatedRandomList)
      #self.chooseAlgo(algoType,'hello')

      endTime = time.time()
      # Subtract the start and end times to get the difference #
      timeElapsed = endTime - startTime
      print('time elapsed {}'.format(timeElapsed))
      # Sum the speed of each algorithm #
      outputAggregate += timeElapsed
      print('output aggregate {}'.format(outputAggregate))

    # Convert seconds to milliseconds and get the average speed of 10 runs #
    averageOutput = (outputAggregate/10)*1000
    print('')
    print('')
    # Round the speed output values to 3 decimal places to make them more readible #
    averageOutputRound = round(averageOutput,3)
    print('average_output ', averageOutputRound)
    return averageOutputRound
    

  def sizeLoop(self,numRuns,maxInputSize,algoType):
    algoSpeedArray = []
    algoSizeArray = []
    minInputSize = 250
    inputIncrement = 500
    countInputSizes = 0  

    for listLength in range(minInputSize,maxInputSize,inputIncrement):
      benchmarked = self.benchmark(numRuns,listLength,algoType)
      algoSpeedArray.append(benchmarked)
      algoSizeArray.append(listLength)
      countInputSizes += 1
      print('countInputSizes ', countInputSizes)
     
    print('algo Speed {}'.format(algoSpeedArray))
    print('Size {}'.format(algoSizeArray))

    df = pd.DataFrame({'Algorithm_type': algoType,
                      'Algorithm_size': algoSizeArray,
                      'Algorithm_speed': algoSpeedArray}) 
    print(df)
    return df
    
  def loopThroughAlgorithms(self,numRuns,maxInputSize,algoArray):
    collectiveOutputDf = pd.DataFrame()
    for algoType in algoArray:
      algoOutput = self.sizeLoop(numRuns,maxInputSize,algoType)
      
      collectiveOutputDf['algorithm_size'] = algoOutput['Algorithm_size']
      collectiveOutputDf[algoType] = algoOutput['Algorithm_speed']

    return collectiveOutputDf
     
  def plotOutput(self,algo_speed_results_df):     
    plt.style.use('seaborn-whitegrid')
    plt.title("Algorithm Speed")
    plt.xlabel("Input size n")
    plt.ylabel("Running time in milliseconds")

    plt.plot(algo_speed_results_df.algorithm_size, algo_speed_results_df.bubble_sort, '-c') 
    plt.plot(algo_speed_results_df.algorithm_size, algo_speed_results_df.selection_sort, '-k') 
    plt.plot(algo_speed_results_df.algorithm_size, algo_speed_results_df.insertion_sort,'-r')  
    plt.plot(algo_speed_results_df.algorithm_size, algo_speed_results_df.count_sort, '-b')  
    plt.plot(algo_speed_results_df.algorithm_size, algo_speed_results_df.quick_sort,'-m') 

    plt.legend()
    plt.show()
   

''' Input values here '''
input_size = 10000
num_runs = 10
algo_array = ['bubble_sort','selection_sort','insertion_sort','count_sort','quick_sort']

''' Run benchmarking class, print and plot results '''
benchmark1 = BenchmarkClass()
algo_speed_results_df = benchmark1.loopThroughAlgorithms(num_runs,input_size,algo_array)
benchmark1.plotOutput(algo_speed_results_df)
print(algo_speed_results_df)

''' Store results in CSV '''
export_csv = algo_speed_results_df.to_csv (r'/Users/joanhealy1/Documents/GMIT-algorithms/project/algorithm-speeds/combined_algo_speeds-5.csv', index = None, header=True) 



