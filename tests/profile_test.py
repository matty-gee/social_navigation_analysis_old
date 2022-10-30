from memory_profiler import profile


# from memscript import my_func
# %mprun -T mprof0 -f my_func my_func()
# print(open('mprof0', 'r').read())


# example:
# from memory_profiler import profile

# @profile
# def main_func():
#     import random
#     arr1 = [random.randint(1,10) for i in range(100000)]
#     arr2 = [random.randint(1,10) for i in range(100000)]
#     arr3 = [arr1[i]+arr2[i] for i in range(100000)]
#     tot = sum(arr3)
#     print(tot)

# if __name__ == "__main__":
#     main_func()

# python -m memory_profiler example1_modified.py

