from timeit import default_timer as timer

def print_exectime(f, name):
    start = timer()
    ret = f()
    print(f'{name} {timer() - start} s')
    return ret