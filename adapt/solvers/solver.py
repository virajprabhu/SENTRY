import torch

solvers = {}
def register_solver(name):
    def decorator(cls):
        solvers[name] = cls
        return cls
    return decorator

def get_solver(name, *args):
    solver = solvers[name](*args)
    return solver