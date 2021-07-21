import sys
sys.path.append('..')
#from .. import utils
import utils

def func_name(a1, a2):
    return a1+a2

def test_func():
    assert func_name(3,4) == 5

def test_func2():
    assert func_name(3,2) == 5

