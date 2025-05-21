import ctypes
import os

class AutophaseDataStruct(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 64), ("value", ctypes.c_int)]

def get_autophase_obs(ir_file_path, llvm_version="llvm-10.0.0"):
    project_directory = os.path.dirname(os.path.abspath(__file__))
    library_path = None
    library_path = os.path.join(project_directory, './', 'libAutophase_10_0_0.so')
    
    result_array = (AutophaseDataStruct * 56)()
    autophase_lib = ctypes.CDLL(library_path)

    autophase_lib.GetAutophase(ir_file_path.encode(), result_array)
    result_dict = {item.name.decode(): item.value for item in result_array}

    return result_dict