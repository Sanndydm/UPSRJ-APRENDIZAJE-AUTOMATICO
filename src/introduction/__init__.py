# Álgebra lineal y optimización
from .intro_scipy import solve_linear
from .intro_scipy import get_matrix_properties
from .intro_scipy import find_min

# Estadística y análisis de datos
from .intro_numpy import get_statistics as get_statistics_numpy
from .intro_scipy import get_statistics as get_statistics_scipy

# Señales y espectros
from .intro_scipy import get_spectrum
from .intro_scipy import low_pass_filter

# Lectura y manipulación de archivos
from .intro_pandas import csv_registers
from .intro_pandas import json_registers
from .intro_pandas import yaml_registers
from .intro_pandas import get_head
from .intro_pandas import get_above
from .intro_pandas import group_and_average
from .intro_pandas import count_in_col
from .intro_pandas import export_data
from .intro_pandas import compare_dfs

# Operaciones con arrays y matrices
from .intro_numpy import ten_zeros_array
from .intro_numpy import floats_array
from .intro_numpy import invert_array
from .intro_numpy import square_matrix
from .intro_numpy import find_upper_five
from .intro_numpy import identity_matrix
from .intro_numpy import multiply_matrices
from .intro_numpy import normalize
from .intro_numpy import count_in_range
