# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: test_evaluation.py
# Descripción: Archivo de pruebas unitarias para validar el comportamiento de funciones del proyecto
# ============================================================

import sys, os, io, unittest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import numpy as np
import pandas as pd

import main
import introduction.intro_numpy as inp
import introduction.intro_pandas as ipd
import introduction.intro_scipy as isp
from regression_models.linear_regression import LinearRegressionCompare
from regression_models.multiple_linear_regression import MultipleLinearRegressionCompare
from regression_models.logistic_regression import LogisticRegressionCompare

# ============================================================
# Colores y constantes
# ============================================================

GREEN = "\033[92m"
RED = "\033[91m"
LIGHT_RED = "\033[31m"
RESET = "\033[0m"
BOLD = "\033[1m"
SEPARATOR = f"{BOLD}{'='*50}{RESET}"

SOURCE_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"
OUTPUT_DIR = "test_outputs"
FEATURE_1 = "ENGINESIZE"
FEATURE_2 = "FUELCONSUMPTION_COMB"
BASE = "CO2EMISSIONS"
CHURN = "churn"
HISTOGRAM = os.path.join(OUTPUT_DIR, "histogram.png")
CORRELATION = os.path.join(OUTPUT_DIR, "correlation.png")

# ============================================================
# Clases personalizadas para unittest
# ============================================================

class CustomTestResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successes = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successes.append(test)

class CustomTestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return CustomTestResult(self.stream, self.descriptions, self.verbosity)

# ============================================================
# Test Cases
# ============================================================

class TestEvaluationOne(unittest.TestCase):
    # ===================== intro_numpy =====================
    def test_ten_zeros_array(self):
        result = inp.ten_zeros_array(10)
        self.assertTrue(isinstance(result, np.ndarray))
        self.assertEqual(len(result), 10)
        self.assertTrue(np.all(result == 0.0))
        self.assertEqual(result.dtype, np.float64)

    def test_floats_array(self):
        result = inp.floats_array(1, 5)
        self.assertTrue(np.allclose(result, np.array([1., 2., 3., 4.])))

    def test_invert_array(self):
        arr = np.array([1, 2, 3])
        result = inp.invert_array(arr)
        self.assertTrue(np.array_equal(result, np.array([3, 2, 1])))

    def test_square_matrix(self):
        result = inp.square_matrix(2, 1, 5)
        self.assertEqual(result.shape, (2, 2))
        self.assertTrue(np.all(result >= 1) and np.all(result < 5))

    def test_find_upper_five(self):
        mat = np.array([[1, 6], [7, 3]])
        result = inp.find_upper_five(mat)
        self.assertTrue(np.array_equal(result, np.array([[0, 1], [1, 0]])))

    def test_identity_matrix(self):
        result = inp.identity_matrix(3)
        self.assertTrue(np.array_equal(result, np.identity(3)))

    def test_multiply_matrices(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[2, 0], [1, 2]])
        result = inp.multiply_matrices(a, b)
        self.assertTrue(np.array_equal(result, np.array([[4, 4], [10, 8]])))

    def test_normalize(self):
        arr = np.array([2, 4, 6])
        result = inp.normalize(arr)
        self.assertTrue(np.allclose(result, np.array([0., 0.5, 1.])))

    def test_count_in_range(self):
        arr = np.array([1, 5, 10, 15])
        result = inp.count_in_range(arr, 5, 12)
        self.assertEqual(result, 2)

    def test_get_statistics_numpy(self):
        arr = np.array([1, 2, 3])
        mean, median, std = inp.get_statistics(arr)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(median, 2.0)
        self.assertAlmostEqual(std, 0.816, places=2)

    # ===================== intro_pandas =====================
    def test_get_head(self):
        df = pd.DataFrame({'a': [1, 2, 3, 4]})
        result = ipd.get_head(df, 2)
        self.assertEqual(len(result), 2)

    def test_get_above(self):
        df = pd.DataFrame({'x': [1, 5, 10]})
        result = ipd.get_above(df, 'x', 4)
        self.assertTrue((result['x'] > 4).all())

    def test_group_and_average(self):
        df = pd.DataFrame({'g': ['A', 'A', 'B'], 'v': [1, 3, 5]})
        result = ipd.group_and_average(df, 'g', 'v')
        self.assertTrue(np.allclose(result['A'], 2.0))
        self.assertTrue(np.allclose(result['B'], 5.0))

    def test_count_in_col(self):
        df = pd.DataFrame({'c': ['a', 'b', 'a']})
        result = ipd.count_in_col(df, 'a', 'c')
        self.assertEqual(result, 2)

    def test_compare_dfs(self):
        df1 = pd.DataFrame({'x': [1, 2]})
        df2 = pd.DataFrame({'x': [1, 2]})
        self.assertTrue(ipd.compare_dfs(df1, df2))

    # ===================== intro_scipy =====================
    def test_solve_linear(self):
        A = np.array([[2, 1], [1, 3]])
        b = np.array([8, 13])
        result = isp.solve_linear(A, b)
        self.assertTrue(np.allclose(np.dot(A, result), b))

    def test_get_matrix_properties(self):
        mat = np.array([[1, 2], [3, 4]])
        det, inv = isp.get_matrix_properties(mat)
        self.assertAlmostEqual(det, -2.0)
        self.assertTrue(np.allclose(np.dot(mat, inv), np.identity(2)))

    def test_find_min(self):
        result = isp.find_min(lambda x: (x - 2)**2)
        self.assertAlmostEqual(result.x, 2.0, places=2)

    def test_get_statistics_scipy(self):
        arr = np.array([1, 2, 2, 3])
        mean, tstd, mode = isp.get_statistics(arr)
        self.assertAlmostEqual(mean, 2.0)
        self.assertAlmostEqual(tstd, np.std(arr, ddof=1))
        self.assertEqual(mode, 2.0)

    def test_low_pass_filter(self):
        signal_data = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 100))
        result = isp.low_pass_filter(signal_data, fs=100)
        self.assertEqual(len(result), len(signal_data))

    def test_main_execution(self):
        status = main.main()
        self.assertEqual(status, os.EX_OK)
        total, df = ipd.csv_registers(main.CSV_FILE)
        self.assertIsNotNone(total)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)

# ============================================================
# Ejercicio 2: Regresión lineal
# ============================================================

class TestEvaluationTwo(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        cls.model = LinearRegressionCompare(
            url=SOURCE_URL,
            hist=HISTOGRAM,
            base=BASE,
            f1=FEATURE_1,
            f2=FEATURE_2,
            out=OUTPUT_DIR
        )

    def test_attributes_exist(self):
        self.assertIsInstance(self.model.x1, np.ndarray)
        self.assertIsInstance(self.model.x2, np.ndarray)
        self.assertIsInstance(self.model.y, np.ndarray)
        self.assertIsNotNone(self.model.m1)
        self.assertIsNotNone(self.model.m2)
        self.assertIsInstance(self.model.p1, np.ndarray)
        self.assertIsInstance(self.model.p2, np.ndarray)

    def test_model_training(self):
        coef1 = self.model.m1.coef_.flatten()[0]
        coef2 = self.model.m2.coef_.flatten()[0]
        self.assertIsInstance(coef1, float)
        self.assertIsInstance(coef2, float)

    def test_output_files_created(self):
        files = [
            f"relationship_{FEATURE_1.lower()}_{BASE.lower()}.png",
            f"relationship_{FEATURE_2.lower()}_{BASE.lower()}.png",
            f"linear_regression_{FEATURE_1.lower()}_{BASE.lower()}.png",
            f"linear_regression_{FEATURE_2.lower()}_{BASE.lower()}.png",
            "histogram.png"
        ]
        for f in files:
            self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, f)))

# ============================================================
# Ejercicio 3: Regresión lineal múltiple
# ============================================================

class TestEvaluationThree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        cls.model = MultipleLinearRegressionCompare(
            url=SOURCE_URL,
            corr=CORRELATION,
            f1=FEATURE_1,
            f2=FEATURE_2,
            base=BASE,
            out=OUTPUT_DIR
        )

    def test_attributes_exist(self):
        self.assertIsInstance(self.model.x, np.ndarray)
        self.assertIsInstance(self.model.y, np.ndarray)
        self.assertIsNotNone(self.model.m)
        self.assertIsNotNone(self.model.d)
        self.assertIsNotNone(self.model.std_scaler) 
        self.assertIsNotNone(self.model.x_std)

    def test_model_training(self):
        coef = self.model.m.coef_[0]
        self.assertIsInstance(coef, np.ndarray)

    def test_output_files_created(self):
        files = [
            f"multiple_linear_regression_{FEATURE_1.lower()}{FEATURE_2.lower()}{BASE.lower()}.png",
            f"split_mlr_{FEATURE_1.lower()}_{BASE.lower()}.png",
            f"split_mlr_{FEATURE_2.lower()}_{BASE.lower()}.png",
            "correlation.png"
        ]
        for f in files:
            path = os.path.join(OUTPUT_DIR, f)
            self.assertTrue(os.path.exists(path), f"No se encontró el archivo: {path}")

# ============================================================
# Ejercicio 4: Regresión logística
# ============================================================

class TestEvaluationFour(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        cls.model = LogisticRegressionCompare(
            url="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/ChurnData.csv",
            base=CHURN,
            out=OUTPUT_DIR
        )

    def test_attributes_exist(self):
        self.assertIsInstance(self.model.x, np.ndarray)
        self.assertIsInstance(self.model.y, np.ndarray)
        self.assertIsNotNone(self.model.m)
        self.assertIsNotNone(self.model.d)
        self.assertIsNotNone(self.model.std_scaler) 
        self.assertIsNotNone(self.model.x_std)

    def test_model_training(self):
        coef = self.model.m.coef_[0]
        self.assertIsInstance(coef, np.ndarray)

    def test_output_files_created(self):
        files = [
            f"logistic_regression_churn_coefficients.png"
        ]
        for f in files:
            self.assertTrue(os.path.exists(os.path.join(OUTPUT_DIR, f)))

# ============================================================
# Ejecución principal
# ============================================================

if __name__ == '__main__':
    suites = [
        (1, TestEvaluationOne),
        (2, TestEvaluationTwo),
        (3, TestEvaluationThree),
        (4, TestEvaluationFour)
    ]
    for i, test_case in suites:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
        silent_stream = io.StringIO()
        runner = CustomTestRunner(stream=silent_stream, verbosity=0)
        result = runner.run(suite)

        print(f"{BOLD}EVALUACION {i}{RESET}")
        print(SEPARATOR)
        print(f"{BOLD}Resultados individuales:{RESET}")
        for t in result.successes:
            print(f"{t._testMethodName}: {GREEN}{BOLD}PASSED{RESET}")
        for t, tb in result.failures + result.errors:
            name = getattr(t, "_testMethodName", str(t))
            last_line = tb.strip().split('\n')[-1]
            mensaje = last_line.split(':')[-1].strip()
            print(f"{name}: {RED}{BOLD}FAILED{RESET}")
            print(f"- detalles: {LIGHT_RED}{mensaje}{RESET}")
        print(SEPARATOR)
        if result.wasSuccessful():
            print(f"{GREEN}{BOLD}SUCCESS:{RESET} Todos los tests pasaron correctamente.")
        else:
            print(f"{RED}{BOLD}FAILED:{RESET} Uno o más tests fallaron.")
        print(SEPARATOR)

