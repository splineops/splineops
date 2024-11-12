
import numpy as np
from splineops.bases.bspline0basis import BSpline0Basis
from splineops.bases.bspline1basis import BSpline1Basis
from splineops.bases.bspline2basis import BSpline2Basis
from splineops.bases.bspline3basis import BSpline3Basis
from splineops.bases.bspline4basis import BSpline4Basis
from splineops.bases.bspline5basis import BSpline5Basis
from splineops.bases.bspline6basis import BSpline6Basis
from splineops.bases.bspline7basis import BSpline7Basis

# Original beta function implementation
def beta_original(x: float, degree: int) -> float:
    betan = 0.0
    if degree == 0:
        if abs(x) < 0.5 or x == -0.5:
            betan = 1.0
    elif degree == 1:
        x = abs(x)
        if x < 1.0:
            betan = 1.0 - x
    elif degree == 2:
        x = abs(x)
        if x < 0.5:
            betan = 3.0 / 4.0 - x * x
        elif x < 1.5:
            x -= 3.0 / 2.0
            betan = x * x * (1.0 / 2.0)
    elif degree == 3:
        x = abs(x)
        if x < 1.0:
            betan = x * x * (x - 2.0) * (1.0 / 2.0) + 2.0 / 3.0
        elif x < 2.0:
            x -= 2.0
            betan = x * x * x * (-1.0 / 6.0)
    elif degree == 4:
        x = abs(x)
        if x < 0.5:
            x *= x
            betan = x * (x * (1.0 / 4.0) - 5.0 / 8.0) + 115.0 / 192.0
        elif x < 1.5:
            betan = x * (x * (x * (5.0 / 6.0 - x * (1.0 / 6.0)) - 5.0 / 4.0) + 5.0 / 24.0) + 55.0 / 96.0
        elif x < 2.5:
            x -= 5.0 / 2.0
            x *= x
            betan = x * x * (1.0 / 24.0)
    elif degree == 5:
        x = abs(x)
        if x < 1.0:
            a = x * x
            betan = a * (a * (1.0 / 4.0 - x * (1.0 / 12.0)) - 1.0 / 2.0) + 11.0 / 20.0
        elif x < 2.0:
            betan = x * (x * (x * (x * (x * (1.0 / 24.0) - 3.0 / 8.0) + 5.0 / 4.0) - 7.0 / 4.0) + 5.0 / 8.0) + 17.0 / 40.0
        elif x < 3.0:
            a = 3.0 - x
            x = a * a
            betan = a * x * x * (1.0 / 120.0)
    elif degree == 6:
        x = abs(x)
        if x < 0.5:
            x *= x
            betan = x * (x * (7.0 / 48.0 - x * (1.0 / 36.0)) - 77.0 / 192.0) + 5887.0 / 11520.0
        elif x < 1.5:
            betan = x * (x * (x * (x * (x * (x * (1.0 / 48.0) - 7.0 / 48.0) + 21.0 / 64.0) - 35.0 / 288.0) - 91.0 / 256.0) - 7.0 / 768.0) + 7861.0 / 15360.0
        elif x < 2.5:
            betan = x * (x * (x * (x * (x * (7.0 / 60.0 - x * (1.0 / 120.0)) - 21.0 / 32.0) + 133.0 / 72.0) - 329.0 / 128.0) + 1267.0 / 960.0) + 1379.0 / 7680.0
        elif x < 3.5:
            x -= 7.0 / 2.0
            x *= x * x
            betan = x * x * (1.0 / 720.0)
    elif degree == 7:
        x = abs(x)
        if x < 1.0:
            a = x * x
            betan = a * (a * (a * (x * (1.0 / 144.0) - 1.0 / 36.0) + 1.0 / 9.0) - 1.0 / 3.0) + 151.0 / 315.0
        elif x < 2.0:
            betan = x * (x * (x * (x * (x * (x * (1.0 / 20.0 - x * (1.0 / 240.0)) - 7.0 / 30.0) + 1.0 / 2.0) - 7.0 / 18.0) - 1.0 / 10.0) - 7.0 / 90.0) + 103.0 / 210.0
        elif x < 3.0:
            betan = x * (x * (x * (x * (x * (x * (x * (1.0 / 720.0) - 1.0 / 36.0) + 7.0 / 30.0) - 19.0 / 18.0) + 49.0 / 18.0) - 23.0 / 6.0) + 217.0 / 90.0) - 139.0 / 630.0
        elif x < 4.0:
            a = 4.0 - x
            x = a * a * a
            betan = x * x * a * (1.0 / 5040.0)
    return betan

# New beta function using imported B-spline basis classes
def beta_new(x: float, degree: int) -> float:
    x_array = np.array([x])
    if degree == 0:
        return BSpline0Basis.eval(x_array)[0]
    elif degree == 1:
        return BSpline1Basis.eval(x_array)[0]
    elif degree == 2:
        return BSpline2Basis.eval(x_array)[0]
    elif degree == 3:
        return BSpline3Basis.eval(x_array)[0]
    elif degree == 4:
        return BSpline4Basis.eval(x_array)[0]
    elif degree == 5:
        return BSpline5Basis.eval(x_array)[0]
    elif degree == 6:
        return BSpline6Basis.eval(x_array)[0]
    elif degree == 7:
        return BSpline7Basis.eval(x_array)[0]
    else:
        raise ValueError("Degree not supported")

# Testing and comparing the functions
degrees = range(8)
test_points = np.linspace(-4, 4, 50)  # Test points to evaluate

for degree in degrees:
    for x in test_points:
        orig_val = beta_original(x, degree)
        new_val = beta_new(x, degree)
        diff = abs(orig_val - new_val)
        if diff > 1e-6:
            print(f"Difference found at degree {degree}, x={x:.2f}: Original={orig_val:.6f}, New={new_val:.6f}, Diff={diff:.6e}")
        else:
            print(f"Degree {degree}, x={x:.2f}: Original={orig_val:.6f}, New={new_val:.6f} (OK)")
