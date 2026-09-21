import numpy as np
from scipy.optimize import least_squares

class BaselineModel:
    def __init__(self):
        pass
    
    def poly2d(self, x, y, c):
        return c[0] + c[1]*x + c[2]*y + c[3]*x**2 + c[4]*y**2 + c[5]*x*y
        
    def _residual(self, c, x, y, z):
        return self.poly2d(x, y, c) - z
        
    def predict(self, image):
        """Fits a 2nd order polynomial to the image and returns the fit (reconstruction)."""
        H, W = image.shape
        x = np.arange(W)
        y = np.arange(H)
        X, Y = np.meshgrid(x, y)
        
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = image.flatten()
        
        # Initial guess (mean, then zeros)
        c0 = [np.mean(Z_flat), 0, 0, 0, 0, 0]
        
        # Fit 2nd order poly
        res = least_squares(self._residual, c0, args=(X_flat, Y_flat, Z_flat))
        
        reconstruction = self.poly2d(X, Y, res.x)
        return reconstruction
