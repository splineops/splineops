/*
 * 1D convolution routines.
 *
 * Important note: 
 * Not all boundary conditions are supported for every convolver.
 *
 * @author 	Swiss Federal Institute of Technology Lausanne
 *			Biomedical Imaging Group
 *
 * @version 1.0
 *
 */
 
//package imageaccess.toolbox;
//import imageaccess.*;

package plugins.big.vascular.steerable3DFilter;

public class Convolver extends Object
{

static public final int MIRROR   = 4;
static public final int PERIODIC = 5;

private int boundaryConditions;
private double tolerance;

/**
* Constructor of the class Convolver.
*
* @param boundaryConditions    Boundary Conditions
*/
public Convolver(int boundaryConditions)
{
	this.boundaryConditions = boundaryConditions;
	tolerance = 1e-6;
} 

/**
* Set the boundary conditions.
*
* @param boundaryConditions    Boundary Conditions
*/
public void setBoundaryConditions(int boundaryConditions)
{
	this.boundaryConditions = boundaryConditions;
}

/**
* Get the boundary conditions.
*/
public int getBoundaryConditions()
{
	return boundaryConditions;
}

/**
* Set the tolerance.
*
* @param tolerance    Tolearance value
*/
public void setTolerance(int tolerance)
{
	this.tolerance = tolerance;
}

/**
* Get the tolerance.
*/
public double getTolerance()
{
	return tolerance;
}

/**
* Convolution with a Finite Impulse Response (FIR) filter.
*
* Note: Only with the periodic boundary conditions.
*
* @param signal   1D input signal, 1D output signal at the end (in-place)
* @param kernel   kernel of the filter
* @param origin   origin
*/
public void convolveFIR(double[] signal, double[] kernel, int origin) 
{
	int l = signal.length;
		
	if (l <= 1) 
		throw new IllegalArgumentException("convolveFIR: input signal too short");

	double[] output = new double[l];
	
	switch (boundaryConditions) {

	case PERIODIC:
		int k = l - (kernel.length - 1) + origin;
		int n;
		double sum = 0.0;
	
		k = (k < 0) ? (k + l * ((l - 1 - k) / l)) : ((l <= k) ? (k - l * (k / l)) : (k));
	
		for (int i = 0; i < l; i++) {
			sum = 0.0;
			n = k;
			for (int j = kernel.length - 1; 0 <= j; j--) {
				sum += signal[n] * kernel[j];
				if (++n == l) {
					n = 0;
				}
			}
			output[i] = sum;
			if (++k == l) {
				k = 0;
			}
		}
		break;

	case MIRROR:
		int indexq = kernel.length - 1;
		int indexp = 0;
		int n2 = 2 * (l - 1);
		int m = 1 + origin - kernel.length;
		m -= (m < 0L) ? (n2 * ((m + 1 - n2) / n2)) : (n2 * (m / n2));
		for (int i = 0; i < l; i++) {
			int j = -kernel.length;
			k = m;
			indexq = kernel.length - 1;
			double Sum = 0.0;
			while (j < 0) {
				indexp = k;
				int kp = ((k - l) < j) ? (j) : (k - l);
				if (kp < 0L) {
					for (n = kp; n < 0; n++) {
						Sum += signal[indexp] * kernel[indexq];
						indexq--;
						indexp++;
					}
					k -= kp;
					j -= kp;
				}
				indexp = n2 - k;
				int km = ((k - n2) < j) ? (j) : (k - n2);
				if (km < 0L) {
					for (n = km; n < 0; n++) {
						Sum += signal[indexp] * kernel[indexq];
						indexq--;
						indexp--;
					}
					j -= km;
				}
				k = 0;
			}
			if (++m == n2) {
				m = 0;
			}
			output[i] = Sum;
		}
		break;
	}

	System.arraycopy(output, 0, signal, 0, l);

}

/**
* Convolve with with a Infinite Impluse Response filter (IIR)
*
* @param signal   1D input signal, 1D output signal at the end (in-place)
* @param poles    1D array containing the poles of the filter
*/
public void convolveIIR(double[] signal, double poles[]) 
{
	double lambda = 1.0;
	for (int k = 0; (k < poles.length); k++) {
		lambda = lambda * (1.0 - poles[k]) * (1.0 - 1.0 / poles[k]);
	}
	for (int n = 0; (n < signal.length); n++) {
		signal[n] = signal[n] * lambda;
	}
	for (int k = 0; (k < poles.length); k++) {
		switch (boundaryConditions) {
			case MIRROR: // MIRROR
				signal[0] = getInitialCausalCoefficientMirror(signal, poles[k]);
				break;
			case PERIODIC: // PERIODIC
				signal[0] = getInitialCausalCoefficientPeriodic(signal, poles[k]);
				break;
			default:
				throw new IllegalArgumentException("Invalid boundary)");
		}
		for (int n = 1; (n < signal.length); n++) {
			signal[n] = signal[n] + poles[k] * signal[n - 1];
		}
		switch (boundaryConditions) {
			case MIRROR: // MirrorOnBounds
				signal[signal.length - 1] = getInitialAntiCausalCoefficientMirror(signal, poles[k]);
				break;
			case PERIODIC: // Periodic
				signal[signal.length - 1] = getInitialAntiCausalCoefficientPeriodic(signal, poles[k]);
				break;
			default:
				throw new IllegalArgumentException("Invalid boundary)");
		}
		for (int n = signal.length - 2; (0 <= n); n--) {
			signal[n] = poles[k] * (signal[n+1] - signal[n]);
		}
	}
}

/**
* Convolve a 1D signal with a Infinite Impluse Response 2nd order (IIR2)
*
* Note: Only with the mirror (on bounds) boundary conditions.
*
* Purpose:	Recursive implementation of a symmetric 2nd order
*			filter with mirror symmetry boundary conditions :
*
*					      1                1
*			H[z] = --------------- * ---------------
*				   (1-b1*z-b2*z^2)   (1-b1/z-b2/z^2)
*				
*			implemented in the following form:
*				
*					   a1+a2*z          a1+a2/z
*			H[z] = --------------- + ---------------  - a1
*				   (1-b1*z+b2*z^2)   (1-b1/z+b2/z^2) 
*				
*			where :
*			a1 = -(b2 + 1.0) * (1 - b1 + b2) / ((b2 - 1.0) * (1 + b1 + b2));
*			a2 = - a1 * b2 * b1 / (b2 + 1.0);
*
* @param signal   1D input signal, 1D output signal at the end (in-place)
* @param b1		  first pole of the filter
* @param b2   	  second pole of the filter
*/
public void convolveIIR2( double signal[], double b1, double b2)
{

	if (boundaryConditions != MIRROR)
		throw new IllegalArgumentException("Invalid boundary)");
		
	int l = signal.length;
	int n2 = 2 * l;

	double a1 = -(b2 + 1.0) * (1 - b1 + b2) / ((b2 - 1.0) * (1 + b1 + b2));
	double a2 = - a1 * b2 * b1 / (b2 + 1.0);
	
	// cBuffer stores temporary spline coefficients
	double cBuffer[] = new double[n2];
    
    // sBuffer contains a copy of s[] and a time reversed version of s[]
	double sBuffer[] = new double[n2];   
  	
  	// copy signal s[] and its time reversed version to sBuffer[]
    for( int n = 0; n < l; n++){
		sBuffer[n] = signal[n];
		sBuffer[n2 - n - 1] = signal[n];
	}
	
	// Determine the start index n0 for the causal recursion. n0 is chosen such
	// that the error of cBuffer[0] and cBuffer[1] is smaller than the
	// specified 'Tolerance'.
  	int n0 = 2;
   	if ((tolerance > 0.0) && (b2 != 1.0)) {
		n0 = n2 - (int) Math.ceil(2.0 * Math.log(tolerance) / Math.log(b2)); 
   	}

	if (n0 < 2) {     
		n0 = 2;
	}
	
	cBuffer[n0 - 1] = 0.0;
	cBuffer[n0 - 2] = 0.0;
	for (int n = n0; n < n2; n++) {
		cBuffer[n] = a1 * sBuffer[n] + a2 * sBuffer[n-1] + b1 * cBuffer[n-1] - b2 * cBuffer[n-2];
	}

	cBuffer[0] = a1 * sBuffer[0] + a2 * sBuffer[n2-1] + b1 * cBuffer[n2-1] - b2 * cBuffer[n2-2];
	cBuffer[1] = a1 * sBuffer[1] + a2 * sBuffer[0] + b1 * cBuffer[0] - b2 * cBuffer[n2-1];

   	// compute the remaining spline coefficients cBuffer(z) = H_{+}(z) * sBuffer(z) by 
    // recursive filtering
    for( int n = 2; n < n2; n++) {
		cBuffer[n] = a1 * sBuffer[n] + a2 * sBuffer[n-1] + b1 * cBuffer[n-1] - b2 * cBuffer[n-2];
	}
	
	// add together the temporary filter outputs to obtain the final  spline coefficients
    for( int n = 0; n < l; n++) {
		signal[n] = cBuffer[n]  + cBuffer[n2-n-1] - a1 * signal[n];
	}
}

/**
 */
private double getInitialAntiCausalCoefficientMirror(double[] c, double z) 
{
	return((z * c[c.length - 2] + c[c.length - 1]) * z / (z * z - 1.0));
}

/**
 */
private double getInitialAntiCausalCoefficientPeriodic (double[] c, double z) 
{
	double sum = c[0] + c[c.length - 1] / z, zn = z;
	int horizon = c.length;

	if (0.0 < tolerance) {
		horizon = 2 + (int)(Math.log(tolerance) / Math.log(Math.abs(z)));
		horizon = (horizon < c.length) ? (horizon) : (c.length);
	}
	for (int n = 1; (n < (horizon - 1)); n++) {
		sum = sum + zn * c[n];
		zn = zn * z;
	}
	return(z * z * sum / (Math.pow(z, c.length) - 1.0));
}

/**
 */
private double getInitialCausalCoefficientAntiMirror(double[] c, double z)
{
	double z1 = z, zn = Math.pow(z, c.length - 1);
	double sum = (c[0] - zn * c[c.length - 1]) * (1.0 + z) / (1.0 - z);
	int horizon = c.length;

	if (0.0 < tolerance) {
		horizon = 2 + (int)(Math.log(tolerance) / Math.log(Math.abs(z)));
		horizon = (horizon < c.length) ? (horizon) : (c.length);
	}
	zn = zn * zn;
	for (int n = 1; (n < (horizon - 1)); n++) {
		zn = zn / z;
		sum = sum + (zn - z1) * c[n];
		z1 = z1 * z;
	}
	return(sum / (1.0 - Math.pow(z, 2 * c.length - 2)));
}

/**
 */
private double getInitialCausalCoefficientFiniteCoefficientSupport (double[] c, double z) 
{
	double z1 = z, zn = Math.pow(z, 2 * c.length);
	double sum = 0.0;
	int horizon = c.length;

	if (0.0 < tolerance) {
		horizon = 2 + (int)(Math.log(tolerance) / Math.log(Math.abs(z)));
		horizon = (horizon < c.length) ? (horizon) : (c.length);
	}
	for (int n = 1; (n < horizon); n++) {
		zn = zn / z;
		sum = sum + (z1 - zn) * c[n];
		z1 = z1 * z;
	}
	z1 = z * z;
	return((1.0 - z1) * (c[0] - sum * z1 / (1.0 - z1))
		/ (1.0 - Math.pow(z, 2 * c.length + 2)));
}


/**
 */
private double getInitialCausalCoefficientMirror(double[] c, double z) 
{
	double z1 = z, zn = Math.pow(z, c.length - 1);
	double sum = c[0] + zn * c[c.length - 1];
	int horizon = c.length;

	if (0.0 < tolerance) {
		horizon = 2 + (int)(Math.log(tolerance) / Math.log(Math.abs(z)));
		horizon = (horizon < c.length) ? (horizon) : (c.length);
	}
	zn = zn * zn;
	for (int n = 1; (n < (horizon - 1)); n++) {
		zn = zn / z;
		sum = sum + (z1 + zn) * c[n];
		z1 = z1 * z;
	}
	return(sum / (1.0 - Math.pow(z, 2 * c.length - 2)));
}

/**
 */
private double getInitialCausalCoefficientPeriodic(double c[], double z) 
{
	double sum = c[0], zn = z;
	int horizon = c.length;

	if (0.0 < tolerance) {
		horizon = 1 + (int)(Math.log(tolerance) / Math.log(Math.abs(z)));
		horizon = (horizon < c.length) ? (horizon) : (c.length);
	}
	for (int n = 1; (n < horizon); n++) {
		sum = sum + zn * c[c.length - n];
		zn = zn * z;
	}
	return(sum / (1.0 - Math.pow(z, c.length)));
}

} // end of classe