/*
 * Resize plugin for ImageJ,
 * Image processing part of the plugin.
 *
 * @author 	Arrate Munoz
 *			Swiss Federal Institute of Technology Lausanne
 *         	Biomedical Imaging Group
 *			BM-Ecublens
 * 			CH-1015 Lausanne EPFL, Switzerland
 *			URL: http:/bigwww.epfl.ch/
 *			email: arrate.munoz@epfl.ch
 *
 * @version July 11, 2001
*/		

public class Resize
{
private int interpDegree;
private int analyDegree;
private int syntheDegree;
private double zoomY;
private double zoomX;
private boolean inversable;

private int analyEven = 0;
private int corrDegree;
private double halfSupport;
private double[] splineArrayHeight;
private double[] splineArrayWidth;
private int[] indexMinHeight;
private int[] indexMaxHeight;
private int[] indexMinWidth;
private int[] indexMaxWidth;

private final double tolerance = 1e-9; // tolerance for the Bspline transform

/**
* Calculate the resize version of the input image.
* 
* @param input 			an ImageAccess object to be resized
* @param output 		an ImageAccess object which is the resized version of the input
* @param interpDegree 	degree of the interpolating spline
* @param analyDegree 	degree of the analysis spline
* @param synthedegree 	degree of the synthesis spline
* @param zoomY 			scaling factor that applies to the columns
* @param zoomX 			scaling factor that applies to the rows
* @param shiftY 		shift value that applies to the columns
* @param shiftX 		shift value that applies to the rows
* @param inversable 	boolean that indicates if calculate inversable image or not
*		 
*/   
public void computeZoom(
		ImageAccess input,ImageAccess output, 
		int analyDegree, int syntheDegree, int interpDegree, 
		double zoomY, double zoomX, 
		double shiftY, double shiftX,
		boolean inversable) 
{
	this.interpDegree = interpDegree;
	this.analyDegree = analyDegree;
	this.syntheDegree = syntheDegree;
	this.zoomY = zoomY;
	this.zoomX = zoomX;
	this.inversable = inversable;
	
	int nx = input.getWidth();                                          
	int ny = input.getHeight();                                             
	int workingSizeX;                                        
	int workingSizeY;                                       
	int finalSizeY;                                         
	int finalSizeX;                                          
	int[] size = new int[4];
	int max;
	int addBorderHeight;
	int addBorderWidth;
	
	int totalDegree = (interpDegree+analyDegree+1);
	                           
	size = calculatefinalsize(inversable, ny, nx, zoomY, zoomX);	
	
 	workingSizeX = size[1];
	workingSizeY = size[0];
	finalSizeX = size[3];
	finalSizeY = size[2];
	
	if ( ((analyDegree+1) / 2) * 2 == analyDegree+1)
		analyEven = 1;
		
	double cociente = (double)(analyDegree+1) / 2.0;
	double go = (double)(analyDegree+1);
	corrDegree = analyDegree+syntheDegree+1;
	halfSupport = (totalDegree+1.0) / 2.0;
	
	addBorderHeight = border(finalSizeY, corrDegree); // 1d spline values for row
	if (addBorderHeight < totalDegree){
		addBorderHeight += totalDegree;
	}
	
	int finalTotalHeight = finalSizeY + addBorderHeight;
	int lengthTotalHeight = workingSizeY + (int)Math.ceil(addBorderHeight/zoomY);
	
	indexMinHeight = new int[finalTotalHeight];
	indexMaxHeight = new int[finalTotalHeight];
	
	int lengthArraySplnHeight = finalTotalHeight * (2+totalDegree);
	int i = 0; 
	double affineIndex;
	double factHeight = Math.pow(zoomY, analyDegree+1);
	
	shiftY += ((analyDegree+1.0) / 2.0 - Math.floor((analyDegree+1.0)/2.0))*(1.0/zoomY-1.0);
	splineArrayHeight = new double[lengthArraySplnHeight];
	
	for (int l=0; l<finalTotalHeight; l++) {
	    affineIndex = (double)(l) / zoomY + shiftY;
	    indexMinHeight[l] = (int)Math.ceil(affineIndex-halfSupport);
	    indexMaxHeight[l] = (int)Math.floor(affineIndex+halfSupport);
	    for (int k=indexMinHeight[l]; k<=indexMaxHeight[l]; k++) {
			splineArrayHeight[i] = factHeight * beta(affineIndex-(double)k, totalDegree);
		i++; 
	    }
	}
	
	addBorderWidth = border(finalSizeX, corrDegree);
	if (addBorderWidth < totalDegree) {
		addBorderWidth += totalDegree;
	}
	
	int finalTotalWidth = finalSizeX+addBorderWidth;
	int lengthTotalWidth = workingSizeX+(int)Math.ceil(addBorderWidth/zoomX);
	
	indexMinWidth = new int[finalTotalWidth];
	indexMaxWidth = new int[finalTotalWidth];
	
	int lengthArraySplnWidth = finalTotalWidth * (2+totalDegree);
	i = 0; 
	double factWidth = Math.pow(zoomX, analyDegree+1);
	
	
	shiftX += ((analyDegree+1.0) / 2.0 - Math.floor((analyDegree+1.0)/2.0))*(1.0/zoomX-1.0);
	splineArrayWidth = new double[lengthArraySplnWidth];
	
	for (int l=0; l<finalTotalWidth; l++) {
	    affineIndex = (double)(l)/zoomX + shiftX;
	    indexMinWidth[l] = (int)Math.ceil(affineIndex-halfSupport);
	    indexMaxWidth[l] = (int)Math.floor(affineIndex+halfSupport);
	    for (int k=indexMinWidth[l]; k<=indexMaxWidth[l]; k++) {
			splineArrayWidth[i] = factWidth * beta(affineIndex-(double)k,totalDegree);
		i++; 
	    }
	}
	double[] outputColumn = new double[finalSizeY];
	double[] outputRow = new double[finalSizeX];                     
	double[] workingRow = new double[workingSizeX];                  
	double[] workingColumn = new double[workingSizeY]; 
	
	double[] addVectorHeight = new double[lengthTotalHeight];
	double[] addOutputVectorHeight = new double[finalTotalHeight];
	double[] addVectorWidth = new double[lengthTotalWidth];
	double[] addOutputVectorWidth = new double[finalTotalWidth]; 
	 
	int periodColumnSym = 2 * workingSizeY - 2;
	int periodRowSym = 2 * workingSizeX-2;
	int periodColumnAsym = 2 * workingSizeY - 3;
	int periodRowAsym = 2 * workingSizeX - 3;
 
	ImageAccess image = new ImageAccess (finalSizeX, workingSizeY, ImageAccess.FLOAT);         

	
    if (inversable == true) { 
	
		ImageAccess inverImage = new ImageAccess(workingSizeX, workingSizeY, ImageAccess.FLOAT);     
	
		for (int x=0; x<nx; x++) {
	    	for (int y=0; y<ny; y++) {	
				inverImage.putPixel(x, y, input.getPixel(x, y));
	    	}
		}
		
		if (workingSizeX > nx) { 
	    	inverImage.getColumn(nx-1, workingColumn);
	    	for (int y=nx; y<workingSizeX; y++) {
				inverImage.putColumn(y, workingColumn);    
	    	}
		}
	
		if (workingSizeY > ny) { 
	    	inverImage.getRow(ny-1, workingRow);
	    	for (int y=ny; y<workingSizeY; y++) {
				inverImage.putRow(y, workingRow);    
	    	}
		}
   
		// Row processing
		for (int y=0; y<workingSizeY; y++) {
	   		inverImage.getRow(y,workingRow);
	     	getInterpolationCoefficients(workingRow, interpDegree);
	     	resamplingRow(workingRow, outputRow, addVectorWidth, 
	     			addOutputVectorWidth, periodRowSym, periodRowAsym);
	     	image.putRow(y,outputRow);
		}

		// Column processing
		for (int y=0; y<finalSizeX; y++) {
			image.getColumn(y,workingColumn);
			getInterpolationCoefficients(workingColumn, interpDegree);
			resamplingColumn(workingColumn, outputColumn, 
					addVectorHeight, addOutputVectorHeight, periodColumnSym, periodColumnAsym);
			output.putColumn(y,outputColumn);
	 	}
	 
	}
	else { // inversable == 0
	 	
		// Row processing
		for (int y=0; y<workingSizeY; y++) {
	    	input.getRow(y, workingRow);
	     	getInterpolationCoefficients(workingRow, interpDegree);
	     	resamplingRow(workingRow, outputRow,addVectorWidth, 
	     			addOutputVectorWidth, periodRowSym, periodRowAsym);
	     	image.putRow(y,outputRow);
		}

		// Column processing
		for (int y=0; y<finalSizeX; y++) {
			image.getColumn(y, workingColumn);
		 	getInterpolationCoefficients(workingColumn, interpDegree);
		 	resamplingColumn(workingColumn, outputColumn, addVectorHeight, 
		 			addOutputVectorHeight, periodColumnSym, periodColumnAsym);
		 	output.putColumn(y,outputColumn);
	 	}
	}
} 
 
/**
* Calculate the affine transformation and the projection (if necessary) in the rows.
* 
* @param inputVector 		the input array 
* @param outputVector 		the output array
* @param addVector 			extended version of the input array
* @param addOutputVector 	extended version of the output array
* @param maxSymBoundary 	period for symmetric border conditions input
* @param maxAsymBoundary 	(period-1) for asymmetric border conditions input
*/   
private void resamplingRow(
		double[] inputVector, double[] outputVector, 
		double[] addVector, double[] addOutputVector,
		int maxSymBoundary, int maxAsymBoundary)
{

	double t;
	int lengthInput = inputVector.length;
	int lengthOutput = outputVector.length;
	int lengthtotal = addVector.length;
	int lengthOutputtotal = addOutputVector.length;
	double sign;
	double average = 0;
	int i;
	int k3;
	int l2;	
	int index;
	
	// Projection Method
	if ( analyDegree != -1) {   
		average = doInteg(inputVector, analyDegree+1);   
	}	
	
	System.arraycopy(inputVector, 0, addVector, 0, lengthInput);		
	
	for (int l=lengthInput; l<lengthtotal; l++){
		if(analyEven==1) {
			l2 = l;
			if (l >= maxSymBoundary)
				l2 = (int)Math.abs(Math.IEEEremainder((double)l,(double)maxSymBoundary));
			if (l2 >= lengthInput)
				l2 = maxSymBoundary - l2;
			addVector[l] = inputVector[l2];
		}
		else {
			l2 = l;
			if (l >= maxAsymBoundary)
				l2 = (int)Math.abs(Math.IEEEremainder((double)l,(double)maxAsymBoundary));
			if (l2 >= lengthInput)
				l2 = maxAsymBoundary - l2;
			addVector[l] =- inputVector[l2];
		}
	}
	     	
	i = 0;
	
	for (int l=0; l<lengthOutputtotal; l++) {   
		addOutputVector[l] = 0.0;
		for (int k=indexMinWidth[l]; k<=indexMaxWidth[l] ; k++) {
			index = k;
			sign = 1;
			if (k < 0) {
				index =- k;
				if (analyEven==0){
					index -= 1;
					 sign =- 1;
				}
			}		
			if (k >= lengthtotal) 
				index = lengthtotal-1;
			// Geometric transformation and resampling
			addOutputVector[l] += sign*addVector[index] * splineArrayWidth[i];
			i++;
		}
	}
	
	// Projection Method
	if ( analyDegree != -1) {
   		// Differentiation analyDegree+1 times of the signal
   		doDiff(addOutputVector,analyDegree+1);
		for (i=0;i<lengthOutputtotal;i++) 
			addOutputVector[i]+= average;
			// IIR filtering
			getInterpolationCoefficients(addOutputVector, corrDegree);
			// Samples
			getSamples(addOutputVector, syntheDegree);
	}
	
	System.arraycopy(addOutputVector, 0, outputVector, 0, lengthOutput); 
	
}

/**
* Calculate the affine transformation and the projection (if necessary) for the columns.
* 
* @param inputVector 		the input array 
* @param outputVector 		the output array
* @param addVector 			extended version of the input array
* @param addOutputVector 	extended version of the output array
* @param maxSymBoundary 	period for symmetric border conditions input
* @param maxAsymBoundary 	(period-1) for asymmetric border conditions input
*/   

private void resamplingColumn(
		double[] inputVector, double[] outputVector, 
		double[] addVector, double[] addOutputVector,
		int maxSymBoundary, int maxAsymBoundary) 
{

	double t;
	int lengthInput = inputVector.length;
	int lengthOutput = outputVector.length;
	int lengthtotal = addVector.length;
	int lengthOutputtotal = addOutputVector.length;
	double sign;
	double average = 0;
	int i;
	int k3;
	int l2;	
	int index;
	
	// Projection Method
	if ( analyDegree != -1) {   
		average = doInteg(inputVector, analyDegree+1);   
	}	
	
	System.arraycopy(inputVector, 0, addVector, 0, lengthInput);		
		
	for (int l=lengthInput; l<lengthtotal; l++) {
		if(analyEven == 1) {
			l2 = l;
			if (l >= maxSymBoundary)
				l2 = (int)Math.abs(Math.IEEEremainder((double)l,(double)maxSymBoundary));
			if (l2 >= lengthInput)
				l2 = maxSymBoundary - l2;
			addVector[l] = inputVector[l2];
		}
		else {
			l2 = l;
			if (l >= maxAsymBoundary)
				l2 = (int)Math.abs(Math.IEEEremainder((double)l,(double)maxAsymBoundary));
			if (l2 >= lengthInput)
				l2 = maxAsymBoundary - l2;
			addVector[l]=-inputVector[l2];
		}
	}
	     	
	i = 0;
	
	for (int l=0; l<lengthOutputtotal; l++) {   
		addOutputVector[l] = 0.0;
		for (int k=indexMinHeight[l]; k<=indexMaxHeight[l] ; k++) {
			index = k;
			sign = 1;
			if (k < 0){
				index=-k;
				if (analyEven==0){
					index-=1;
					 sign=-1;
				}
			}		
			if (k >= lengthtotal) 
				index = lengthtotal-1;
 			// Geometric transformation and resampling				
			addOutputVector[l] += sign*addVector[index] * splineArrayHeight[i];
			i++;
			}
	}
	
	if ( analyDegree != -1) {   
		// Projection Method
   		// Differentiation analyDegree+1 times of the signal		
		doDiff(addOutputVector,analyDegree+1);
		for (i=0;i<lengthOutputtotal;i++) 
			addOutputVector[i]+= average;
 			// IIR filtering			
			getInterpolationCoefficients(addOutputVector, corrDegree);
	 		// Samples
			getSamples(addOutputVector, syntheDegree);
	}
	
	System.arraycopy(addOutputVector, 0, outputVector, 0, lengthOutput); 
}


/**
* Calculate the value of the interpDegree of degree s at point x.
* 
* @param x 			position of interpDegree evaluation
* @param degree 	degree of the interpDegree
*/   
private double beta( double x, int degree)
{
	double betan = 0.0;
	double a;
	
	switch (degree) {
		case 0:	 
	      	if (Math.abs(x) < 0.5) { 
	      		betan = 1.0;
	      	}
	      	else {
	      		if (x == -0.5) { 
	      			betan = 1.0;
	      		} 
	      	}
	      	break;
		case 1:
			x = Math.abs(x);
	   	 	if (x < 1.0) { 
	   	 		betan = (1.0-x);
	   	 	}
	    	break;
		case 2:
			x = Math.abs(x);
			if (x < 0.5) {
				betan = 3.0/4.0-x*x;
			}
			else {
				if (x < 1.5) {
					x-=3.0/2.0;
					betan = x*x*(1.0/2.0);
				}
			}
			break;
		case 3:
			x = Math.abs(x);
	    	if (x < 1.0) {
	    	 	betan = x*x*(x-2.0)*(1.0/2.0)+2.0/3.0;
	    	}
	    	else if (x<2.0) {
	    		x-=2.0;
	    		betan = x*x*x*(-1.0/6.0);
	    	}
	    	break;
		case 4:
			x = Math.abs(x);
			if (x < 0.5) {
                x *= x;
                betan = (x * (x * (1.0 / 4.0) - 5.0 / 8.0) + 115.0 / 192.0);
        	}
       	 	else if (x < 1.5) {
                betan = (x * (x * (x * (5.0 / 6.0 - x * (1.0 / 6.0))
                	- 5.0 / 4.0) + 5.0 / 24.0) + 55.0 / 96.0);
        	}
        	else if (x < 2.5) {
                x -= 5.0 / 2.0;
                x *= x;
                betan = x * x * (1.0 / 24.0);
        	}
			break;
		case 5:
			x = Math.abs(x);
	 		if (x < 1.0) {
				a = x * x;
				betan = (a * (a * (1.0 / 4.0 - x * (1.0 / 12.0)) - 1.0 / 2.0)
                        + 11.0 / 20.0);
        	}
       		else if (x < 2.0) {
				betan = (x * (x * (x * (x * (x
					* (1.0 / 24.0) - 3.0 / 8.0) + 5.0 / 4.0) - 7.0 / 4.0) + 5.0 / 8.0) + 17.0 / 40.0);
        	}
        	else if (x < 3.0) {
				a = 3.0 - x;
				x = a * a;
				betan = (a * x * x * (1.0 / 120.0));
        	}
			break;
		case 6:
			x = Math.abs(x);
			if (x < 0.5) {
				x *= x;
				betan = (x * (x * (7.0 / 48.0 - x
					* (1.0 / 36.0)) - 77.0 / 192.0) + 5887.0 / 11520.0);
        	}
        	else if (x < 1.5) {
				betan = (x * (x * (x * (x * (x * (x
					* (1.0 / 48.0) - 7.0 / 48.0) + 21.0 / 64.0) - 35.0 / 288.0) - 91.0 / 256.0)
					- 7.0 / 768.0) + 7861.0 / 15360.0);
        	}
       		else if (x < 2.5) {
				betan = (x * (x * (x * (x * (x
                	* (7.0 / 60.0 - x * (1.0 / 120.0)) - 21.0 / 32.0) + 133.0 / 72.0)
                	- 329.0 / 128.0) + 1267.0 / 960.0) + 1379.0 / 7680.0);
        	}
        	else if (x < 3.5) {
				x -= 7.0 / 2.0;
				x *= x * x;
				betan = (x * x * (1.0 / 720.0));
        	}
  			break;
		case 7:
			x = Math.abs(x);
			if (x < 1.0) {
				a = x * x;
				betan = (a * (a * (a * (x * (1.0 / 144.0) - 1.0 / 36.0) + 1.0 / 9.0)
					- 1.0 / 3.0) + 151.0 / 315.0);
        	}
        	else if (x < 2.0) {
				betan = (x * (x * (x * (x * (x * (x
                	* (1.0 / 20.0 - x * (1.0 / 240.0)) - 7.0 / 30.0) + 1.0 / 2.0)
                	- 7.0 / 18.0) - 1.0 / 10.0) - 7.0 / 90.0) + 103.0 / 210.0);
       		}
        	else if (x < 3.0) {
                betan = (x * (x * (x * (x * (x * (x
                        * (x * (1.0 / 720.0) - 1.0 / 36.0) + 7.0 / 30.0) - 19.0 / 18.0)
                        + 49.0 / 18.0) - 23.0 / 6.0) + 217.0 / 90.0) - 139.0 / 630.0);
        	}
        	else if (x < 4.0) {
                a = 4.0 - x;
                x = a * a * a;
               betan = (x * x * a * (1.0 / 5040.0));
        	}
			break;
	}
    
	return betan;
}

/**
* Implements the whole integration procedure.
* 
* @param c 		an input array of double          
* @param nb 	number of integrations 
*/   
private double doInteg(double[] c, int nb) 
{
	int size = c.length;
	double m = 0.0, average = 0.0;
	
	switch (nb) {
		case 1: 
			for (int f=0;f<size;f++)
	   	 		average += c[f];
			average = (2.0*average - c[size-1] - c[0]) / (double)(2*size-2);
			integSA(c, average);
			break;
		case 2:
			for (int f=0;f<size;f++)
	   	 		average += c[f];
			average = (2.0*average - c[size-1] - c[0]) / (double)(2*size-2);
			integSA(c, average);
			integAS(c, c);
			break;
		case 3:
			for (int f=0;f<size;f++)
	   	 		average += c[f];
			average = (2.0*average - c[size-1] - c[0]) / (double)(2*size-2);
			integSA(c, average);
			integAS(c, c);
			for (int f=0;f<size;f++)
	    		m += c[f];
			m = (2.0*m - c[size-1] - c[0]) / (double)(2*size-2);
	    	integSA(c, m);
			break;
		case 4:
			for (int f=0;f<size;f++)
	   	 		average += c[f];
			average = (2.0*average - c[size-1] - c[0]) / (double)(2*size-2);
			integSA(c, average);
			integAS(c, c);
			for (int f=0;f<size;f++)
	    		m += c[f];
			m = (2.0*m - c[size-1] - c[0]) / (double)(2*size-2);
	    	integSA(c, m);
	    	integAS(c, c);
			break;	
	}
	return average;
}

/**
* Implements discrete integral filter for a
* symmetric input boundary conditions.
* 
* @param c 		an input array of double          
* @param m 		mean value over the period (2*size-2) of the input 
*/   
private void integSA(double[] c, double m) 
{
	int size = c.length;
	c[0] = (c[0]-m) * 0.5;
	for (int i=1;i<size;i++)
	    c[i] = c[i]-m+c[i-1];
 }  
  

/**
* Implements discrete integral filter for a
* asymmetric input boundary conditions.
* 
* @param c 		an input array of double          
* @param y 		an ouput array of double 
*/   
private void integAS(double[] c, double[] y) 
{
	int size = c.length;
	double [] z = new double[size];
	System.arraycopy(c, 0, z, 0, size);
	y[0] = z[0];
	y[1] = 0;
	for (int i=2;i<size;i++)
	    y[i] = y[i-1] - z[i-1];
}    

/**
* Implements the whole differentiation procedure.
* 
* @param c 			an input array of double          
* @param nb 		number of integrations 
* @param size 		the size of the input and output arrays	
*/   
private void doDiff(double[] c, int nb) 
{
	int size = c.length;
	switch (nb) {
		case 1: 
			diffAS(c);
			break;
		case 2:
			diffSA(c);
			diffAS(c);
			break;
		case 3:
			diffAS(c);
			diffSA(c);
			diffAS(c);
			break;
		case 4:
			diffSA(c);
			diffAS(c);
			diffSA(c);
			diffAS(c);
			break;	
	}	
}
  
/**
* Implements finite differences filter for a
* symmetric input boundary conditions.
* 
* @param c 			an input array of double          
* @param size 		the size of the input and output arrays	
*/
private void diffSA(double[] c) 
{
	int size = c.length;
	double old = c[size-2];
	for (int i=0; i<=size-2; i++)
	    c[i] = c[i] - c[i+1];
	c[size-1] = c[size-1] - old;
}  

/**
* Implements finite differences filter for an
* asymmetric input boundary conditions.
* 
* @param c 			an input array of double          
* @param size 		the size of the input and output arrays	
*/ 
private void diffAS(double[] c) 
{
	int size = c.length;
	for (int i=size-1; i>0; i--)
	    c[i] = c[i] - c[i-1];
	c[0] = 2.0 * c[0];
}

/**
* Calculate the number of additional samples to add
* in order not to have problems with the borders
* when applying the iir filter. 
* 
* @param size 		the size of the array to padd        
* @param degree 	the degree corresponding to the iir filter	
*/ 
private int border(int size, int degree)
{
	double z;
	int horizon = size;
	
	switch (degree) {
		case 0:
		case 1:
			return 0;
		case 2:
			z = Math.sqrt(8.0) - 3.0;
			break;
		case 3:
			z = Math.sqrt(3.0) - 2.0;
			break;
		case 4:
			z = Math.sqrt(664.0 - Math.sqrt(438976.0)) + Math.sqrt(304.0) - 19.0;
			break;
		case 5:
			z = Math.sqrt(135.0 / 2.0 - Math.sqrt(17745.0 / 4.0))
				+ Math.sqrt(105.0 / 4.0) - 13.0 / 2.0;
			break;
		case 6:
			z = -0.488294589303044755130118038883789062112279161239377608394;
			break;
		case 7:
			z = -0.5352804307964381655424037816816460718339231523426924148812;
			break;
		default:
			throw new IllegalArgumentException("Invalid interpDegree degree (should be [0..7])");
	}

	horizon = 2 + (int)(Math.log(tolerance) / Math.log(Math.abs(z)));
	horizon = (horizon < size) ? (horizon) : (size);
	return horizon;
	
}
 
/**
* Calculate the reversable (if necessary)
* and the final size.
* 
* @param inversable 		boolean 
* @param height 			number of rows 	    
* @param width 				number of columns
* @param zoomY 				scaling factor for the columns
* @param zoomX 				scaling factor for the rows
*/ 
static public int[] calculatefinalsize(boolean inversable, int height, int width, double zoomY, double zoomX)
{

	int[] size = new int[4];

	int w2;
	int h2;
	
	size[0]=height;
	size[1]=width;

	if (inversable == true) {
		w2 = (int)Math.round(Math.round((size[0]-1)*zoomY)/zoomY);
		while (size[0]-1-w2!=0) {
	    	size[0] = size[0]+1;
	    	w2 = (int)Math.round(Math.round((size[0]-1)*zoomY)/zoomY);
		}
	
		h2 = (int)Math.round(Math.round((size[1]-1)*zoomX)/zoomX);
		while (size[1]-1-h2!=0) {
	    	size[1] = size[1]+1;
	    	h2 = (int)Math.round(Math.round((size[1]-1)*zoomX)/zoomX);
		}
		size[2] = (int)Math.round((size[0]-1)*zoomY)+1; 
 		size[3] = (int)Math.round((size[1]-1)*zoomX)+1;
 	}
 	else {
 		size[2] = (int)Math.round(size[0]*zoomY); 
 		size[3] = (int)Math.round(size[1]*zoomX);
 	}
 	return size;
}  

/**
*/
private void getInterpolationCoefficients(double[] c, int degree) 
{
	double z[] = new double[0];
	double lambda = 1.0;

	switch (degree) {
		case 0:
		case 1:
			return;
		case 2:
			z = new double[1];
			z[0] = Math.sqrt(8.0) - 3.0;
			break;
		case 3:
			z = new double[1];
			z[0] = Math.sqrt(3.0) - 2.0;
			break;
		case 4:
			z = new double[2];
			z[0] = Math.sqrt(664.0 - Math.sqrt(438976.0)) + Math.sqrt(304.0) - 19.0;
			z[1] = Math.sqrt(664.0 + Math.sqrt(438976.0)) - Math.sqrt(304.0) - 19.0;
			break;
		case 5:
			z = new double[2];
			z[0] = Math.sqrt(135.0 / 2.0 - Math.sqrt(17745.0 / 4.0))
				+ Math.sqrt(105.0 / 4.0) - 13.0 / 2.0;
			z[1] = Math.sqrt(135.0 / 2.0 + Math.sqrt(17745.0 / 4.0))
				- Math.sqrt(105.0 / 4.0) - 13.0 / 2.0;
			break;
		case 6:
			z = new double[3];
			z[0] = -0.488294589303044755130118038883789062112279161239377608394;
			z[1] = -0.081679271076237512597937765737059080653379610398148178525368;
			z[2] = -0.00141415180832581775108724397655859252786416905534669851652709;
			break;
		case 7:
			z = new double[3];
			z[0] = -0.5352804307964381655424037816816460718339231523426924148812;
			z[1] = -0.122554615192326690515272264359357343605486549427295558490763;
			z[2] = -0.0091486948096082769285930216516478534156925639545994482648003;
			break;
		default:
			throw new IllegalArgumentException("Invalid spline degree (should be [0..7])");
	}
	
	if (c.length == 1) {
		return;
	}
	
	for (int k = 0; (k < z.length); k++) {
		lambda = lambda * (1.0 - z[k]) * (1.0 - 1.0 / z[k]);
	}
	
	for (int n = 0; (n < c.length); n++) {
		c[n] = c[n] * lambda;
	}
	
	for (int k = 0; (k < z.length); k++) {
		c[0] = getInitialCausalCoefficient(c, z[k], tolerance);
		for (int n = 1; (n < c.length); n++) {
			c[n] = c[n] + z[k] * c[n - 1];
		}
		c[c.length - 1] = getInitialAntiCausalCoefficient(c, z[k], tolerance);
		for (int n = c.length - 2; (0 <= n); n--) {
			c[n] = z[k] * (c[n+1] - c[n]);
		}
	}
}

/**
*/
private void getSamples (double[] c, int degree) 
{
	double h[] = new double[0];
	double s[] = new double[c.length];

	switch (degree) {
		case 0:
		case 1:
			return;
		case 2:
			h = new double[2];
			h[0] = 3.0 / 4.0;
			h[1] = 1.0 / 8.0;
			break;
		case 3:
			h = new double[2];
			h[0] = 2.0 / 3.0;
			h[1] = 1.0 / 6.0;
			break;
		case 4:
			h = new double[3];
			h[0] = 115.0 / 192.0;
			h[1] = 19.0 / 96.0;
			h[2] = 1.0 / 384.0;
			break;
		case 5:
			h = new double[3];
			h[0] = 11.0 / 20.0;
			h[1] = 13.0 / 60.0;
			h[2] = 1.0 / 120.0;
			break;
		case 6:
			h = new double[4];
			h[0] = 5887.0 / 11520.0;
			h[1] = 10543.0 / 46080.0;
			h[2] = 361.0 / 23040.0;
			h[3] = 1.0 / 46080.0;
			break;
		case 7:
			h = new double[4];
			h[0] = 151.0 / 315.0;
			h[1] = 397.0 / 1680.0;
			h[2] = 1.0 / 42.0;
			h[3] = 1.0 / 5040.0;
			break;
		default:
			throw new IllegalArgumentException("Invalid spline degree (should be [0..7])");
	}
	
	symmetricFir(h, c, s);
	System.arraycopy(s, 0, c, 0, s.length);
}


/**
* Note: Mirror On Bounds
*/
private double getInitialAntiCausalCoefficient(double[] c,double z,double tolerance) 
{
	return((z * c[c.length - 2] + c[c.length - 1]) * z / (z * z - 1.0));
}

/**
* Note: Mirror On Bounds
*/
private double getInitialCausalCoefficient (double[] c, double z, double tolerance) 
{
	double z1 = z, zn = Math.pow(z, c.length - 1);
	double sum = c[0] + zn * c[c.length - 1];
	int horizon = c.length;

	if (tolerance > 0.0) {
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
* Note: Mirror On Bounds
*/
private void symmetricFir(double[] h, double[] c, double[] s)
{
	if (c.length != s.length) {
		throw new IndexOutOfBoundsException("Incompatible size");
	}
	switch (h.length) {
		case 2:
			if (2 <= c.length) {
				s[0] = h[0] * c[0] + 2.0 * h[1] * c[1];
				for (int i = 1; (i < (c.length - 1)); i++) {
					s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1]);
				}
				s[s.length - 1] = h[0] * c[c.length - 1]
					+ 2.0 * h[1] * c[c.length - 2];
			}
			else {
				switch (c.length) {
					case 1:
						s[0] = (h[0] + 2.0 * h[1]) * c[0];
						break;
					default:
						throw new NegativeArraySizeException("Invalid length of data");
				}
			}
			break;
		case 3:
			if (4 <= c.length) {
				s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2];
				s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3]);
				for (int i = 2; (i < (c.length - 2)); i++) {
					s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1])
						+ h[2] * (c[i - 2] + c[i + 2]);
				}
				s[s.length - 2] = h[0] * c[c.length - 2]
					+ h[1] * (c[c.length - 3] + c[c.length - 1])
					+ h[2] * (c[c.length - 4] + c[c.length - 2]);
				s[s.length - 1] = h[0] * c[c.length - 1]
					+ 2.0 * h[1] * c[c.length - 2] + 2.0 * h[2] * c[c.length - 3];
			}
			else {
				switch (c.length) {
					case 3:
						s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2];
						s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + 2.0 * h[2] * c[1];
						s[2] = h[0] * c[2] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[0];
						break;
					case 2:
						s[0] = (h[0] + 2.0 * h[2]) * c[0] + 2.0 * h[1] * c[1];
						s[1] = (h[0] + 2.0 * h[2]) * c[1] + 2.0 * h[1] * c[0];
						break;
					case 1:
						s[0] = (h[0] + 2.0 * (h[1] + h[2])) * c[0];
						break;
					default:
						throw new NegativeArraySizeException("Invalid length of data");
				}
			}
			break;
		case 4:
			if (6 <= c.length) {
				s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
					+ 2.0 * h[3] * c[3];
				s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3])
					+ h[3] * (c[2] + c[4]);
				s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + h[2] * (c[0] + c[4])
					+ h[3] * (c[1] + c[5]);
				for (int i = 3; (i < (c.length - 3)); i++) {
					s[i] = h[0] * c[i] + h[1] * (c[i - 1] + c[i + 1])
						+ h[2] * (c[i - 2] + c[i + 2]) + h[3] * (c[i - 3] + c[i + 3]);
				}
				s[s.length - 3] = h[0] * c[c.length - 3]
					+ h[1] * (c[c.length - 4] + c[c.length - 2])
					+ h[2] * (c[c.length - 5] + c[c.length - 1])
					+ h[3] * (c[c.length - 6] + c[c.length - 2]);
				s[s.length - 2] = h[0] * c[c.length - 2]
					+ h[1] * (c[c.length - 3] + c[c.length - 1])
					+ h[2] * (c[c.length - 4] + c[c.length - 2])
					+ h[3] * (c[c.length - 5] + c[c.length - 3]);
				s[s.length - 1] = h[0] * c[c.length - 1] + 2.0 * h[1] * c[c.length - 2]
					+ 2.0 * h[2] * c[c.length - 3] + 2.0 * h[3] * c[c.length - 4];
			}
			else {
				switch (c.length) {
					case 5:
						s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
							+ 2.0 * h[3] * c[3];
						s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3])
							+ h[3] * (c[2] + c[4]);
						s[2] = h[0] * c[2] + (h[1] + h[3]) * (c[1] + c[3])
							+ h[2] * (c[0] + c[4]);
						s[3] = h[0] * c[3] + h[1] * (c[2] + c[4]) + h[2] * (c[1] + c[3])
							+ h[3] * (c[0] + c[2]);
						s[4] = h[0] * c[4] + 2.0 * h[1] * c[3] + 2.0 * h[2] * c[2]
							+ 2.0 * h[3] * c[1];
						break;
					case 4:
						s[0] = h[0] * c[0] + 2.0 * h[1] * c[1] + 2.0 * h[2] * c[2]
							+ 2.0 * h[3] * c[3];
						s[1] = h[0] * c[1] + h[1] * (c[0] + c[2]) + h[2] * (c[1] + c[3])
							+ 2.0 * h[3] * c[2];
						s[2] = h[0] * c[2] + h[1] * (c[1] + c[3]) + h[2] * (c[0] + c[2])
							+ 2.0 * h[3] * c[1];
						s[3] = h[0] * c[3] + 2.0 * h[1] * c[2] + 2.0 * h[2] * c[1]
							+ 2.0 * h[3] * c[0];
						break;
					case 3:
						s[0] = h[0] * c[0] + 2.0 * (h[1] + h[3]) * c[1] + 2.0 * h[2] * c[2];
						s[1] = h[0] * c[1] + (h[1] + h[3]) * (c[0] + c[2]) + 2.0 * h[2] * c[1];
						s[2] = h[0] * c[2] + 2.0 * (h[1] + h[3]) * c[1] + 2.0 * h[2] * c[0];
						break;
					case 2:
						s[0] = (h[0] + 2.0 * h[2]) * c[0] + 2.0 * (h[1] + h[3]) * c[1];
						s[1] = (h[0] + 2.0 * h[2]) * c[1] + 2.0 * (h[1] + h[3]) * c[0];
						break;
					case 1:
						s[0] = (h[0] + 2.0 * (h[1] + h[2] + h[3])) * c[0];
						break;
					default:
						throw new NegativeArraySizeException("Invalid length of data");
				}
			}
			break;
		default:
			throw new IllegalArgumentException(
				"Invalid filter half-length (should be [2..4])");
	}
}
 
} // end of class



















