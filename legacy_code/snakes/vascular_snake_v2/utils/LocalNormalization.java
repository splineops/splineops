package plugins.big.vascular.utils;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;

import javax.swing.JTextArea;

import additionaluserinterface.WalkBar;

public class LocalNormalization extends Plugin{
	
	private JTextArea logWindow;
	private WalkBar walk;
	
	private double localMean;
	private double localVariance;
	
	double data[][][]; // input data
	public double volume[][][]; // output data
	private int nx;
	private int ny;
	private int nz;
	
	public double[][][] inputAllInOne;
	
	private static double TOLERANCE = 9.9999999999999995E-07D;
	
	private boolean showInterfaceTools;
	
	public LocalNormalization(JTextArea logWindow, WalkBar walk, double localMean, double localVariance){
		this.logWindow = logWindow;
		this.walk = walk;
		this.localMean = localMean;
		this.localVariance = localVariance;
		
		showInterfaceTools = true;
	}
	
	public LocalNormalization(double localMean, double localVariance, double[][][] input){
		this.localMean = localMean;
		this.localVariance = localVariance;
		inputAllInOne = input;
		
		showInterfaceTools = false;
	}
	
	public void normalize(){
		
		
		if (showInterfaceTools){
			logWindow.append("\nstart local normalization");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
			
			Sequence sequence = getFocusedSequence();
			
			if (sequence == null) 		
				return;
			
			nx = sequence.getSizeX();
			ny = sequence.getSizeY();
			nz = sequence.getSizeZ();
			
			data = new double[nx][ny][nz];
			
			for(int x=0 ; x<nx; x++)
		    for(int y=0; y<ny; y++)
		    for(int z=0; z<nz; z++)
		    	data[x][y][z] = sequence.getData(0, z, 0, y, x);
		}
		else{
			nx = inputAllInOne.length;
			ny = (inputAllInOne[0]).length;
			nz = (inputAllInOne[0][0]).length;
			
			data = inputAllInOne;
		}
		
		volume = new double[nx][ny][nz];
      
        double tmp[][][] = smoothGaussian(data, localMean); 
        for(int x=0 ; x<nx; x++)
        for(int y=0; y<ny; y++)
        for(int z=0; z<nz; z++){
        	data[x][y][z] = data[x][y][z] - tmp[x][y][z];
            tmp[x][y][z] = data[x][y][z];
            tmp[x][y][z] = tmp[x][y][z] * tmp[x][y][z];
        }  
        
        double var[][][] = smoothGaussian(tmp, localVariance);
        
        for(int x=0 ; x<nx; x++)
        for(int y=0; y<ny; y++)
        for(int z=0; z<nz; z++){
        	var[x][y][z] = Math.sqrt(var[x][y][z]);
            volume[x][y][z] = data[x][y][z] / var[x][y][z];
        } 
        
        if (showInterfaceTools){
        	logWindow.append("\nlocal normalization done.");
        	logWindow.setCaretPosition(logWindow.getDocument().getLength());
        }
        
        if (Const.GUI2_chkSteps)
        	showZ("Step 1: local normalization, mean = " + localMean + ", var = " + localVariance);
	}
	
	
	/*
	 * smoothGaussian
	 */
	private double[][][] smoothGaussian(double in[][][], double sigma){	
		if (showInterfaceTools)
			walk.reset();

		double out[][][] = new double[nx][ny][nz];
        int N = 3;
        double poles[] = new double[N];
        double s2 = sigma * sigma;
        double alpha = (1.0 + (double)N / s2) - Math.sqrt((double)(N * N) + (double)(2 * N) * s2) / s2;
        poles[0] = poles[1] = poles[2] = alpha;
        
        // x-direction
        double row[] = new double[nx];
        
        for(int z=0; z<nz; z++){
        	if (showInterfaceTools)
        		walk.progress("smooth Gaussian X.. ", (100.0*z)/nz);
        		
        	for(int y = 0; y < ny; y++){
        	
        		for(int x = 0; x < nx; x++)
        			row[x] = in[x][y][z];

        		convolveIIR(row, poles);
        		for(int x = 0; x < nx; x++)
        			out[x][y][z] = row[x];
        	}
        }

        // y-direction
        if (showInterfaceTools){
        	walk.finish();
        	walk.reset();
        }
        double col[] = new double[ny];
        
        for(int z=0; z<nz; z++){
        	
        	if (showInterfaceTools)
        		walk.progress("smooth Gaussian Y.. ", (100.0*z)/nz);
        	
        	for(int x = 0; x < nx; x++){
        	
        		for(int y = 0; y < ny; y++)
        			col[y] = out[x][y][z];

        		convolveIIR(col, poles);
        		for(int y = 0; y < ny; y++)
        			out[x][y][z] = col[y];
        	}
        }
        
        // z-direction
        if (showInterfaceTools){
        	walk.finish();
        	walk.reset();
        }
        
        double heights[] = new double[nz];
        
        for(int y=0; y<ny; y++){
        	
        	if (showInterfaceTools)
        		walk.progress("smooth Gaussian Z.. ", (100.0*y)/ny);
        	
        	for(int x = 0; x < nx; x++){
        	
        		for(int z=0; z<nz; z++)
        			heights[z] = out[x][y][z];

        		convolveIIR(heights, poles);
        		for(int z=0; z<nz; z++)
        			out[x][y][z] = heights[z];
        	}
        }

        if (showInterfaceTools)
        	walk.finish("smooth Gaussian done.");
        
        return out;
    }
	
	
	/*
	 * convolveIIR
	 */
	private void convolveIIR(double signal[], double poles[]){
        double lambda = 1.0D;
        for(int k = 0; k < poles.length; k++)
            lambda = lambda * (1.0D - poles[k]) * (1.0D - 1.0D / poles[k]);

        for(int n = 0; n < signal.length; n++)
            signal[n] = signal[n] * lambda;

        for(int k = 0; k < poles.length; k++)
        {
            signal[0] = getInitialCausalCoefficientMirror(signal, poles[k]);
            for(int n = 1; n < signal.length; n++)
                signal[n] = signal[n] + poles[k] * signal[n - 1];

            signal[signal.length - 1] = getInitialAntiCausalCoefficientMirror(signal, poles[k]);
            for(int n = signal.length - 2; n >= 0; n--)
                signal[n] = poles[k] * (signal[n + 1] - signal[n]);

        }

    }
	
	
	private double getInitialAntiCausalCoefficientMirror(double c[], double z)
    {
        return ((z * c[c.length - 2] + c[c.length - 1]) * z) / (z * z - 1.0D);
    }

    private double getInitialCausalCoefficientMirror(double c[], double z)
    {
        double z1 = z;
        double zn = Math.pow(z, c.length - 1);
        double sum = c[0] + zn * c[c.length - 1];
        int horizon = c.length;
        if(0.0 < TOLERANCE)
        {
            horizon = 2 + (int)(Math.log(TOLERANCE) / Math.log(Math.abs(z)));
            horizon = horizon >= c.length ? c.length : horizon;
        }
        zn *= zn;
        for(int n = 1; n < horizon - 1; n++)
        {
            zn /= z;
            sum += (z1 + zn) * c[n];
            z1 *= z;
        }

        return sum / (1.0D - Math.pow(z, 2 * c.length - 2));
    }
    
    
    public void showZ(String title) {		
		Sequence sequence = new Sequence(); 
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double current;
			
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					current  = volume[i][j][k];
					temp[i+j*nx] = current;
					if (current < min) 
						min = current;
					
					if (current > max) 
						max = current;					
				}
			}    
           
			tempImage.dataChanged();
			sequence.addImage(tempImage);      
    	}                
		sequence.setName(title);
		addSequence(sequence);		
	}
	
	
	public void showY(String title) {		
		Sequence sequence = new Sequence(); 			
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double current;
			
		//for (int y=ny-1;y>=0;y--) {
		for (int y=0; y<ny; y++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, nz, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=0;z<nz;z++) {
				for (int x=0;x<nx;x++) {
					current  = volume[x][y][z];
					temp[x+z*nx] = current;
					if (current < min) 
						min = current;
					
					if (current > max) 
						max = current;					
				}
			}  	
			tempImage.dataChanged();
			sequence.addImage(tempImage);
		}
		sequence.setName(title);
		addSequence(sequence);		
	}
	
	
	public void showX(String title) {		
		Sequence sequence = new Sequence(); 
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double current;
			
		//for (int x=nx-1;x>=0;x--) {
		for (int x=0; x<nx; x++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(ny, nz, 1, DataType.DOUBLE);		
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=nz-1;z>=0;z--) {
				for (int y=ny-1;y>=0;y--) {        
					current  = volume[x][y][z];	//current  = volume[x][ny-y-1][z];	        
					temp[y+z*ny] = current;
					if (current < min) 
						min = current;
					
					if (current > max) 
						max = current;					
				}
			}         	
			tempImage.dataChanged();
			sequence.addImage(tempImage);
		}
		sequence.setName(title);     
		addSequence(sequence);		
	}

}
