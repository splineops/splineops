package plugins.big.vascular.utils;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;

import javax.swing.JTextArea;

import plugins.big.vascular.steerable3DFilter.ImageVolume;
import additionaluserinterface.WalkBar;


public class HysteresisThresholding extends Plugin{
	private WalkBar walk;
	private JTextArea logWindow;
	private boolean showInterfaceTools;
	
	private static final int STRONG	= 127;
	private static final int CANDIDATE = 64;
	private static final int BACKGROUND = 0;
	
	private double tl, th;
	public byte[][][] output;
	
	double[][][] input;
	// Image dimensions
	final int nx;
	final int ny;
	final int nz;
	
	public HysteresisThresholding(ImageVolume inputImage, double tl, double th, WalkBar walk, JTextArea logWindow){
		this.walk = walk;
		this.logWindow = logWindow;
		this.tl = tl;
		this.th = th;	
		input = inputImage.getVolumeArray();		
		nx = input.length;
		ny = (input[0]).length;
		nz = (input[0][0]).length;
		
		showInterfaceTools = true;
	}
	
	public HysteresisThresholding(ImageVolume inputImage, double tl, double th){		
		this.tl = tl;
		this.th = th;	
		input = inputImage.getVolumeArray();		
		nx = input.length;
		ny = (input[0]).length;
		nz = (input[0][0]).length;
		
		showInterfaceTools = false;
	}
	
	public void process(){	
		if (showInterfaceTools){
			logWindow.append("\nstart hysteresis thresholding");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}

		double max = Double.MIN_VALUE;

		for (int z=0; z<nz; z++) 
		for (int y=0;y<ny;y++) 
		for (int x=0;x<nx;x++) 
		if (input[x][y][z] > max) 
    		max = input[x][y][z];

		tl = tl / 100.0 * max;
		th = th / 100.0 * max;

		output = new byte[nx][ny][nz];

		for (int z=0; z<nz; z++) 
		for (int x=0; x<nx; x++) 
		for (int y=0; y<ny; y++) {
			if (input[x][y][z] > th) 
				output[x][y][z] = STRONG;			
			else {
				if (input[x][y][z] > tl) 
					output[x][y][z] = CANDIDATE;				
				else 
					output[x][y][z] = BACKGROUND;				
			}
		}
		
		Queue Q = new Queue();
		
		if (showInterfaceTools)
			walk.reset();

		for (int z=1; z<nz-1; z++) {  
			
			if (showInterfaceTools)
				walk.progress("thresholding.. ", (100.0*z)/nz);
			
			for (int x=1; x<nx-1; x++) {
				for (int y=1; y<ny-1; y++) {
					if (output[x][y][z] == STRONG) {
					
						if (output[x-1][y-1][z-1] == CANDIDATE)
							Q.enqueue(new Vector3(x-1,y-1,z-1));
						if (output[x][y-1][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y-1, z-1));
						if (output[x+1][y-1][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y-1, z-1));
						if (output[x-1][y][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y, z-1));
						if (output[x][y][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y, z-1));
						if (output[x+1][y][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y, z-1));
						if (output[x-1][y+1][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y+1, z-1));
						if (output[x][y+1][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y+1, z-1));
						if (output[x+1][y+1][z-1] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y+1, z-1));
							
						if (output[x-1][y-1][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y-1, z));
						if (output[x][y-1][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y-1, z));
						if (output[x+1][y-1][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y-1, z));
						if (output[x-1][y][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y, z));
						if (output[x+1][y][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y, z));
						if (output[x-1][y+1][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y+1, z));
						if (output[x][y+1][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y+1, z));
						if (output[x+1][y+1][z] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y+1, z));
						
						if (output[x-1][y-1][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y-1, z+1));
						if (output[x][y-1][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y-1, z+1));
						if (output[x+1][y-1][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y-1, z+1));
						if (output[x-1][y][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y, z+1));
						if (output[x][y][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y, z+1));
						if (output[x+1][y][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y, z+1));
						if (output[x-1][y+1][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x-1, y+1, z+1));
						if (output[x][y+1][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x, y+1, z+1));
						if (output[x+1][y+1][z+1] == CANDIDATE) 
							Q.enqueue(new Vector3(x+1, y+1, z+1));
					
					}
				}
			}
		}
		
		if (showInterfaceTools)
			logWindow.append("\nthresholding ..");		
		
		while(!Q.isEmpty()) {			
			
			Vector3 current = (Vector3) Q.dequeue();
			int xc = current.getX().intValue();
			int yc = current.getY().intValue();
			int zc = current.getZ().intValue();
			
			if (output[xc][yc][zc] != CANDIDATE)
				continue;
			
			output[xc][yc][zc] = STRONG;
			
			if (zc>0) { 
				if (yc>0) {
					if (xc>0 && output[xc-1][yc-1][zc-1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc-1, yc-1, zc-1));
					if (output[xc][yc-1][zc-1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc, yc-1, zc-1));
					if (xc<nx-1 && output[xc+1][yc-1][zc-1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc+1, yc-1, zc-1));
				}	
				
				if (xc>0 && output[xc-1][yc][zc-1] == CANDIDATE)
					Q.enqueue(new Vector3(xc-1, yc, zc-1));
				if (output[xc][yc][zc-1] == CANDIDATE) 
					Q.enqueue(new Vector3(xc, yc, zc-1));
				if (xc<nx-1 && output[xc+1][yc][zc-1] == CANDIDATE) 
					Q.enqueue(new Vector3(xc+1, yc, zc-1));
					
				if (yc<ny-1) {
					if (xc>0 && output[xc-1][yc+1][zc-1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc-1, yc+1, zc-1));
					if (output[xc][yc+1][zc-1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc, yc+1, zc-1));
					if (xc<nx-1 && output[xc+1][yc+1][zc-1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc+1, yc+1, zc-1));
				}
			}
			
			if (yc>0) {
				if (xc>0 && output[xc-1][yc-1][zc] == CANDIDATE) 
					Q.enqueue(new Vector3(xc-1, yc-1, zc));
				if (output[xc][yc-1][zc] == CANDIDATE) 
					Q.enqueue(new Vector3(xc, yc-1, zc));
				if (xc<nx-1 && output[xc+1][yc-1][zc] == CANDIDATE) 
					Q.enqueue(new Vector3(xc+1, yc-1, zc));
			}
			
			if (xc>0 && output[xc-1][yc][zc] == CANDIDATE) 
				Q.enqueue(new Vector3(xc-1, yc, zc));
			if (xc<nx-1 && output[xc+1][yc][zc] == CANDIDATE) 
				Q.enqueue(new Vector3(xc+1, yc, zc));
			
			if (yc<ny-1) {
				if (xc>0 && output[xc-1][yc+1][zc] == CANDIDATE) 
					Q.enqueue(new Vector3(xc-1, yc+1, zc));
				if (output[xc][yc+1][zc] == CANDIDATE) 
					Q.enqueue(new Vector3(xc, yc+1, zc));
				if (xc<nx-1 && output[xc+1][yc+1][zc] == CANDIDATE) 
					Q.enqueue(new Vector3(xc+1, yc+1, zc));
			}
			
			if (zc<nz-1) {
				if (yc>0) {
					if (xc>0 && output[xc-1][yc-1][zc+1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc-1, yc-1, zc+1));
					if (output[xc][yc-1][zc+1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc, yc-1, zc+1));
					if (xc<nx-1 && output[xc+1][yc-1][zc+1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc+1, yc-1, zc+1));
				}
				
				if (xc>0 && output[xc-1][yc][zc+1] == CANDIDATE) 
					Q.enqueue(new Vector3(xc-1, yc, zc+1));
				if (output[xc][yc][zc+1] == CANDIDATE) 
					Q.enqueue(new Vector3(xc, yc, zc+1));
				if (xc<nx-1 && output[xc+1][yc][zc+1] == CANDIDATE) 
					Q.enqueue(new Vector3(xc+1, yc, zc+1));
				
				if (yc<ny-1) {
					if (xc>0 && output[xc-1][yc+1][zc+1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc-1, yc+1, zc+1));
					if (output[xc][yc+1][zc+1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc, yc+1, zc+1));
					if (xc<nx-1 && output[xc+1][yc+1][zc+1] == CANDIDATE) 
						Q.enqueue(new Vector3(xc+1, yc+1, zc+1));
				}
			}
			
		}

		for (int z=0; z<nz; z++) {    
			for (int x=0; x<nx; x++) {
				for (int y=0; y<ny; y++) {
					if (output[x][y][z] == CANDIDATE)
						output[x][y][z] = BACKGROUND;
				}
			}
		}
		
		//if (Const.GUI2_chkSteps)
			showZ("Step 4: Hysteresis Thresholding");
			
		
		if (showInterfaceTools){
			walk.finish("thresholding done.");
			logWindow.append("\nhysteresis thresholding done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
	}
	
	
	/*
	 * showZ
	 */
	public void showZ(String title) {		
		Const.HYST_Z = title;
		Sequence sequence = new Sequence(); 		
		double current;
			
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					current  = (int)output[i][j][k];
					temp[i+j*nx] = current;										
				}
			}             
			tempImage.dataChanged();
			sequence.addImage(tempImage);      
    	}                
		sequence.setName(title);
		addSequence(sequence);		
	}	
	
	/*
	 * showY
	 */
	public void showY(String title) {		
		Const.HYST_Y = title;
		Sequence sequence = new Sequence(); 		
		double current;
			
		//for (int y=ny-1;y>=0;y--) {
		for (int y=0; y<ny; y++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, nz, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=0;z<nz;z++) {
				for (int x=0;x<nx;x++) {
					current  = (int)output[x][y][z];
					temp[x+z*nx] = current;									
				}
			}  	
			tempImage.dataChanged();
			sequence.addImage(tempImage);
		}
		sequence.setName(title);
		addSequence(sequence);		
	}
	
	/*
	 * showX
	 */
	public void showX(String title) {		
		Const.HYST_X = title;
		Sequence sequence = new Sequence(); 		
		double current;
			
		//for (int x=nx-1;x>=0;x--) {
		for (int x=0;x<nx;x++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(ny, nz, 1, DataType.DOUBLE);		
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=nz-1;z>=0;z--) {
				for (int y=ny-1;y>=0;y--) {        
					current  = (int)output[x][y][z];	        
					temp[y+z*ny] = current;									
				}
			}         	
			tempImage.dataChanged();
			sequence.addImage(tempImage);
		}
		sequence.setName(title);     
		addSequence(sequence);		
	}
	
}
