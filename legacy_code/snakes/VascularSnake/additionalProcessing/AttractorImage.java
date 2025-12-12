package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.utils.Const;

public class AttractorImage extends Plugin{
	private double[][][] output;	
	private int nx, ny, nz;	
	
	public void process(){
		
		Sequence hystZ = Icy.getMainInterface().getSequences(Const.HYST_Z).get(0);
		
		nx = hystZ.getSizeX();
		ny = hystZ.getSizeY();
		nz = hystZ.getSizeZ();
		
		double[] n = new double[3];
		double val = 0;
		
		double[][][][] orientation = Const.surfaceDetection.getOrientation();
		
		output = new double[nx][ny][nz];//(new ImageVolume(hystZ)).volume;
		int dist = 20;		
		
		for (int x=1;x<nx-1;x++) 
		for (int y=1;y<ny-1;y++) 
		for (int z=1;z<nz-1;z++) {
				
			val = hystZ.getData(0, z, 0, y, x);
			if (val > 0){
				output[x][y][z] = 1;				
				
				n[0] = orientation[x][y][z][0];
				n[1] = orientation[x][y][z][1];
				n[2] = orientation[x][y][z][2];
				
				boolean dir1 = true;
				boolean dir2 = true;
				
				for (int i=1; i<dist; i++){						
					
					int xRound = (int)Math.round(i*n[0]);
					int yRound = (int)Math.round(i*n[1]);
					int zRound = (int)Math.round(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (hystZ.getData(0, z-zRound, 0, y-yRound, x-xRound) != 0.0)					
						dir1 = false;											
					
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (hystZ.getData(0, z+zRound, 0, y+yRound, x+xRound) != 0.0)
						dir2 = false;					
				}
				
				
				if (dir1){					
					for (int i=1; i<dist; i++){						
						
						int xRound = (int)Math.round(i*n[0]);
						int yRound = (int)Math.round(i*n[1]);
						int zRound = (int)Math.round(i*n[2]);							
						
						if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
						if (z-zRound < nz && y-yRound < ny && x-xRound < nx){
							if (output[x-xRound][y-yRound][z-zRound] == 0.0)
								output[x-xRound][y-yRound][z-zRound] = 1/(double)(i+1);	
							else
								output[x-xRound][y-yRound][z-zRound] += 1/(double)(i+1);
						}
					}					
				}
				
				
				
				if (dir2){
					
					for (int i=1; i<dist; i++){						
						
						int xRound = (int)Math.round(i*n[0]);
						int yRound = (int)Math.round(i*n[1]);
						int zRound = (int)Math.round(i*n[2]);							
						
						if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
						if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0){
							if (output[x+xRound][y+yRound][z+zRound] == 0.0)
								output[x+xRound][y+yRound][z+zRound] = 1.0/(double)(i+1);
							else
								output[x+xRound][y+yRound][z+zRound] += 1.0/(double)(i+1);
							
								
						}
					}					
				}				
			}
		}
		showZ("Attractor image Z");
		//showX("Attractor image X");
		//showY("Attractor image Y");		
	}
	
	/*
	 * showZ
	 */
	public void showZ(String title) {		
		
		Sequence sequence = new Sequence(); 		
		double current;
			
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {					
					current  = 1000*output[i][j][k];					
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
		
		Sequence sequence = new Sequence(); 		
		double current;
			
		//for (int y=ny-1;y>=0;y--) {
		for (int y=0; y<ny; y++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, nz, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=0;z<nz;z++) {
				for (int x=0;x<nx;x++) {
					current  = output[x][y][z];
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
		
		Sequence sequence = new Sequence(); 		
		double current;
			
		//for (int x=nx-1;x>=0;x--) {
		for (int x=0;x<nx;x++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(ny, nz, 1, DataType.DOUBLE);		
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=nz-1;z>=0;z--) {
				for (int y=ny-1;y>=0;y--) {        
					current  = output[x][y][z];	        
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
