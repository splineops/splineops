package plugins.big.vascular.additionalProcessing;



import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;



public class SkullStrip3 extends Plugin{
	
	private double[][][] output;	
	private int nx, ny, nz;	

	public void process(){
		Sequence activeSeq = getActiveSequence();
		Sequence hystZ = Icy.getMainInterface().getSequences(Const.HYST_Z).get(0);
		
		nx = hystZ.getSizeX();
		ny = hystZ.getSizeY();
		nz = hystZ.getSizeZ();
		
		double[] n = new double[3];
		double val = 0;
		
		double[][][][] orientation = Const.surfaceDetection.getOrientation();
		
		output = (new ImageVolume(activeSeq)).volume;
				
		for (int x=0;x<nx;x++) 
		for (int y=0;y<ny;y++) 
		for (int z=0;z<nz;z++) {
				
			val = hystZ.getData(0, z, 0, y, x);
			if (val > 0){
				
				n[0] = orientation[x][y][z][0];
				n[1] = orientation[x][y][z][1];
				n[2] = orientation[x][y][z][2];
				
				for (int i=0; i<10; i++){						
					
					// all floor = 0 0 0
					int xRound = (int)Math.floor(i*n[0]);
					int yRound = (int)Math.floor(i*n[1]);
					int zRound = (int)Math.floor(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;	
					
					// all floor = 0 1 0
					xRound = (int)Math.floor(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.floor(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;	
					
					// all ceil = 1 1 1
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;
					//
					
					// all ceil = 1 0 1
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.floor(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;
					//
					
					// all ceil = 1 1 0
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.floor(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;
					//
					
					// all ceil = 1 0 0
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.floor(i*n[1]);
					zRound = (int)Math.floor(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;
					//
					
					
					// x floor y floor z ceil = 0 0 1
					xRound = (int)Math.floor(i*n[0]);
					yRound = (int)Math.floor(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;
					//
					
					// x floor y ceil z ceil = 0 1 1
					xRound = (int)Math.floor(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						output[x-xRound][y-yRound][z-zRound] = 0;						
												
					
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						output[x+xRound][y+yRound][z+zRound] = 0;
					//
					
				}
					
			}
		}
		showZ("skull strip 3 Z");
		//showX("skull strip 3 X");
		//showY("skull strip 3 Y");
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
