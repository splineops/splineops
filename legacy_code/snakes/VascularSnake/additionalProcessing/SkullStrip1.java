package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;

public class SkullStrip1 extends Plugin{
	
	private static final int BACKGROUND = 0;
	private static final int STRONG	= 127;
	private byte[][][] output;
	
	private int nx, ny, nz;
	
	

	public void process(){
		System.out.println("p-1: Skull strip 1 processing");
		if (Icy.getMainInterface().getSequences(Const.HYST_Z).size() != 0){				
			Sequence hystZ = Icy.getMainInterface().getSequences(Const.HYST_Z).get(0);
			System.out.println("p0: Skull strip 1 processing");
			if (hystZ != null)
			if (Const.surfaceDetection != null){	
				System.out.println("p1: Skull strip 1 processing");
				
				double[][][][] orientation = Const.surfaceDetection.getOrientation();
				nx = hystZ.getSizeX();
				ny = hystZ.getSizeY();
				nz = hystZ.getSizeZ();
				
				double[] n = new double[3];
				double val = 0;
				
				ImageVolume volume = new ImageVolume(hystZ);
				output = new byte[nx][ny][nz];
				// border conditions missing
				for (int x=1;x<nx-1;x++) 
				for (int y=1;y<ny-1;y++) 
				for (int z=1;z<nz-1;z++) {
					
					val = hystZ.getData(0, z, 0, y, x);
					if (val > 0){
					
						n[0] = orientation[x][y][z][0];
						n[1] = orientation[x][y][z][1];
						n[2] = orientation[x][y][z][2];						
						
						boolean keep1 = true;
						boolean keep2 = true;
						
						for (int i=1; i<15; i++){							
							
							/*
							double xInt = i*n[0];
							double yInt = i*n[1];
							double zInt = i*n[2];								
							
							if (z-zInt >= 0  && y-yInt >= 0 && x-xInt >= 0)
							if (z-zInt < nz && y-yInt < ny && x-xInt < nx){
								//System.out.println("val = " + volume.getInterpolatedPixel(x-xInt, y-yInt, z-zInt));
							//if (hystZ.getData(0, z-zInt, 0, y-yInt, x-xInt) != 0)
							if (Math.round(volume.getInterpolatedPixel(x-xInt, y-yInt, z-zInt)) != 0.0)
								keep1 = false;		
							}
							
							if (z+zInt < nz && y+yInt < ny && x+xInt < nx)
							if (z+zInt >= 0  && y+yInt >= 0 && x+xInt >= 0)
							//if (hystZ.getData(0, z+zRound, 0, y+yRound, x+xRound) != 0)
							if (Math.round(volume.getInterpolatedPixel(x+xInt, y+yInt, z+zInt)) != 0.0)	
								keep2 = false;
							*/
							
							
							int xRound = (int)Math.round(i*n[0]);
							int yRound = (int)Math.round(i*n[1]);
							int zRound = (int)Math.round(i*n[2]);							
							
							if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
							if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
							if (hystZ.getData(0, z-zRound, 0, y-yRound, x-xRound) != 0)
								keep1 = false;						
							
							if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
							if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
							if (hystZ.getData(0, z+zRound, 0, y+yRound, x+xRound) != 0)
								keep2 = false;	
							
							
						}
						if (keep1 || keep2)						
							output[x][y][z] = STRONG;
						else
							output[x][y][z] = BACKGROUND;
					}
				}					
				showZ("Skull Strip 1 Z");
				//showX("Skull Strip 1 X");
				//showY("Skull Strip 1 Y");
			}
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
