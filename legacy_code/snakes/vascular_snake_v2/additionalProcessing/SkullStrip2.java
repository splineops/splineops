package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.utils.Const;


public class SkullStrip2 extends Plugin{
	
	private double[][][] output;
	
	private int nx, ny, nz;
	
	public void process(){		
		System.out.println("p-2: Skull strip 1 processing");
		if (Icy.getMainInterface().getSequences(Const.HYST_Z).size() != 0){			
			System.out.println("p-1: Skull strip 2 processing");
			Sequence hystZ = Icy.getMainInterface().getSequences(Const.HYST_Z).get(0);
			if (Icy.getMainInterface().getSequences(Const.NMS_Z).size() != 0){	
				System.out.println("p 0: Skull strip 2 processing");
				Sequence NMSZ = Icy.getMainInterface().getSequences(Const.NMS_Z).get(0);
				if (hystZ != null && NMSZ != null){
					System.out.println("p 1: Skull strip 2 processing");
					
					nx = hystZ.getSizeX();
					ny = hystZ.getSizeY();
					nz = hystZ.getSizeZ();
					
					output = new double[nx][ny][nz];
					double val;
					for (int x=1;x<nx-1;x++) 
					for (int y=1;y<ny-1;y++) 
					for (int z=1;z<nz-1;z++) {
							
							val = hystZ.getData(0, z, 0, y, x);
							if (val == 0){
								output[x][y][z] = NMSZ.getData(0, z, 0, y, x);
							}
							else
								output[x][y][z] = 0;
					}
							
					showZ("Skull Strip 2 Z");
				}
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
