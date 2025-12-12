package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;

public class Normalize extends Plugin{
	private double[][][] output;	
	private int nx, ny, nz;	
	
	public void process(){
		Sequence activeSeq = getActiveSequence();
		
		nx = activeSeq.getSizeX();
		ny = activeSeq.getSizeY();
		nz = activeSeq.getSizeZ();
		
		output = new double[nx][ny][nz];
		
		double max = 0;
		double val = 0;
		for (int x=0;x<nx-1;x++) 
		for (int y=0;y<ny-1;y++) 
		for (int z=0;z<nz-1;z++) {
			val = activeSeq.getData(0, z, 0, y, x);
			if (val > max)
				max = val;
		}
		for (int x=0;x<nx-1;x++) 
		for (int y=0;y<ny-1;y++) 
		for (int z=0;z<nz-1;z++) {
			val = activeSeq.getData(0, z, 0, y, x);		
			output[x][y][z] = val / max;
		}	
		
		System.out.println("max = " + max);
		showZ("normalized");
	}
	
	public void showZ(String title) {		
		
		Sequence sequence = new Sequence(); 		
		double current;
			
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					current  = output[i][j][k];
					temp[i+j*nx] = current;										
				}
			}             
			tempImage.dataChanged();
			sequence.addImage(tempImage);      
    	}                
		sequence.setName(title);
		addSequence(sequence);		
	}

}
