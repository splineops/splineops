package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;

public class ApplyMask extends Plugin{
	
	private Sequence activeSeq;
	private Sequence binaryMask;
	
	private double[][][] output;	
	private int nx, ny, nz;	
	
	public ApplyMask(){
		activeSeq = getFocusedSequence();
		binaryMask = Icy.getMainInterface().getSequences("Binary mask").get(0);//Const.binaryMask;
		
		//String strMask = "tiv_mask.nii"; //"BinaryMask_DS";
		//binaryMask = Icy.getMainInterface().getSequences(strMask).get(0);
		
		nx = activeSeq.getSizeX();
		ny = activeSeq.getSizeY();
		nz = activeSeq.getSizeZ();
		
		output = new double[nx][ny][nz];
	}
	
	public void process(){
		
		double val;
		for (int x=0;x<nx;x++) 
		for (int y=0;y<ny;y++) 
		for (int z=0;z<nz;z++) {				
			val = binaryMask.getData(0, z, 0, y, x);
			
			if (val > 0){	// val == 0			
				output[x][y][z] = activeSeq.getData(0, z, 0, y, x);
			}
		}
		
		showZ ("result segmentation Z");
	}
	
	public void showZ(String title) {		
		
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

}
