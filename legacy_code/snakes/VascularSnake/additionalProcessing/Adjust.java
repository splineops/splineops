package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;

public class Adjust extends Plugin{
	
	double TH = 0.999;
	double TL = 0.01;
	
	int T2_INV2 = 3000; // INV2 = 3000; INV1 == 1500;
	int T1_INV2 = 50; // INV 2 = 50; INV1 = 0
	
	int T2_INV1 = 1500;
	int T1_INV1 = 0;
	
	int nx, ny, nz;
	
	double[][][] output1;
	double[][][] output2;	

	public void process(){
		//Sequence activeSeq = getFocusedSequence();
		//output1 = (new ImageVolume(Icy.getMainInterface().getSequences("DR_timeRes_1_INV1").get(0))).volume;
		//output2 = (new ImageVolume(Icy.getMainInterface().getSequences("DR_timeRes_1_INV2").get(0))).volume;	
		
		double[][][] outputCombined = (new ImageVolume(Icy.getMainInterface().getSequences("GB_timeRes_1").get(0))).volume;	
		double[][][] orig = (new ImageVolume(Icy.getMainInterface().getSequences("GB_timeRes_1_orig").get(0))).volume;	
		
		//double max = Double.MIN_VALUE;		
		
		nx = outputCombined.length;
		ny = (outputCombined[0]).length;
		nz = (outputCombined[0][0]).length;
		
		/*
		for (int z=0; z<nz; z++) 
		for (int y=0;y<ny;y++) 
		for (int x=0;x<nx;x++){ 
			if (output1[x][y][z] < T1_INV1)
				output1[x][y][z] = 0;
			if (output1[x][y][z] > T2_INV1)
				output1[x][y][z] = 0;
			
			if (output2[x][y][z] < T1_INV2)
				output2[x][y][z] = 0;
			if (output2[x][y][z] > T2_INV2)
				output2[x][y][z] = 0;
		}
		
		// linear histogram equalization
		int K = 64; // number of intensity values
		int N = nx*ny*nz; // total number of voxels			
		
		*/
		
		int t1l = 160;
		int t2l = 160; 
		int t3h = 2500;
		
		int t1h = 300;
		int t2h = 700;
		
		
		double[][][] output3 = new double[nx][ny][nz];
		for (int z=0; z<nz; z++) 
		for (int y=0;y<ny;y++) 
		for (int x=0;x<nx;x++) {	
			
			
			//double val1 = output1[x][y][z]/T2_INV1*K;
			//double val2 = output2[x][y][z]/T2_INV2*K;
			
			/*
			output1[x][y][z] = 	val1;
			output2[x][y][z] = 	val2;
			*/
			
			//double val1 = output1[x][y][z];
			//double val2 = output2[x][y][z];
			double val3 = orig[x][y][z];
			double val4 = outputCombined[x][y][z];
			
			if (val3 > 2700)
				output3[x][y][z] = 0;
			else 
				output3[x][y][z] = outputCombined[x][y][z];			
					
		}	
		
		
		/*
		for (int z=0; z<nz; z++) 
		for (int y=0;y<ny;y++) 
		for (int x=0;x<nx;x++) 
		if (output[x][y][z] > max) 
	    	max = output[x][y][z];
		
		int K = (int)Math.ceil(max); // number of intensity values
		
		
		int[] H = new int[B+1]; // histogram
		
		for (int z=0; z<nz; z++) 
		for (int y=0;y<ny;y++) 
		for (int x=0;x<nx;x++) {
			int val = (int)output[x][y][z];
			if (val > 0){
				int i = val*B/K;			
				H[i] = H[i]+1;
			}
		}
		
		// cumulative histogram
		for (int i=1; i<H.length; i++)
			H[i] = H[i-1] + H[i];
		
		int N = nx*ny*nz; // total number of voxels
		
		int sL = (int)(N*TL); // saturation low
		int sH = (int)(N*TH); // saturation high
		
		int newMin = Integer.MAX_VALUE;
		int newMax = 0;
		
		
		for (int i=0; i<H.length; i++)
		if (H[i] >= sL){
			newMin = i;
			break;
		}
		
		
		for (int i=H.length-1; i>0; i--)
		if (H[i] <= sH){
			newMax = i;
			break;
		}
		
		System.out.println("newMin = " + newMin + ", newMax = " +newMax);
		
		for (int z=0; z<nz; z++) 
		for (int y=0;y<ny;y++) 
		for (int x=0;x<nx;x++) {
			double val = output[x][y][z];
			if (val < newMin){
				output[x][y][z] = 0;
				continue;
			}
			if (val > newMax){
				output[x][y][z] = 0;
				continue;
			}			
			
		}
		*/
		
		//showZ("inv 1", output1);
		//showZ("inv 2", output2);
		showZ("image 3", output3);
			
	}
	
	
	/*
	 * showZ
	 */
	public void showZ(String title, double[][][] output) {				
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
