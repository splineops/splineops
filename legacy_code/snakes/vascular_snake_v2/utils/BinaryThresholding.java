package plugins.big.vascular.utils;

//import plugins.schmitter.brainsegmentation.steerable3Dfilter.ImageVolume;
import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;

public class BinaryThresholding extends Plugin{
	
	Sequence sequence;
	int threshold = 0;
	
	private double[][][] output;	
	private int nx, ny, nz;
	
	private static final double KEEP_VALUE = 10000000;
	private static final int BACKGROUND_VALUE = -10000000;
	
	public BinaryThresholding(Sequence seq, int t){
		sequence = seq;
		threshold = t;
		
		nx = seq.getSizeX();
		ny = seq.getSizeY();
		nz = seq.getSizeZ();		
		
		output = new double[nx][ny][nz];
	}
	
	public void process(){
		
		double val;
		for (int x=0;x<nx;x++) 
		for (int y=0;y<ny;y++) 
		for (int z=0;z<nz;z++){
			val = sequence.getData(0, z, 0, y, x);
			
			if (val == threshold)
				output[x][y][z] = KEEP_VALUE;
			else
				output[x][y][z] = BACKGROUND_VALUE;
		}		
		
		//if (Const.GUI2_chkSteps)
		String title = "binaryImage"; //"Step 8: binary image, t = " + threshold;
		showZ(title);
		
		System.out.println("bckg = " + BACKGROUND_VALUE + ", forg = " + KEEP_VALUE);
	}
	
	public double[][][] getOutputArray(){
		return output;
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
