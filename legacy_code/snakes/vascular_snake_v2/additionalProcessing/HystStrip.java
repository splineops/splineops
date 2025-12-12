package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;

public class HystStrip extends Plugin{
	
	private double[][][] output;	
	private int nx, ny, nz;	
	
	public void process(){
		Sequence activeSeq = getActiveSequence();
		
		nx = activeSeq.getSizeX();
		ny = activeSeq.getSizeY();
		nz = activeSeq.getSizeZ();
		
		double[] n = new double[3];
		double val = 0;
		
		double[][][][] orientation = Const.surfaceDetection.getOrientation();
		
		output = (new ImageVolume(activeSeq)).volume;
				
		int DIST = 25;
		
		for (int x=1;x<nx-1;x++) 
		for (int y=1;y<ny-1;y++) 
		for (int z=1;z<nz-1;z++) {
				
			val = activeSeq.getData(0, z, 0, y, x);
			if (val > 0){
				
				n[0] = orientation[x][y][z][0];
				n[1] = orientation[x][y][z][1];
				n[2] = orientation[x][y][z][2];
				
				boolean keep = true;				
				
				double[] p1 = {x, y, z};
				
				outerloop:
				for (int i=1; i<DIST; i++){							
					
					int xRound = (int)Math.round(i*n[0]);
					int yRound = (int)Math.round(i*n[1]);
					int zRound = (int)Math.round(i*n[2]);							
					
					if (keep)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){//activeSeq.getData(0, z-zRound, 0, y-yRound, x-xRound) != 0){
						double[] p2 = {x-xRound, y-yRound, z-zRound};
						keep = checkDistBorder(p1, p2);
						if (!keep)
							break outerloop;
						//output[x-xRound][y-yRound][z-zRound] = 0;
					}				
												
					if (keep)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){//activeSeq.getData(0, z+zRound, 0, y+yRound, x+xRound) != 0){
						double[] p2 = {x+xRound, y+yRound, z+zRound};
						keep = checkDistBorder(p1, p2);
						if (!keep)
							break outerloop;
						//output[x+xRound][y+yRound][z+zRound] = 0;
					}					
				}
				if (!keep)
					output[x][y][z] = 0;	
			}
			
		}
		showZ(Const.HYST_STRIP_Z);
		
		//showX("skull strip 3 X");
		//showY("skull strip 3 Y");
	}
	
	// return true if surface point p1 is nearest to the border
	private boolean checkDistBorder(double[] p1, double[] p2){	
		
		double halfX = (nx-1)/2.0;
		double halfY = (ny-1)/2.0;
		double halfZ = (nz-1)/2.0;
		
		double x1 = p1[0]-halfX;
		double x2 = p2[0]-halfX;
		double y1 = p1[1]-halfY;
		double y2 = p2[1]-halfY;
		double z1 = p1[2]-halfZ;
		double z2 = p2[2]-halfZ;
		
		if (Math.sqrt(x1*x1 + y1*y1 + z1*z1) > Math.sqrt(x2*x2 + y2*y2 + z2*z2))
			return true;
		return false;
		
		/*
		boolean xOK = true;
		boolean yOK = true;
		boolean zOK = true;
		
		if (p1[0] <= halfX && p2[0] <= halfX)
		if (p1[0] > p2[0])
			xOK = false;
		
		if (p1[0] > halfX && p2[0] > halfX)
		if (p1[0] < p2[0])	
			xOK = false;
		
		if (p1[1] <= halfY && p2[1] <= halfY)
		if (p1[1] > p2[1])
			yOK = false;
			
		if (p1[1] > halfY && p2[1] > halfY)
		if (p1[1] < p2[1])	
			yOK = false;
		
		if (p1[2] <= halfZ && p2[2] <= halfZ)
		if (p1[2] > p2[2])
			zOK = false;
				
		if (p1[2] > halfZ && p2[2] > halfZ)
		if (p1[2] < p2[2])	
			zOK = false;
		
		return xOK == yOK == zOK == true;	
		*/
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
