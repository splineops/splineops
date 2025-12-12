package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;

public class Refine2 extends Plugin{
	
	private double[][][][] orientation;
	private int nx, ny, nz;	
	
	public byte[][][] output;	
	Sequence activeSeq;
	
	public Refine2(){
		activeSeq = getActiveSequence();
		
		nx = activeSeq.getSizeX();
		ny = activeSeq.getSizeY();
		nz = activeSeq.getSizeZ();	
	
		output = (new ImageVolume(activeSeq, true)).byteVolume;
		orientation = Const.surfaceDetection.getOrientation();		
	}
	
	public void run(){		
		Sequence boundarySeq = Icy.getMainInterface().getSequences("Step 4: Hysteresis Thresholding").get(0);
		double[][][] boundaryVol = (new ImageVolume(boundarySeq)).volume;
		
		double halfX = (nx-1)/2.5;
		double halfY = (ny-1)/3.0;
		double halfZ = (nz-1)/2.0;
		
		boolean[][][] stop = new boolean[nx][ny][nz];
		for (int x=0;x<nx;x++) 
		for (int y=0;y<ny;y++) 
		for (int z=0;z<nz;z++)
			stop[x][y][z] = true;
		
		
		double[] n = new double[3];
		double val = 0;		
		
		int DIST = 20;
		
		for (int x=1;x<nx-1;x++) 
		for (int y=1;y<ny-1;y++) 
		for (int z=1;z<nz-1;z++) {
			
			val = activeSeq.getData(0, z, 0, y, x);
			if (val > 0){
				
				stop[x][y][z] = false;
				
				n[0] = orientation[x][y][z][0];
				n[1] = orientation[x][y][z][1];
				n[2] = orientation[x][y][z][2];
				
				
				for (int i=1; i<DIST; i++)
				if (!stop[x][y][z]){	
					
					double xDir = i*n[0];
					double yDir = i*n[1];
					double zDir = i*n[2];						
					
					double xP = x + xDir;
					double yP = y + yDir;
					double zP = z + zDir;
					
					double xPD = xP - halfX;
					double yPD = yP - halfY;
					double zPD = zP - halfZ;
					
					if (Math.sqrt(x*1 + y*y + z*z) > Math.sqrt(xPD*xPD + yPD*yPD + zPD*zPD)){
						xP = x - xDir;
						yP = y - yDir;
						zP = z - zDir;					
					}
					
					if (Const.getInterpolatedPixel(xP, yP, zP, nx, ny, nz, boundaryVol) == 0){
						stop[x][y][z] = true;
						
						int xRound = (int)Math.round(xP);
						int yRound = (int)Math.round(yP);
						int zRound = (int)Math.round(zP);
						
						if (xRound > 0 && xRound < nx && yRound > 0 && yRound < ny && zRound > 0 && zRound <nz){
							output[x][y][z] = 0;
							output[xRound][yRound][zRound] = 127;
						}
					}
						
				}
			}
		}
		showZ("Refined");
		
		
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
