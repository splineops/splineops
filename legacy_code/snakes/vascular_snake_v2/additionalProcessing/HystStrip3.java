package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;


public class HystStrip3 extends Plugin{
	
	private byte[][][] inputAllInOne; 
	private double[][][][] orientation;
	
	private boolean isAllInOne = false;
	
	private Sequence activeSeq;
	
	private byte[][][] output;	
	private int nx, ny, nz;	
	
	public HystStrip3(){}
	
	public HystStrip3(byte[][][] input, double[][][][] orientationAllInOne){
		inputAllInOne = input;
		orientation = orientationAllInOne;
		isAllInOne = true;
		
		nx = input.length;
		ny = (input[0]).length;
		nz = (input[0][0]).length;
	}
	
	public void process(){
		
		if (!isAllInOne){
			activeSeq = getActiveSequence();
		
			nx = activeSeq.getSizeX();
			ny = activeSeq.getSizeY();
			nz = activeSeq.getSizeZ();
			
			orientation = Const.surfaceDetection.getOrientation();
			
			output = (new ImageVolume(activeSeq, true)).byteVolume;
		}
		else
			output = inputAllInOne;
		
		double[] n = new double[3];
		double val = 0;
		
		orientation = Const.surfaceDetection.getOrientation();		
				
		int DIST = 20;
		
		for (int x=1;x<nx-1;x++) 
		for (int y=1;y<ny-1;y++) 
		for (int z=1;z<nz-1;z++) {
				
			if (!isAllInOne)
				val = activeSeq.getData(0, z, 0, y, x);
			else
				val = inputAllInOne[x][y][z];
			
			if (val > 0){
				
				n[0] = orientation[x][y][z][0];
				n[1] = orientation[x][y][z][1];
				n[2] = orientation[x][y][z][2];
				
				//boolean keep = true;				
				
				boolean keepMinus = true;
				boolean keepPlus = true;
				
				//double[] p1 = {x, y, z};				
				//double[][] p2Minus = new double[8][3];
				//double[][] p2Plus = new double[8][3];
				
				outerloop:
				for (int i=2; i<DIST; i++){							
					
					if (keepMinus || keepPlus){						
						
						// start 6 neighborhood						
						int xRound = (int)Math.round(i*n[0]);
						int yRound = (int)Math.round(i*n[1]);
						int zRound = (int)Math.round(i*n[2]);	
						
						if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
						if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
						if (output[x-xRound][y-yRound][z-zRound]!=0){
							keepMinus = false;
						}
						if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
						if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
						if (output[x+xRound][y+yRound][z+zRound]!=0){
							keepPlus = false;	
						}
						
						
						// z = -1
						int xRound2 = xRound;
						int yRound2 = yRound;
						int zRound2 = zRound -1;
						if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
						if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
						if (output[x-xRound2][y-yRound2][z-zRound2]!=0){
							keepMinus = false;
						}
						if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
						if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
						if (output[x+xRound2][y+yRound2][z+zRound2]!=0){
							keepPlus = false;	
						}
						
						// z = 0
						xRound2 = xRound;
						yRound2 = yRound -1;
						zRound2 = zRound;
						if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
						if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
						if (output[x-xRound2][y-yRound2][z-zRound2]!=0){
							keepMinus = false;
						}
						if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
						if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
						if (output[x+xRound2][y+yRound2][z+zRound2]!=0){
							keepPlus = false;	
						}
						
						xRound2 = xRound -1;
						yRound2 = yRound;
						zRound2 = zRound;
						if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
						if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
						if (output[x-xRound2][y-yRound2][z-zRound2]!=0){
							keepMinus = false;
						}
						if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
						if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
						if (output[x+xRound2][y+yRound2][z+zRound2]!=0){
							keepPlus = false;	
						}
						
						xRound2 = xRound +1;
						yRound2 = yRound;
						zRound2 = zRound;
						if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
						if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
						if (output[x-xRound2][y-yRound2][z-zRound2]!=0){
							keepMinus = false;
						}
						if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
						if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
						if (output[x+xRound2][y+yRound2][z+zRound2]!=0){
							keepPlus = false;	
						}
						
						xRound2 = xRound;
						yRound2 = yRound +1;
						zRound2 = zRound;
						if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
						if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
						if (output[x-xRound2][y-yRound2][z-zRound2]!=0){
							keepMinus = false;
						}
						if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
						if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
						if (output[x+xRound2][y+yRound2][z+zRound2]!=0){
							keepPlus = false;	
						}
						
						// z = 1
						xRound2 = xRound;
						yRound2 = yRound;
						zRound2 = zRound -1;
						if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
						if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
						if (output[x-xRound2][y-yRound2][z-zRound2]!=0){
							keepMinus = false;
						}
						if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
						if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
						if (output[x+xRound2][y+yRound2][z+zRound2]!=0){
							keepPlus = false;	
						}
						
						// end 6 neighborhood
						
						
						
					/*	
					// all floor = 0 0 0
					int xRound = (int)Math.floor(i*n[0]);
					int yRound = (int)Math.floor(i*n[1]);
					int zRound = (int)Math.floor(i*n[2]);		
					
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					
					// all floor = 0 1 0
					xRound = (int)Math.floor(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.floor(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}					
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					
					// all ceil = 1 1 1
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}					
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					//
					
					// all ceil = 1 0 1
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.floor(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}				
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					//
					
					// all ceil = 1 1 0
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.floor(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}						
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;
					}
					//
					
					// all ceil = 1 0 0
					xRound = (int)Math.ceil(i*n[0]);
					yRound = (int)Math.floor(i*n[1]);
					zRound = (int)Math.floor(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}					
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					//
					
					
					// x floor y floor z ceil = 0 0 1
					xRound = (int)Math.floor(i*n[0]);
					yRound = (int)Math.floor(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}					
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					//
					
					// x floor y ceil z ceil = 0 1 1
					xRound = (int)Math.floor(i*n[0]);
					yRound = (int)Math.ceil(i*n[1]);
					zRound = (int)Math.ceil(i*n[2]);							
					
					if (keepMinus)
					if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
					if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
					if (output[x-xRound][y-yRound][z-zRound]!=0){
						keepMinus = false;
					}					
												
					if (keepPlus)
					if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
					if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
					if (output[x+xRound][y+yRound][z+zRound]!=0){
						keepPlus = false;	
					}
					
					*/
					
				
					//
					/*
					keep = checkDistBorder(p1, p2Plus);
					if (keep)
						keep = checkDistBorder(p1, p2Minus);
					*/
					}
					
					if (!(keepMinus || keepPlus))
						break outerloop;
									
				}
				if (!(keepMinus || keepPlus))
					output[x][y][z] = 0;	
			}
			
			
		}
		
		
		
		if(!isAllInOne)		
			showZ("hyst strip 3");
		
		if (Const.GUI2_chkSteps)
			showZ("Step 6: Skull stripping 2");
		
		//showX("skull strip 3 X");
		//showY("skull strip 3 Y");
	}
	
	// return true if surface point p1 is nearest to the border
	private boolean checkDistBorder(double[] p1, double[][] p2){	
		
		double halfX = (nx-1)/2.5;
		double halfY = (ny-1)/3.0;
		double halfZ = (nz-1)/2.0;
		
		double x1 = p1[0]-halfX;
		double y1 = p1[1]-halfY;
		double z1 = p1[2]-halfZ;
		
		
		for (int i=0; i<8; i++)
		if (p2[i] != null){
			
			double x2 = p2[i][0]-halfX;			
			double y2 = p2[i][1]-halfY;			
			double z2 = p2[i][2]-halfZ;
			
			if (Math.sqrt(x1*x1 + y1*y1 + z1*z1) < Math.sqrt(x2*x2 + y2*y2 + z2*z2))
				return true;//continue;
			//else
				//return false;
		}	
		
		return false;	
	}
	
	
	public Sequence getSequenceZ(){
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
		return sequence;
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
