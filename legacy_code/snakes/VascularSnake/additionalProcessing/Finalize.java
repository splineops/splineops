package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.main.Icy;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.steerable3DFilter.ImageVolume;
import plugins.big.vascular.utils.Const;

public class Finalize extends Plugin{	
	private double[][][][] orientation;
	
	
	private Sequence activeSeq; // contour image
	private Sequence maskSeq;
	private Sequence origSeq;
	
	private int THRESHOLD_ORIG = 2500;
	
	private byte[][][] output;	
	private int nx, ny, nz;	
	
	
	public Finalize(){
		maskSeq = Icy.getMainInterface().getSequences("Binary mask").get(0);
		activeSeq = Icy.getMainInterface().getSequences("Step 8: binary image, t = 8152").get(0);
		origSeq = Icy.getMainInterface().getSequences("DR_timeRes_1_orig").get(0);
		
	}
	
	public void process(){
		//activeSeq = getFocusedSequence(); // contour image
		
		nx = activeSeq.getSizeX();
		ny = activeSeq.getSizeY();
		nz = activeSeq.getSizeZ();
		
		orientation = Const.surfaceDetection.getOrientation();
		
		output = (new ImageVolume(maskSeq, true)).byteVolume;
		//showZ("output");
		
		double[] n = new double[3];
		double val = 0;
		
		orientation = Const.surfaceDetection.getOrientation();		
				
		int DIST = 30;
		
		for (int x=1;x<nx-1;x++) 
		for (int y=1;y<ny-1;y++) 
		for (int z=1;z<nz-1;z++) {
			
			val = activeSeq.getData(0, z, 0, y, x);			
			if (val > 0){
				
				n[0] = orientation[x][y][z][0];
				n[1] = orientation[x][y][z][1];
				n[2] = orientation[x][y][z][2];	
				
				put(DIST, x, y, z,  0, 0, 0, n);				
				
				
				// z=-1
				put(DIST, x, y, z,  -1, -1, -1, n);
				put(DIST, x, y, z,  0, -1, -1, n);
				put(DIST, x, y, z,  1, -1, -1, n);
				
				put(DIST, x, y, z,  -1, 0, -1, n);
				put(DIST, x, y, z,  0, 0, -1, n);
				put(DIST, x, y, z,  1, 0, -1, n);
				
				put(DIST, x, y, z,  -1, 1, -1, n);
				put(DIST, x, y, z,  0, 1, -1, n);
				put(DIST, x, y, z,  1, 1, -1, n);
				
				// z = 0
				put(DIST, x, y, z,  -1, -1, 0, n);
				put(DIST, x, y, z,  0, -1, 0, n);
				put(DIST, x, y, z,  1, -1, 0, n);
				
				put(DIST, x, y, z,  -1, 0, 0, n);
				put(DIST, x, y, z,  0, 0, 0, n);
				put(DIST, x, y, z,  1, 0, 0, n);
				
				put(DIST, x, y, z,  -1, 1, 0, n);
				put(DIST, x, y, z,  0, 1, 0, n);
				put(DIST, x, y, z,  1, 1, 0, n);
				
				// z = 1
				put(DIST, x, y, z,  -1, -1, 1, n);
				put(DIST, x, y, z,  0, -1, 1, n);
				put(DIST, x, y, z,  1, -1, 1, n);
				
				put(DIST, x, y, z,  -1, 0, 1, n);
				put(DIST, x, y, z,  0, 0, 1, n);
				put(DIST, x, y, z,  1, 0, 1, n);
				
				put(DIST, x, y, z,  -1, 1, 1, n);
				put(DIST, x, y, z,  0, 1, 1, n);
				put(DIST, x, y, z,  1, 1, 1, n);						
				
			}
		}
		
		showZ("mask finalized");
	}
	
	
	private void put(int dist, int x, int y, int z, int xc, int yc, int zc, double[] n){
		outerloop:
			for (int i=0; i<dist; i++){						
						
									
			int xRound = (int)Math.round(i*n[0]);
			int yRound = (int)Math.round(i*n[1]);
			int zRound = (int)Math.round(i*n[2]);
				
			
			x = x+xc;
			y = y+yc;
			z = z+zc;
				// z=1
				int xRound2 = xRound;// + xc;
				int yRound2 = yRound;// + yc;
				int zRound2 = zRound;// + zc;
				
				if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
				if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
				if (output[x-xRound2][y-yRound2][z-zRound2]!=0){						
					for (int j=i; j >= 0; j--){						
						
						xRound2 = (int)Math.round(j*n[0]);
						yRound2 = (int)Math.round(j*n[1]);
						zRound2 = (int)Math.round(j*n[2]);
						
						putMinus(x, y, z, xRound2, yRound2, zRound2);										
					}
					break outerloop;
				}
				
				if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
				if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
				if (output[x+xRound2][y+yRound2][z+zRound2]!=0){						
						
					for (int j=i; j >= 0; j--){								
						
						xRound2 = (int)Math.round(j*n[0]);
						yRound2 = (int)Math.round(j*n[1]);
						zRound2 = (int)Math.round(j*n[2]);
							
						putPlus(x, y, z, xRound2, yRound2, zRound2);									
					}
					break outerloop;
				}
			}
		
	}
	
	private boolean checkThreshold(int x, int y, int z){
		//return true;
		
		if (origSeq.getData(0, z, 0, y, x) < THRESHOLD_ORIG)
			return true;
		return false;
		
	}
	
	// pubPlus
	private void putPlus(int x, int y, int z, int xRound, int yRound, int zRound){	
		
		if (z+zRound >= 0  && y+yRound >= 0 && x+xRound >= 0)
		if (z+zRound < nz && y+yRound < ny && x+xRound < nx)
		if (checkThreshold(x+xRound, y+yRound, z+zRound))	
			output[x+xRound][y+yRound][z+zRound] = 127;
		
		// ***************************** z=-1  *****************************		
		
		
		// top row, y=-1
		int xRound2 = xRound -1;
		int yRound2 = yRound -1;
		int zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))	
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound -1;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))	
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound -1;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))			
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// middle row, y=0
		xRound2 = xRound -1;
		yRound2 = yRound;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))	
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))	
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// bottom row, y=1
		xRound2 = xRound -1;
		yRound2 = yRound +1;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound +1;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound +1;
		zRound2 = zRound -1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// ***************************** z=0  *****************************
		// top row, y=-1
		xRound2 = xRound -1;
		yRound2 = yRound -1;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound -1;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound -1;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// middle row, y=0
		xRound2 = xRound -1;
		yRound2 = yRound;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
								
		xRound2 = xRound +1;
		yRound2 = yRound;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// bottom row, y=1
		xRound2 = xRound -1;
		yRound2 = yRound +1;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound +1;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound +1;
		zRound2 = zRound;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		
		// ***************************** z=1  *****************************
		// top row, y=-1
		xRound2 = xRound -1;
		yRound2 = yRound -1;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound -1;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound -1;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// middle row, y=0
		xRound2 = xRound -1;
		yRound2 = yRound;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// bottom row, y=1
		xRound2 = xRound -1;
		yRound2 = yRound +1;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound;
		yRound2 = yRound +1;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		xRound2 = xRound +1;
		yRound2 = yRound +1;
		zRound2 = zRound +1;							
		if (z+zRound2 >= 0  && y+yRound2 >= 0 && x+xRound2 >= 0)
		if (z+zRound2 < nz && y+yRound2 < ny && x+xRound2 < nx)
		if (checkThreshold(x+xRound2, y+yRound2, z+zRound2))
			output[x+xRound2][y+yRound2][z+zRound2] = 127;
		
		// end 26 neighborhood
	}
	//
	
	
	// putMinus
	private void putMinus(int x, int y, int z, int xRound, int yRound, int zRound){
		
		if (z-zRound >= 0  && y-yRound >= 0 && x-xRound >= 0)
		if (z-zRound < nz && y-yRound < ny && x-xRound < nx)
		if (checkThreshold(x-xRound, y-yRound, z-zRound))	
			output[x-xRound][y-yRound][z-zRound] = 127;
			
			// ***************************** z=-1  *****************************
			
			
			// top row, y=-1
			int xRound2 = xRound -1;
			int yRound2 = yRound -1;
			int zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))	
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound -1;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound -1;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// middle row, y=0
			xRound2 = xRound -1;
			yRound2 = yRound;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// bottom row, y=1
			xRound2 = xRound -1;
			yRound2 = yRound +1;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound +1;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound +1;
			zRound2 = zRound -1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// ***************************** z=0  *****************************
			// top row, y=-1
			xRound2 = xRound -1;
			yRound2 = yRound -1;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound -1;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound -1;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// middle row, y=0
			xRound2 = xRound -1;
			yRound2 = yRound;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
									
			xRound2 = xRound +1;
			yRound2 = yRound;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// bottom row, y=1
			xRound2 = xRound -1;
			yRound2 = yRound +1;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound +1;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound +1;
			zRound2 = zRound;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			
			// ***************************** z=1  *****************************
			// top row, y=-1
			xRound2 = xRound -1;
			yRound2 = yRound -1;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound -1;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound -1;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// middle row, y=0
			xRound2 = xRound -1;
			yRound2 = yRound;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// bottom row, y=1
			xRound2 = xRound -1;
			yRound2 = yRound +1;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound;
			yRound2 = yRound +1;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			xRound2 = xRound +1;
			yRound2 = yRound +1;
			zRound2 = zRound +1;							
			if (z-zRound2 >= 0  && y-yRound2 >= 0 && x-xRound2 >= 0)
			if (z-zRound2 < nz && y-yRound2 < ny && x-xRound2 < nx)
			if (checkThreshold(x-xRound2, y-yRound2, z-zRound2))
				output[x-xRound2][y-yRound2][z-zRound2] = 127;
			
			// end 26 neighborhood
	}
	//
	
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
