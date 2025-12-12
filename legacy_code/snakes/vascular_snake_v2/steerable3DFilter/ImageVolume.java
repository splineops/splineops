package plugins.big.vascular.steerable3DFilter;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.system.IcyExceptionHandler;
import icy.type.DataType;


import java.io.*;
import java.text.*;


public class ImageVolume extends Plugin{


	private int nx;
	private int ny;
	private int nz;
	public double[][][] volume;
	
	public byte[][][] byteVolume;
	private double min;
	private double max;
	
	
	public ImageVolume(int nx, int ny, int nz) {
	
		this.nx = nx;
		this.ny = ny;
		this.nz = nz;
		volume = new double[nx][ny][nz];	
	}
	
	public ImageVolume(Sequence sequence) {
		
		nx = sequence.getSizeX();
		ny = sequence.getSizeY();
		nz = sequence.getSizeZ();		
		
		volume = new double[nx][ny][nz];	
		
		for(int x=0 ; x<nx; x++)
	    for(int y=0; y<ny; y++)
	    for(int z=0; z<nz; z++)
	     	volume[x][y][z] = sequence.getData(0, z, 0, y, x);
	}
	
	public ImageVolume(Sequence sequence, boolean isByte) {
		
		nx = sequence.getSizeX();
		ny = sequence.getSizeY();
		nz = sequence.getSizeZ();		
		
		byteVolume = new byte[nx][ny][nz];	
		
		for(int x=0 ; x<nx; x++)
	    for(int y=0; y<ny; y++)
	    for(int z=0; z<nz; z++)
	     	byteVolume[x][y][z] = (byte) sequence.getData(0, z, 0, y, x);
	}
	
	public ImageVolume (boolean setActiveSequence){
		Sequence sequence = getFocusedSequence();
		
		nx = sequence.getSizeX();
		ny = sequence.getSizeY();
		nz = sequence.getSizeZ();		
		
		volume = new double[nx][ny][nz];	
		
		for(int x=0 ; x<nx; x++)
	    for(int y=0; y<ny; y++)
	    for(int z=0; z<nz; z++)
	     	volume[x][y][z] = sequence.getData(0, z, 0, y, x);
	}
	
	
	/*
	public ImageVolume(ImagePlus imp) {
		this.nx = imp.getWidth();
		this.ny = imp.getHeight();
		this.nz = imp.getStackSize();
		volume = new double[nx][ny][nz];
		Object[] stack = imp.getStack().getImageArray();
		
		if (imp.getProcessor() instanceof FloatProcessor) {
			for (int z=0;z<nz;z++) {
		      	for (int y=0;y<ny;y++) {
		         	for (int x=0;x<nx;x++) {
		          	volume[x][y][z] = (double)(((float[])stack[z])[x+nx*y]);
		          }
		        }
			}		
		} else if (imp.getProcessor() instanceof ShortProcessor) {
			for (int z=0;z<nz;z++) {
		      	for (int y=0;y<ny;y++) {
		         	for (int x=0;x<nx;x++) {
		          	volume[x][y][z] = (double)(((short[])stack[z])[x+nx*y] & 0xFFFF);
		          }
		        }
			}	
		} else if (imp.getProcessor() instanceof ByteProcessor) {
			for (int z=0;z<nz;z++) {
		      	for (int y=0;y<ny;y++) {
		         	for (int x=0;x<nx;x++) {
		          	volume[x][y][z] = (double)(((byte[])stack[z])[x+nx*y] & 0xFF);
		          }
		        }
			}	
		} else {
			IJ.write("32-bit, 16-bit, or 8-bit input required");
	    throw new ArrayStoreException ("Unexpected image type");
		}
	}
	*/
	

	public ImageVolume(double[][][] data) {
		this.nx = data.length;
		this.ny = data[0].length;
		this.nz = data[0][0].length;
		volume = new double[nx][ny][nz];
		for (int z=0;z<nz;z++) 
		for (int y=0;y<ny;y++) 
       	for (int x=0;x<nx;x++) 
       		volume[x][y][z] = data[x][y][z];   		
	}
	
	
	public int getSizeX() {
		return nx;
	}
	public int getSizeY() {
		return ny;
	}
	public int getSizeZ() {
		return nz;
	}
	
	public double[][][] getVolumeArray() {
		return volume;
	}	
	
	
	public void showZ(String title) {		
		Sequence sequence = new Sequence(); 
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double current;
			
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {
					current  = volume[i][j][k];
					temp[i+j*nx] = current;
					if (current < min) 
						min = current;
					
					if (current > max) 
						max = current;					
				}
			}    
           
			tempImage.dataChanged();
			sequence.addImage(tempImage);      
    	}                
		sequence.setName(title);
		addSequence(sequence);		
	}
	
	
	public void showY(String title) {		
		Sequence sequence = new Sequence(); 			
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double current;
			
		//for (int y=ny-1;y>=0;y--) {
		for (int y=0; y<ny; y++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, nz, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=0;z<nz;z++) {
				for (int x=0;x<nx;x++) {
					current  = volume[x][y][z];
					temp[x+z*nx] = current;
					if (current < min) 
						min = current;
					
					if (current > max) 
						max = current;					
				}
			}  	
			tempImage.dataChanged();
			sequence.addImage(tempImage);
		}
		sequence.setName(title);
		addSequence(sequence);		
	}
	
	
	public void showX(String title) {		
		Sequence sequence = new Sequence(); 
		
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double current;
			
		//for (int x=nx-1;x>=0;x--) {
		for (int x=0; x<nx; x++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(ny, nz, 1, DataType.DOUBLE);		
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int z=nz-1;z>=0;z--) {
				for (int y=ny-1;y>=0;y--) {        
					current  = volume[x][ny-y-1][z];	        
					temp[y+z*ny] = current;
					if (current < min) 
						min = current;
					
					if (current > max) 
						max = current;					
				}
			}         	
			tempImage.dataChanged();
			sequence.addImage(tempImage);
		}
		sequence.setName(title);     
		addSequence(sequence);		
	}


	public double getPixel(int x, int y, int z) {
		return volume[x][y][z];
	}
	

	public void putPixel(int x, int y, int z, double value) {
		volume[x][y][z] = value;
	}	


	public double[] getRowX(int y, int z) {
		double[] row = new double[nx];
		for(int i=0;i<nx;i++) {
	  		row[i] = volume[i][y][z];
	  }	
		return row;	  		
	}
		
		
	public double[] getRowY(int x, int z){
		double[] row = new double[ny];
		for(int i=0;i<ny;i++) {
			row[i] = volume[x][i][z];
		}	
		return row;	  		
	}


	public double[] getRowZ(int x, int y) {
		double[] row = new double[nz];
		for(int i=0;i<nz;i++) {
			row[i] = volume[x][y][i];
		}	
		return row;	  		
	}


	public void putRowX(int y, int z, double[] row) {
		for(int i=0;i<nx;i++) {
			volume[i][y][z] = row[i];
		}
	}


	public void putRowY(int x, int z, double[] row) {
		for(int i=0;i<ny;i++) {
			volume[x][i][z] = row[i];
		}	 		
	}


	public void putRowZ(int x, int y, double[] row) {
		for(int i=0;i<nz;i++) {
			volume[x][y][i] = row[i];
		}	
	}
	
	public double getMin() {
		return min;
	}
	
	public double getMax() {
		return max;
	}
	
	public void cstMin(double c) {
		computeMinMax();
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					volume[x][y][z] = volume[x][y][z] - min + c;
				}
			}
		}
	}				
	
	public void computeMinMax() {
		min = Double.MAX_VALUE;
		max = Double.MIN_VALUE;
		
		double temp;
	
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					temp = volume[x][y][z];
					if (temp > max) {
						max = temp;
					}	
					if (temp < min) {
						min = temp;
					}
				}
			}
		}
	
	}	
	
	
	public void rescale() {
	
		computeMinMax();
		
		double f;
		
		if (min - max == 0.0) {
			f = 1.0;
		} else {
			f = 255.0 / (max - min);
		}
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					volume[x][y][z] = f * (volume[x][y][z]-min);
				}
			}
		}
		
	} 
	
	
	public ImageVolume scaled() {
	
		double min = Double.MAX_VALUE;
		double max = Double.MIN_VALUE;
		double temp;
		ImageVolume out = new ImageVolume(nx, ny, nz);
	
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					temp = volume[x][y][z];
					if (temp > max) {
						max = temp;
					} else if (temp < min) {
						min = temp;
					}
				}
			}
		}
		
		double f;
		
		if (min - max == 0.0) {
			f = 1.0;
		} else {
			f = 255.0 / (max - min);
		}
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					out.putPixel(nx, ny, nz, f * (volume[x][y][z]-min));
				}
			}
		}
					
		return out; 	
	} 



	public void outputAsText(String filename) {
	
		try {
			FileWriter fwriter = new FileWriter(filename);
			BufferedWriter writer = new BufferedWriter(fwriter);
			writer.write(nx+"\n");
			writer.write(ny+"\n");
			writer.write(nz+"\n");
			DecimalFormat df = new DecimalFormat("#0.0000000000\n");
			for (int z=0;z<nz;z++) {
				for (int y=0;y<ny;y++) {
					for (int x=0;x<nx;x++) {
						writer.write(df.format(volume[x][y][z])); 
					}
				}
			}		
			writer.close();
			fwriter.close();
		} catch (IOException e) {
			IcyExceptionHandler.showErrorMessage(null, true);  //IJ.write("IOException: " + e);
		} 	
	
	}
	
	
	public double getInterpolatedPixel(double x, double y, double z) {	

		
		if (x < 0) {
	    int periodx = 2*nx - 2;				
			while (x < 0) x += periodx;		// Periodize
			if (x >= nx)  x = periodx - x;	// Symmetrize
		}
		else if ( x >= nx) {
	    int periodx = 2*nx - 2;				
			while (x >= nx) x -= periodx;	// Periodize
			if (x < 0)  x = - x;			// Symmetrize
		}

		if (y < 0) {
	    int periody = 2*ny - 2;				
			while (y < 0) y += periody;		// Periodize
			if (y >= ny)  y = periody - y;	// Symmetrize
		}
		else if (y >= ny) {
	    int periody = 2*ny - 2;				
			while (y >= ny) y -= periody;	// Periodize
			if (y < 0)  y = - y;			// Symmetrize
		}
		
		if (z < 0) {
	    int periodz = 2*nz - 2;				
			while (z < 0) z += periodz;		// Periodize
			if (z >= nz)  z = periodz - z;	// Symmetrize
		}
		else if (z >= nz) {
	    int periodz = 2*nz - 2;				
			while (z >= nz) z -= periodz;	// Periodize
			if (z < 0)  z = - z;			// Symmetrize
		}	
		
		
		int i = (int)x;
		int j = (int)y;
		int k = (int)z;		
		
		double dx = x - (double)((int)x);
		double dy = y - (double)((int)y);
		double dz = z - (double)((int)z);
		
		
	  int di, dj, dk;
		
		if(i >= nx-1) { // border mirror condition
			di = -1;
		}	else {
			di = 1;
		}
		
		if(j >= ny-1) { // border mirror condition
			dj = -1;
		}	else {
			dj = 1;
		}	
		
		if(k >= nz-1) { // border mirror condition
			dk = -1;
		}	else {
			dk = 1;
		}	
		
		
		double v000 = volume[i][j][k];
		double v100 = volume[i+di][j][k];
		double v010 = volume[i][j+dj][k];
		double v110 = volume[i+di][j+dj][k];
		
		double v001 = volume[i][j][k+dk];
		double v101 = volume[i+di][j][k+dk];
		double v011 = volume[i][j+dj][k+dk];
		double v111 = volume[i+di][j+dj][k+dk];
		

		double interpolation = ((v000*(1.0-dx) + v100*dx)*(1-dy) + (v010*(1.0-dx) + v110*dx)*dy)*(1-dz)
												 + ((v001*(1.0-dx) + v101*dx)*(1-dy) + (v011*(1.0-dx) + v111*dx)*dy)*dz;
		
		
		return interpolation;
	}
	
	public void setVolume(double[][][] data) {
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					volume[x][y][z] = data [x][y][z];
				}
			}
		}
	}				
	
	public void rotateX() { // turn the volume 90 around x axis
		double[][][] newVolume = new double[nx][ny][nz];
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					newVolume[x][z][y] = volume[x][y][z];
				}
			}
		}			 
		setVolume(newVolume);
	}
	
	
	public void rotateY() { // turn the volume 90 around y axis
		double[][][] newVolume = new double[nx][ny][nz];
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					newVolume[z][y][x] = volume[x][y][z];
				}
			}
		}			 
		setVolume(newVolume);
	}
	
	
	public void invert() {
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				for (int x=0;x<nx;x++) {
					volume[x][y][z] = -volume[x][y][z];
				}
			}
		}			 
	}
	
	
	public ImageVolume copy() {
		ImageVolume out = new ImageVolume(volume);
		return out;
	}
	
	public static ImageVolume linearCombination(double a, ImageVolume A, double b, ImageVolume B) {
		int nx = A.getSizeX();
		int ny = A.getSizeY();
		int nz = A.getSizeZ();
		ImageVolume result = new ImageVolume(nx,ny,nz);
		for (int x=0;x<nx;x++) {
			for (int y=0;y<ny;y++) {	
				for (int z=0;z<nz;z++) {
					result.putPixel(x,y,z, a*A.getPixel(x,y,z) + b*B.getPixel(x,y,z));
				}
			}
		}
		return result;
	}
	
	public void approxZero(double precision) {
		for (int x=0;x<nx;x++) {
			for (int y=0;y<ny;y++) {
				for (int z=0;z<nz;z++) {
					if (volume[x][y][z] < precision) {
						volume[x][y][z] = 0.0;
					}
				}		
			}
		}
	}
	
	public void addConstant(double c) {
		for (int x=0;x<nx;x++) {
			for (int y=0;y<ny;y++) {
				for (int z=0;z<nz;z++) {
					volume[x][y][z] += c;
				}		
			}
		}
	}	
	
	public void init() {
		for (int x=0;x<nx;x++) {
			for (int y=0;y<ny;y++) {
				for (int z=0;z<nz;z++) {
					volume[x][y][z] = 0.0;
				}		
			}
		}
	}
	

} // ImageVolume

