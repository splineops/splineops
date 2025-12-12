package plugins.big.vascular.additionalProcessing;

import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;

public class DistanceBorder extends Plugin{
	
	private int nx, ny, nz;
	private double[][][] result;
	
	/*
	 * distanceBorder
	 *
	public void distanceBorder(Sequence in, int ni){
		int nx = in.getWidth();
		int ny = in.getHeight();
		result = new double[nx][ny][nz];
		
		for (int x=0; x<nx; x++)
		for (int y=0; y<ny; y++)
		for (int z=0; z<nz; z++){
			if (in.getData(0, z, 0, y, x) > 0)
				result[x][y][z] = ni+2;
			else
				result[x][y][z] = ni+1;
		}			
		return fillBorder(in, result, 1, ni); 		
	}
	
	
	/*
	 * fillBorder
	 *
	private void fillBorder(Sequence input, double[][][] result, int value, int ni){
		double[][][] out = dilate(input, SQUARE, 3);
		input.subtract(out, input);
		
		int nx = input.getWidth();
		int ny = input.getHeight();
		
		for (int x=0; x<nx; x++)
		for (int y=0; y<ny; y++)
		if (input.getPixel(x, y) > 0)
			result.putPixel(x, y, value);
			
		if (value + 1 <= ni)
			return fillBorder(out, result, value+1, ni);	
		
		return result;
	}

	*/
}
