
/** This class implements an affine coordinate mapping */

package plugins.big.vascular.additionalProcessing;

public class AffineCoors extends CoordinateMapper { 

	public double s[][] = new double[3][4];
	private String name = "";

	public AffineCoors( double [][] matrix ) { 
		this(matrix, CoordinateMapper.UNKNOWN);
	}
	public AffineCoors( double [][] matrix, int type) { 
		this(matrix, type, "");
	}
	
	public AffineCoors( double [][] matrix, int type, String name ) { 
		this.coorType = type;
		this.name = name;
		for (int i=0; i<3; i++) { 
			for (int j=0; j<4; j++) { 
				s[i][j] = matrix[i][j];
			}
		}
	}
	
	public String getName() { 
		return name; 
	}

	public double [][] getMatrix() { return s; }

	public CoordinateMapper copy() { 
		return new AffineCoors( s, coorType, name ); 
	}

	public double getX( double x, double y, double z) { 
		return ((s[0][0]*x) + (s[0][1]*y) +  (s[0][2]*z) +  s[0][3]); 
	}
	public double getY( double x, double y, double z) { 
		return ((s[1][0]*x) + (s[1][1]*y) +  (s[1][2]*z) +  s[1][3]); 
	}
	public double getZ( double x, double y, double z) { 
		return ((s[2][0]*x) + (s[2][1]*y) +  (s[2][2]*z) +  s[2][3]); 
	}
	
	public void flipResultX() { 
		for (int i=0; i<4; i++) s[0][i] = -s[0][i];
	}
	public void flipResultY() { 
		for (int i=0; i<4; i++) s[1][i] = -s[1][i];
	}
	public void flipResultZ() { 
		for (int i=0; i<4; i++) s[2][i] = -s[2][i];
	}

	public void rotate( int [] axes, double [] dims ) { 

		double [][] postMatrix = new double[3][4];
		double [][] newS = new double[3][4];

		for (int i=0; i<3; i++) { 
			int absAxis = Math.abs( axes[i] ) -1;
			postMatrix[absAxis][i] = (axes[i]>0) ? 1 : -1;
			if (axes[i]<0) postMatrix[absAxis][3] = dims[absAxis]-1;
		}
		for (int i=0; i<3; i++) { 
			for (int j=0; j<4; j++) { 
				for (int k=0; k<3; k++) 
					newS[i][j] += s[i][k] * postMatrix[k][j];
			}
			newS[i][3] += s[i][3];	
		}
		s = newS;
	}
}	
