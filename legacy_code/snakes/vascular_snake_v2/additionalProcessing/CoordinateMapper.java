
/** Abstract class to represent a mapping into a co-ordinate space. 
 * Implementing classes are QuaternCoors & AffineCoors which
 * map using quaterns and affine transformations respectively
*/

package plugins.big.vascular.additionalProcessing;

public abstract class CoordinateMapper { 

	public static final int UNKNOWN = 0;
	public static final int NIFTI = 1;
	public static final int DICOM = 2;

	protected int coorType = 1; 

	public int getCoorType() { 
		return coorType;
	}

	public String getXDescription() { 
		switch (coorType) { 
			case NIFTI: return "left to right"; 
			case DICOM: return "right to left"; 
			default: return "unknown";
		}
	}

	public String getYDescription() { 
		switch (coorType) { 
			case NIFTI: return "posterior to anterior"; 
			case DICOM: return "anterior to posterior"; 
			default: return "unknown";
		}
	}

	public String getZDescription() { 
		switch (coorType) { 
			case NIFTI:  
			case DICOM: return "inferior to superior"; 
			default: return "unknown";
		}
	}

	/* The difference between DICOM & Nifti coordinates is a 
	 * flip in the x & y axes */

	public boolean convertToType( int newType ) { 
		if (coorType==newType) return true; 	
		if ( ( (coorType==NIFTI) && (newType==DICOM) ) ||
			( (coorType==DICOM) && (newType==NIFTI) ) ) { 
			
			flipResultX();
			flipResultY();
			coorType = newType; 
			return true;	
		}
		return false;	
	}

	public abstract String getName(); 
	
	public abstract CoordinateMapper copy();

	public abstract double getX( double x, double y, double z );
	public abstract double getY( double x, double y, double z );
	public abstract double getZ( double x, double y, double z );

	public abstract void flipResultX();
	public abstract void flipResultY();
	public abstract void flipResultZ();

	/** Rotates/reflects the coordinate system, which is useful
	 * after a flip or rotation in the original image. This is
	 * only implemented for 90 degree rotations and reflections
	 * parallel to the axes.
	 * 
	 * See Rotate plugin for syntax. */
	public abstract void rotate( int [] axes, double [] dims );

	public double getX(int x, int y, int z) { 
		return getX( (double) x, (double) y, (double) z ); 
	}
	public double getY(int x, int y, int z) { 
		return getY( (double) x, (double) y, (double) z ); 
	}
	public double getZ(int x, int y, int z) { 
		return getZ( (double) x, (double) y, (double) z ); 
	}

	public double [] transform( double x, double y, double z ) { 
		return new double [] { 
			getX( x, y, z), 
			getY( x, y, z), 
			getZ( x, y, z) };  
	}
	
	public double [] transform( double [] coors ) { 
		return transform( coors[0], coors[1], coors[2] ); 
	}
	public double [] transform( int [] coors ) { 
		return transform( (double) coors[0], (double) coors[1], (double) coors[2] ); 
	}
	public double [] transform( int x, int y, int z ) { 
		return transform( (double) x, (double) y, (double) z ); 
	}
}

