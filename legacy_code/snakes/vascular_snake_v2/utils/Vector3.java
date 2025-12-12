package plugins.big.vascular.utils;

public class Vector3 implements Comparable {


	private Integer x = null;
	private Integer y = null;
	private Integer z = null;
	
/*------------------------------------------------------------------*/
	public Vector3(int x, int y, int z) {
		this.x = new Integer(x);
		this.y = new Integer(y);
		this.z = new Integer(z);
	} // end Vector3
	
/*------------------------------------------------------------------*/
	public int compareTo (Object obj) {
	
		final Vector3 e = (Vector3) obj;
		
		final int zComparison = e.getZ().compareTo(z);
		if (zComparison != 0) {
			return(zComparison);
		}
		
		final int yComparison = e.getY().compareTo(y);
		if (yComparison != 0) {
			return(yComparison);
		}
		
		final int xComparison = e.getX().compareTo(x);
		return(xComparison);
	} // end compareTo
	
/*------------------------------------------------------------------*/
	public Integer getX() {
		return x;
	} // end getX
	
/*------------------------------------------------------------------*/
	public Integer getY() {
		return y;
	} // end getY
	
/*------------------------------------------------------------------*/
	public Integer getZ() {
		return z;
	} // end getZ

} // end class Vector3