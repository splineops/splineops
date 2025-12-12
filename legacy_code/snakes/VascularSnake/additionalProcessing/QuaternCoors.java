
/** This class implements a Quatern coordinate mapping */
package plugins.big.vascular.additionalProcessing;

public class QuaternCoors extends CoordinateMapper { 

	public double s[][] = new double[3][4];
	public double [] quaterns = new double[5]; 
	private double [] pixdim = new double[3]; 
	private double [] qoffsets = new double[3]; 
	private String name = "";

	public QuaternCoors( double [] q, double [] pixdim, double [] qoffsets ) { 
		this(q, pixdim, qoffsets, CoordinateMapper.UNKNOWN);
	}
	public QuaternCoors( double [] q, double [] pixdim, double [] qoffsets, int type ) { 
		this(q, pixdim, qoffsets, type, "");
	}
	
	public QuaternCoors( double [] q, double [] pixdim, double [] qoffsets, int type, String space ) { 
		this.coorType = type; 
		this.quaterns = q;
		this.pixdim = pixdim; 
		this.qoffsets = qoffsets; 
		this.name = space;
		s = getOrientation( quaterns, pixdim, qoffsets );	
	}
	
	public String getName() { 
		return name; 
	}
	
	public CoordinateMapper copy() { 
		return new QuaternCoors( quaterns, pixdim, qoffsets, coorType, name );
	}

	public void flipResultX() {
		double [] newq = new double[5];
		newq[0] = -quaterns[0];
		newq[1] = quaterns[3];
		newq[2] = quaterns[4];
		newq[3] = quaterns[1];
		newq[4] = quaterns[2];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
		qoffsets[0] = -qoffsets[0];
		this.quaterns = newq;
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void flipResultY() {
		double [] newq = new double[5];
		newq[0] = -quaterns[0];
		newq[1] = quaterns[2];
		newq[2] = quaterns[1];
		newq[3] = -quaterns[4];
		newq[4] = -quaterns[3];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
		qoffsets[1] = -qoffsets[1];
		this.quaterns = newq;
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void flipResultZ() {
		double [] newq = new double[5];
		newq[0] = -quaterns[0];
		newq[1] = quaterns[1];
		newq[2] = -quaterns[2];
		newq[3] = -quaterns[3];
		newq[4] = quaterns[4];
		qoffsets[2] = -qoffsets[2];
		this.quaterns = newq;
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void flipX( double[] dims) {
		double [] newq = new double[5];
		/* newq[0] = -quaterns[0];
		newq[1] = -quaterns[3];
		newq[2] = quaterns[4];
		newq[3] = -quaterns[1];
		newq[4] = -quaterns[2];
		*/
		newq[0] = -quaterns[0];
		newq[1] = -quaterns[3];
		newq[2] = -quaterns[4];
		newq[3] = quaterns[1];
		newq[4] = quaterns[2];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
		this.quaterns = newq;
		for (int i=0; i<3; i++) qoffsets[i] += s[i][0] * (dims[0]-1);
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void flipY( double[] dims) {
		double [] newq = new double[5];
		/*newq[0] = -quaterns[0];
		newq[1] = -quaterns[2];
		newq[2] = -quaterns[1];
		newq[3] = -quaterns[4];
		newq[4] = quaterns[3];
		*/
		newq[0] = -quaterns[0];
		newq[1] = -quaterns[2];
		newq[2] = quaterns[1];
		newq[3] = quaterns[4];
		newq[4] = -quaterns[3];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
		this.quaterns = newq;
		for (int i=0; i<3; i++) qoffsets[i] += s[i][1] * (dims[1]-1);
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void flipZ( double[] dims) {
		quaterns[0] *= -1.0;
		for (int i=0; i<3; i++) qoffsets[i] += s[i][2] * (dims[2]-1);
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void swapXY()  { 
		double fac = Math.sqrt( 0.5 );
		double [] newq = new double[5];
		newq[1] = -fac * ( quaterns[2] + quaterns[3] );
		newq[2] = fac * ( quaterns[1] - quaterns[4] );
		newq[3] = fac * ( quaterns[1] + quaterns[4] );
		newq[4] = fac * ( quaterns[2] - quaterns[3] );
		newq[0] = -quaterns[0];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
		/* newq[0] = -quaterns[0];
		newq[1] = fac * ( quaterns[2] + quaterns[3] );
		newq[2] = fac * ( quaterns[4] - quaterns[1] );
		newq[3] = -fac * ( quaterns[1] + quaterns[4] );
		newq[4] = fac * ( quaterns[3] - quaterns[2] );
		*/
		/* double tmp = qoffsets[1];
		qoffsets[1] = qoffsets[0];
		qoffsets[0] = tmp; */ 
		double tmp = pixdim[1];
		pixdim[1] = pixdim[0];
		pixdim[0] = tmp;
		this.quaterns = newq;
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void swapXZ()  { 
		double fac = Math.sqrt( 0.5 );
		double [] newq = new double[5];
		/*newq[0] = -quaterns[0];
		newq[1] = fac * ( quaterns[1] + quaterns[3] );
		newq[2] = fac * ( quaterns[2] + quaterns[4] );
		newq[3] = fac * ( quaterns[3] - quaterns[1] );
		newq[4] = fac * ( quaterns[4] - quaterns[2] );
		*/
		newq[1] = fac * ((quaterns[0]<0.0) ? quaterns[1] - quaterns[3] : quaterns[1] + quaterns[3]);
		newq[2] = fac * ((quaterns[0]<0.0) ? quaterns[2] - quaterns[4] : quaterns[2] + quaterns[4]);
		newq[3] = fac * ((quaterns[0]<0.0) ? quaterns[3] + quaterns[1] : quaterns[3] - quaterns[1]);
		newq[4] = fac * ((quaterns[0]<0.0) ? quaterns[4] + quaterns[2] : quaterns[4] - quaterns[2]);
		newq[0] = -quaterns[0];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
/*		double tmp = qoffsets[2];
		qoffsets[2] = qoffsets[0];
		qoffsets[0] = tmp; */ 
		double tmp = pixdim[2];
		pixdim[2] = pixdim[0];
		pixdim[0] = tmp;
		this.quaterns = newq;
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void swapYZ()  { 
		double fac = Math.sqrt( 0.5 );
		double [] newq = new double[5];
		/*newq[0] = -quaterns[0];
		newq[1] = fac * ( quaterns[1] - quaterns[2] );
		newq[2] = fac * ( quaterns[1] + quaterns[2] );
		newq[3] = fac * ( quaterns[3] + quaterns[4] );
		newq[4] = fac * ( quaterns[4] - quaterns[3] );
		*/
		newq[1] = fac * ((quaterns[0]>0.0) ? quaterns[1] - quaterns[2] : quaterns[1] + quaterns[2]);
		newq[2] = fac * ((quaterns[0]>0.0) ? quaterns[2] + quaterns[1] : quaterns[2] - quaterns[1]);
		newq[3] = fac * ((quaterns[0]>0.0) ? quaterns[3] + quaterns[4] : quaterns[3] - quaterns[4]);
		newq[4] = fac * ((quaterns[0]>0.0) ? quaterns[4] - quaterns[3] : quaterns[4] - quaterns[3]);
		newq[0] = -quaterns[0];
		if (newq[1]<0.0) { 
			for (int i=1; i<5; i++) { newq[i] = -newq[i]; }
		}
/*		double tmp = qoffsets[2];
		qoffsets[2] = qoffsets[1];
		qoffsets[1] = tmp; */ 
		double tmp = pixdim[2];
		pixdim[2] = pixdim[1];
		pixdim[1] = tmp;
		this.quaterns = newq;
		s = getOrientation( quaterns, pixdim, qoffsets );
	}
	public void rotate(int [] in_axes, double [] dims ) {  
		
		int [] axes = new int[3]; // Do we need the inverse ??

/*		for (int i=0; i<3; i++) { 
			int absAxis = Math.abs(in_axes[i]) - 1;
			axes[ absAxis ] = (in_axes[i]>0) ? i+1 : -i-1;
		}
*/
		axes = in_axes;  
		int [] interim = new int[3];
		double [] ndims = new double[3];
		for (int i=0; i<3; i++) { 
			interim[i] = i+1;
			ndims[i] = dims[i];
		}
		double tmp;
		int itmp;
		
		switch(Math.abs(axes[0])) { 
			case 2: 
				swapXY();
				tmp = ndims[0]; ndims[0] = ndims[1];
				ndims[1] = tmp; 
				interim[0] = 2;
				interim[1] = 1;
				break;
			case 3:
				swapXZ();
				tmp = ndims[0]; ndims[0] = ndims[2];
				ndims[2] = tmp; 
				interim[0] = 3;
				interim[2] = 1;
				break;
		}
		if (Math.abs(axes[1])!=interim[1]) { 
			swapYZ();
			itmp = interim[1];
			interim[1] = interim[2];
			interim[2] = itmp;
			tmp = ndims[1]; ndims[1] = ndims[2];
			ndims[2] = tmp;
		}

		if (axes[0]<0) flipX( ndims );
		if (axes[1]<0) flipY( ndims );
		if (axes[2]<0) flipZ( ndims );
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

	private double [][] getOrientation(double [] q, double [] p, double [] o) { 
	
		double [][] orientation = new double[3][4]; 
		
		double quatern_b = q[2];
		double quatern_c = q[3];
		double quatern_d = q[4];

		double q_a =1.0 - (quatern_b*quatern_b) - (quatern_c*quatern_c) - (quatern_d*quatern_d); 

		if (q_a <= 0.0) { 
			q_a = 0.0; 
		} else { 
			q_a = Math.sqrt( q_a ); 
		}
		this.quaterns[1] = q_a;
		orientation[0][0] = (q_a*q_a)+(quatern_b*quatern_b)-(quatern_c*quatern_c)-(quatern_d*quatern_d); 
		orientation[0][1] = (2.0*quatern_b*quatern_c)-(2.0*q_a*quatern_d);
		orientation[0][2] = (2.0*quatern_b*quatern_d)+(2.0*q_a*quatern_c);
		orientation[1][0] = (2.0*quatern_b*quatern_c)+(2.0*q_a*quatern_d);
		orientation[1][1] = (q_a*q_a)+(quatern_c*quatern_c)-(quatern_b*quatern_b)-(quatern_d*quatern_d);
		orientation[1][2] = (2.0*quatern_c*quatern_d)-(2.0*q_a*quatern_b);
		orientation[2][0] = (2.0*quatern_b*quatern_d)-(2.0*q_a*quatern_c);
		orientation[2][1] = (2.0*quatern_c*quatern_d)+(2.0*q_a*quatern_b);
		orientation[2][2] = (q_a*q_a)+(quatern_d*quatern_d)-(quatern_c*quatern_c)-(quatern_b*quatern_b);
		
		int qfac = (q[0] >= 0.0) ? 1 : -1; 

		if (qfac == -1) { 
			orientation[0][2] *= -1; 
			orientation[1][2] *= -1; 
			orientation[2][2] *= -1; 
		}
		
		double [][] flip = new double[3][4]; 
		for (int i=0; i<3; i++) { 
			for (int j=0; j<3; j++) { 
				flip[i][j] = orientation[i][j] * p[j];
			}
		}
		
		flip[0][3] = o[0];
		flip[1][3] = o[1];
		flip[2][3] = o[2];
		
		return flip;
	}

	public double [] getQuaterns() { 
		return quaterns;
	}

	public static double[] getQuaterns(double [][] inp) { 
	
		double [][] o = new double[3][3];

		for (int i=0; i<3; i++) { 
			for (int j=0; j<3; j++) { 
				o[i][j] = inp[i][j];
			}
		}

		double [] quaterns = new double[5];
		double [] ol = new double [3];
		double qfac = 1.0, a, b, c, d; 
		
		double det = (o[0][0]*o[1][1]*o[2][2])-(o[0][0]*o[2][1]*o[1][2]) - 
				(o[1][0]*o[0][1]*o[2][2])+(o[1][0]*o[2][1]*o[0][2]) + 
				(o[2][0]*o[0][1]*o[1][2])-(o[2][0]*o[1][1]*o[0][2]);

		
		if (det < 0.0) { 
			o[0][2] = -o[0][2]; o[1][2] = -o[1][2];	o[2][2] = -o[2][2];	
			qfac = -1.0;	
		}
	
		a = o[0][0] + o[1][1] + o[2][2] + 1.0;
		if (a > 0.5) { 
			a = 0.5 * Math.sqrt(a); 
			b = 0.25 * (o[2][1] - o[1][2]) / a;
			c = 0.25 * (o[0][2] - o[2][0]) / a;
			d = 0.25 * (o[1][0] - o[0][1]) / a;
		} else { 
			double xd = 1.0 + o[0][0] - o[1][1] - o[2][2]; 
			double yd = 1.0 + o[1][1] - o[0][0] - o[2][2]; 
			double zd = 1.0 + o[2][2] - o[0][0] - o[1][1]; 
			if (xd > 1.0) { 
				b = 0.5 * Math.sqrt(xd);
				c = 0.25 * (o[0][1]+o[1][0]) / b;
				d = 0.25 * (o[0][2]+o[2][1]) / b; 
				a = 0.25 * (o[2][1]-o[1][2]) / b; 
			} else if (yd > 1.0 ) { 
				c = 0.5 * Math.sqrt( yd ); 
				b = 0.25 * (o[0][1]+o[1][0]) / c;
				d = 0.25 * (o[1][2]+o[2][1]) / c; 
				a = 0.25 * (o[0][2]-o[2][0]) / c; 
			} else { 
				d = 0.5 * Math.sqrt( zd ); 
				b = 0.25 * (o[0][2]+o[2][0]) / d; 
				c = 0.25 * (o[1][2]+o[2][1]) / d;
				a = 0.25 * (o[1][0]-o[0][1]) / d;
			}
			if (a < 0.0) { 
				b = -b; 
				c = -c; 
				d = -d;
			}
		}

		quaterns[0] = qfac;
		quaterns[1] = a;
		quaterns[2] = b; 
		quaterns[3] = c; 
		quaterns[4] = d;

		return quaterns; 
	}
	
}

