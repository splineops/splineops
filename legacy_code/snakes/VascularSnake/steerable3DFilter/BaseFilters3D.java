package plugins.big.vascular.steerable3DFilter;

import javax.swing.JTextArea;

import icy.gui.dialog.MessageDialog;


public class BaseFilters3D implements Runnable{
	
	private JTextArea logWindow;
	
	// surface detector
	static public int GXX = 0;
	static public int GYY = 1;
	static public int GZZ = 2;
	static public int GXY = 3;
	static public int GXZ = 4;
	static public int GYZ = 5;
	
	// step detector
	static public int GX = 6;
	static public int GY = 7;
	static public int GZ = 8;
	
	private int derivative ;
	private ImageVolume input;
	private ImageVolume output;
	double sigma;
	
	public BaseFilters3D(JTextArea logWindow, ImageVolume input, double sigma, int derivative){
		this.logWindow = logWindow;
		this.input = input;
		this.sigma = sigma;
		this.derivative = derivative;
	}



	public static ImageVolume fConvolvedG(ImageVolume input, double sigma) {
	
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double t;
		double sigma2 = sigma*sigma;
		double sqrt2pi = Math.sqrt(2.0*Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);		
		
		xComponent[wWidth] = 1.0/(sqrt2pi*sigma);
		yComponent[wWidth] = 1.0/(sqrt2pi*sigma);
		zComponent[wWidth] = 1.0/(sqrt2pi*sigma);
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			t = Math.exp(-(w*w)/(2.0*sigma2))/(sqrt2pi*sigma);
			xComponent[i] = t;
			xComponent[window-1-i] = t;		
			yComponent[i] = t;
			yComponent[window-1-i] = t;
			zComponent[i] = t;
			zComponent[window-1-i] = t;
		}
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}		
		
		return out;
	}
	
	
	
	private ImageVolume fConvolvedGxx() {
		if (logWindow != null){		
			logWindow.append("\nTHREAD 1: calculating gxx");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
	
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);		
		
		xComponent[wWidth] = -1.0 / (sqrt2pi*sigma2*sigma2*sigma);
		yComponent[wWidth] = 1.0;
		zComponent[wWidth] = 1.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = (w*w - sigma2) * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma7);
			xComponent[window-1-i] = xComponent[i];		
			yComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			yComponent[window-1-i] = yComponent[i];
			zComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = zComponent[i];
		}
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}		
				
		if (logWindow != null){
			logWindow.append("\nTHREAD 1: calculating gxx done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	private ImageVolume fConvolvedGyy() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 2: calculating gyy");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);		
		
		xComponent[wWidth] = 1.0;
		yComponent[wWidth] = -1.0 / (sqrt2pi*sigma2*sigma2*sigma);
		zComponent[wWidth] = 1.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			xComponent[window-1-i] = xComponent[i];		
			yComponent[i] = (w*w - sigma2) * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma7);
			yComponent[window-1-i] = yComponent[i];
			zComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = zComponent[i];
		}	
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 2: calculating gyy done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	private ImageVolume fConvolvedGzz() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 3: calculating gzz");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);		
		
		xComponent[wWidth] = 1.0;
		yComponent[wWidth] = 1.0;
		zComponent[wWidth] = -1.0 / (sqrt2pi*sigma2*sigma2*sigma);
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			xComponent[window-1-i] = xComponent[i];
			yComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			yComponent[window-1-i] = yComponent[i];
			zComponent[i] = (w*w - sigma2) * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma7);
			zComponent[window-1-i] = zComponent[i];	
		}	
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 3: calculating gzz done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	private ImageVolume fConvolvedGxy() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 4: calculating gxy");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);
		
		
		xComponent[wWidth] = 0.0;
		yComponent[wWidth] = 0.0;
		zComponent[wWidth] = 1.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = w * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma7);
			xComponent[window-1-i] = -xComponent[i];		
			yComponent[i] = w * Math.exp(-(w*w)/(2.0*sigma2));
			yComponent[window-1-i] = -yComponent[i];
			zComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = zComponent[i]; // symmetry
		}
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 4: calculating gxy done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	
	private ImageVolume fConvolvedGxz() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 5: calculating gxz");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
	
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);
		
		
		xComponent[wWidth] = 0.0;
		yComponent[wWidth] = 1.0;
		zComponent[wWidth] = 0.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = w * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma7);
			xComponent[window-1-i] = -xComponent[i];		
			yComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			yComponent[window-1-i] = yComponent[i]; // symmetry
			zComponent[i] = w * Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = -zComponent[i];
		}			
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}		
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 5: calculating gxz done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	
	private ImageVolume fConvolvedGyz() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 6: calculating gyz");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
	
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);
		
		
		xComponent[wWidth] = 1.0;
		yComponent[wWidth] = 0.0;
		zComponent[wWidth] = 0.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {		
			xComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			xComponent[window-1-i] = xComponent[i]; // symmetry
			yComponent[i] = w * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma7);
			yComponent[window-1-i] = -yComponent[i];
			zComponent[i] = w * Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = -zComponent[i];
		}			
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD 6: calculating gyz done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	/*
	 * Gx
	 */
	private ImageVolume fConvolvedGx() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD: calculating gx");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		//double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);
		
		
		xComponent[wWidth] = 0.0;
		yComponent[wWidth] = 1.0;
		zComponent[wWidth] = 1.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = -w * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma2*sigma2*sigma);
			xComponent[window-1-i] = -xComponent[i];		
			yComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			yComponent[window-1-i] = -yComponent[i];
			zComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = zComponent[i]; // symmetry
		}
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD: calculating gx done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	/*
	 * Gy
	 */
	private ImageVolume fConvolvedGy() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD: calculating gy");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		//double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);
		
		
		xComponent[wWidth] = 1.0;
		yComponent[wWidth] = 0.0;
		zComponent[wWidth] = 1.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			xComponent[window-1-i] = -xComponent[i];		
			yComponent[i] = -w * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma2*sigma2*sigma);
			yComponent[window-1-i] = -yComponent[i];
			zComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			zComponent[window-1-i] = zComponent[i]; // symmetry
		}
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD: calculating gy done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	/*
	 * Gz
	 */
	private ImageVolume fConvolvedGz() {
		
		if (logWindow != null){
			logWindow.append("\nTHREAD: calculating gz");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		int nx = input.getSizeX();
		int ny = input.getSizeY();
		int nz = input.getSizeZ();
		
		int wWidth = (int)(4.0*sigma);
		int window = 2*wWidth+1;

		double xComponent[] = new double[window];
		double yComponent[] = new double[window];
		double zComponent[] = new double[window];
		double row[];
		double sigma2 = sigma*sigma;
		//double sigma7 = sigma2*sigma2*sigma2*sigma;
		double sqrt2pi = 2.0*Math.sqrt(2.0)*Math.PI*Math.sqrt(Math.PI);
		
		Convolver convolver = new Convolver(Convolver.MIRROR);
		ImageVolume out = new ImageVolume(nx, ny, nz);
		
		
		xComponent[wWidth] = 1.0;
		yComponent[wWidth] = 1.0;
		zComponent[wWidth] = 0.0;
		
		for (int i=wWidth+1,w=1;i<window;i++,w++) {
			xComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			xComponent[window-1-i] = -xComponent[i];		
			yComponent[i] = Math.exp(-(w*w)/(2.0*sigma2));
			yComponent[window-1-i] = -yComponent[i];
			zComponent[i] = -w * Math.exp(-(w*w)/(2.0*sigma2)) / (sqrt2pi*sigma2*sigma2*sigma);
			zComponent[window-1-i] = zComponent[i]; // symmetry
		}
		
		
		for (int z=0;z<nz;z++) {
			for (int y=0;y<ny;y++) {
				row = input.getRowX(y, z);
				convolver.convolveFIR(row, xComponent, wWidth);
				out.putRowX(y, z, row);
			}
		}
		
		for (int z=0;z<nz;z++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowY(x, z);
				convolver.convolveFIR(row, yComponent, wWidth);
				out.putRowY(x, z, row);
			}
		}
		
		for (int y=0;y<ny;y++) {
			for (int x=0;x<nx;x++) {
				row = out.getRowZ(x, y);
				convolver.convolveFIR(row, zComponent, wWidth);
				out.putRowZ(x, y, row);
			}
		}	
		
		if (logWindow != null){
			logWindow.append("\nTHREAD: calculating gz done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		return out;
	}
	
	
	/*
	 * 
	 */
	public ImageVolume getOutput(){
		return output;
	}



	@Override
	public void run() {
		switch (derivative){
			case 0: output = fConvolvedGxx();
				break;
			case 1: output = fConvolvedGyy();
				break;
			case 2: output = fConvolvedGzz();
				break;
			case 3: output = fConvolvedGxy();
				break;
			case 4: output = fConvolvedGxz();
				break;
			case 5: output = fConvolvedGyz();
				break;
			case 6: output = fConvolvedGx();
				break;
			case 7: output = fConvolvedGy();
				break;
			case 8: output = fConvolvedGz();
				break;
			
			default:MessageDialog.showDialog("Error computing Gaussian derivatives!");
				break;
		}
		
	}
	
		

}	