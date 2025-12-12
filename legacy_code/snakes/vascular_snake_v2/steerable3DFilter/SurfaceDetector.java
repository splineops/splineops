package plugins.big.vascular.steerable3DFilter;

import javax.swing.JTextArea;

import plugins.big.vascular.utils.Const;
import additionaluserinterface.WalkBar;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;


public class SurfaceDetector extends Plugin{
	
	private WalkBar walk = null;
	private JTextArea logWindow = null;
	private boolean showInterfaceTools;
	
	private double[][][] origImg = null;	
	private double sigma = 0;	
	
	public FilterResponse surfaceDetection;
	private ImageVolume response;
	public ImageVolume nmsResult;
	
	public SurfaceDetector(double sigma, WalkBar walk, JTextArea logWindow){
		this.walk = walk;
		this.logWindow = logWindow;
		this.sigma = sigma;	
		
		showInterfaceTools = true;
	}
	
	public SurfaceDetector(double sigma, double[][][] input){		
		this.sigma = sigma;	
		origImg = input;
		
		showInterfaceTools = false;
	}
	
	public void detect(){	
		
		if (showInterfaceTools){
			logWindow.append("\nstart surface detection");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());		
		
			logWindow.append("\ncreate image buffer");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		if (showInterfaceTools)
			origImg = getOrigImg();
		
		
		// Image dimensions
		final int nx = origImg.length;
		final int ny = (origImg[0]).length;
		final int nz = (origImg[0][0]).length;
		
		// Surface Detection		
		double[][][] SurfDetectResult = new double[nx][ny][nz];		
		
		ImageVolume origImgVol = new ImageVolume(origImg);		
		
		// detect positive surfaces
		surfaceDetection = new Steerable3D().detector3D(logWindow, walk, origImgVol, sigma, 4.0, 1.0); // 1.0: positive, -1.0 negative		
		
		// step detector
		//surfaceDetection = new Steerable3D().detector3DStep(logWindow, walk, origImgVol, sigma);
			
		response = surfaceDetection.getResponse();	
		
		Const.surfaceDetection = surfaceDetection;
		
		nmsResult = new Steerable3D().surfaceNMS(surfaceDetection);
		SurfDetectResult = nmsResult.getVolumeArray();				
		
		if (showInterfaceTools){
			logWindow.append("\nsurface detection done.");
			logWindow.setCaretPosition(logWindow.getDocument().getLength());
		}
		
		if (Const.GUI2_chkSteps){
			showZFilter("Step 2: steerable filter response, sigma = " + sigma);
			showZNMS("Step 3: non-maximum suppresion");
		}
			
	}
	
	public void showXNMS(){
		nmsResult.showX("response X NMS");		
	}
	
	public void showYNMS(){
		nmsResult.showY("response Y NMS");
	}
	
	public void showZNMS(){
		nmsResult.showZ("response Z NMS");		
		Const.NMS_Z = "response Z NMS";
	}
	
	public void showZNMS(String title){
		nmsResult.showZ(title);	
	}
	
	public void showXFilter(){
		response.showX("response X");
	}
	
	public void showYFilter(){
		response.showY("response Y");
	}
	
	public void showZFilter(){
		response.showZ(Const.SURFACE_Z);
	}
	
	public void showZFilter(String title){
		response.showZ(title);
	}
	
	/**
	* Process the image.
	*/
	public double[][][] getOrigImg() {		
		
		if (showInterfaceTools)
			walk.reset();
		
		Sequence sequence = getFocusedSequence();
		
		if (sequence == null) 		
			return null;					
		
		final int nx = sequence.getSizeX();
		final int ny = sequence.getSizeY();
		final int nz = sequence.getSizeZ();
		
		double [][][] originalImage = new double[nx][ny][nz];			
		
		for (int z=0; z<nz; z++) {	
			
			if (showInterfaceTools)
				walk.progress("create image buffer.. ", (100.0*z)/nz);
			
			for (int y=0; y<ny; y++) {
		    	for (int x=0; x<nx; x++) {			    		
		    		originalImage[x][y][z] = sequence.getData(0, z, 0, y, x);
		    	}
			}
		}	
		
		if (showInterfaceTools)
			walk.finish("creating image buffer done.");
		
		return originalImage;		
	}
}
