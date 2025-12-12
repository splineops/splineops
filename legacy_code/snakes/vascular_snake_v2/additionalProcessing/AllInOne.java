package plugins.big.vascular.additionalProcessing;

import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import plugins.big.vascular.steerable3DFilter.SurfaceDetector;
import plugins.big.vascular.utils.BinaryThresholding;
import plugins.big.vascular.utils.ConnectedComponentsCustom;
import plugins.big.vascular.utils.Const;
import plugins.big.vascular.utils.HysteresisThresholding;
import plugins.big.vascular.utils.LocalNormalization;

public class AllInOne extends Plugin{

	private Sequence activeSeq;
	private boolean showSteps = false;
	
	// last values (as comments) are for MP2RAGE, the other values are for the ISBI paper (brainWeb data).
	
	private double LOCAL_MEAN = 80; //(40 -> PD, i.e. ISBI paper, 80 for MP2RAGE)
	private double LOCAL_VARIANCE = 0.1;
	private double STEERABLE_SIGMA = 2;
	private int HYST_T1 = 15;
	private int HYST_T2 = 45; //(75 -> PD, i.e. ISBI paper, 45 for MP2RAGE)
	
	private double[][][] input;
	
	final int nx;
	final int ny;
	final int nz;	
	 
	
	public AllInOne(boolean showSteps){
		
		this.activeSeq = getActiveSequence();
		
		nx = activeSeq.getSizeX();
		ny = activeSeq.getSizeY();
		nz = activeSeq.getSizeZ();		
		
		input = new double[nx][ny][nz];
		this.showSteps = showSteps;			
	}
	
	public void process(){
		
		long startTime = System.currentTimeMillis();
		
		// convert sequence to double[][][]
		
			
		for(int x=0 ; x<nx; x++)
		for(int y=0; y<ny; y++)
		for(int z=0; z<nz; z++)
		  	input[x][y][z] = activeSeq.getData(0, z, 0, y, x);
		
		// locNorm
		long startTimeLocNorm = System.currentTimeMillis();
		
		LocalNormalization localNormalization = new LocalNormalization(LOCAL_MEAN, LOCAL_VARIANCE, input);
		localNormalization.normalize();				
		
		long endTimeLocNorm = System.currentTimeMillis();
		long elapsedTimeMillis = endTimeLocNorm-startTimeLocNorm;
		
		float elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time locNorm = " + elapsedTimeSec);
		
		// steerable filter
		long startTimeSteerable = System.currentTimeMillis();
		
		// **************** **************************//
		// **********                      ***********//   
		// ** For Brainweb PD image the local normalization step can be skipped !!!
		// *******
		// *******
		// *******		
		//SurfaceDetector surfaceDetector = new SurfaceDetector(STEERABLE_SIGMA, input);
		// *******	
		// *******
		// *******
		SurfaceDetector surfaceDetector = new SurfaceDetector(STEERABLE_SIGMA, localNormalization.volume);		
		
		surfaceDetector.detect();
		
		Const.surfaceDetectorAllInOne = surfaceDetector;
		
		long endTimeSteerable = System.currentTimeMillis();
		elapsedTimeMillis = endTimeSteerable-startTimeSteerable;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time steerable = " + elapsedTimeSec);
		
		// hysteresis thresholding
		long startTimeHyst = System.currentTimeMillis();
		
		HysteresisThresholding hysteresisThresholding = new HysteresisThresholding(surfaceDetector.nmsResult, HYST_T1, HYST_T2);
		hysteresisThresholding.process();
		
		long endTimeHyst = System.currentTimeMillis();
		elapsedTimeMillis = endTimeHyst-startTimeHyst;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time hysteresis thresholding = " + elapsedTimeSec);
		
		// Hyst strip 1
		long startTimeHystStrip1 = System.currentTimeMillis();
		
		HystStrip2 hystStrip2 = new HystStrip2(hysteresisThresholding.output, surfaceDetector.surfaceDetection.getOrientation());
		hystStrip2.process();
		
		long endTimeHystStrip1 = System.currentTimeMillis();
		elapsedTimeMillis = endTimeHystStrip1-startTimeHystStrip1;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time hyst strip 1 = " + elapsedTimeSec);
		
		
		// Hyst strip 2
		long startTimeHystStrip2 = System.currentTimeMillis();
		
		HystStrip3 hystStrip3 = new HystStrip3(hystStrip2.output, surfaceDetector.surfaceDetection.getOrientation());
		hystStrip3.process();
		
		long endTimeHystStrip2 = System.currentTimeMillis();
		elapsedTimeMillis = endTimeHystStrip2-startTimeHystStrip2;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time hyst strip 2 = " + elapsedTimeSec);
		
		
		// Connected components
		long startTimeCC= System.currentTimeMillis();
		
		ConnectedComponentsCustom cc = new ConnectedComponentsCustom(hystStrip3.getSequenceZ());
		cc.process();
		
		long endTimeCC= System.currentTimeMillis();
		elapsedTimeMillis = endTimeCC-startTimeCC;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time conncected components = " + elapsedTimeSec);
		
		
		// binary thresholding
		long startTimeBinary= System.currentTimeMillis();
		
		BinaryThresholding binaryThresholding = new BinaryThresholding(cc.getSequenceProcessed(), cc.getLabelLargestComponent());
		binaryThresholding.process();
		
		Const.binaryThresholdingOutput = binaryThresholding.getOutputArray();
		
		long endTimeBinary= System.currentTimeMillis();
		elapsedTimeMillis = endTimeBinary-startTimeBinary;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time binary thresholding = " + elapsedTimeSec);
		
		// total time
		long endTime = System.currentTimeMillis();
		elapsedTimeMillis = endTime-startTime;
		elapsedTimeSec = elapsedTimeMillis/1000F;
		System.out.println("time total = " + elapsedTimeSec);		
	}
}
