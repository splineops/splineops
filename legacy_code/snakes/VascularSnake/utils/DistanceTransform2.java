/* Bob Dougherty 8/8/2006
Saito-Toriwaki algorithm for Euclidian Distance Transformation.
Direct application of Algorithm 1.
Version S1A: lower memory usage.
Version S1A.1 A fixed indexing bug for 666-bin data set
Version S1A.2 Aug. 9, 2006.  Changed noResult value.
Version S1B Aug. 9, 2006.  Faster.
Version S1B.1 Sept. 6, 2006.  Changed comments.
Version S1C Oct. 1, 2006.  Option for inverse case.
                           Fixed inverse behavior in y and z directions.
Version D July 30, 2007.  Multithread processing for step 2.

This version assumes the input stack is already in memory, 8-bit, and
outputs to a new 32-bit stack.  Versions that are more stingy with memory
may be forthcoming.

 License:
	Copyright (c) 2006, OptiNav, Inc.
	All rights reserved.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions
	are met:

		Redistributions of source code must retain the above copyright
	notice, this list of conditions and the following disclaimer.
		Redistributions in binary form must reproduce the above copyright
	notice, this list of conditions and the following disclaimer in the
	documentation and/or other materials provided with the distribution.
		Neither the name of OptiNav, Inc. nor the names of its contributors
	may be used to endorse or promote products derived from this software
	without specific prior written permission.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
	"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
	LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
	A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
	CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
	EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
	PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
	PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
	LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
	NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/


package plugins.big.vascular.utils;


import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import ij.IJ;
import ij.ImagePlus;
import ij.ImageStack;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import imageware.Builder;
import imageware.ImageWare;


public class DistanceTransform2 extends Plugin{
	
	double FACTOR = 1;
	int BORDER = 1;
	
	String TITLE = "Distance Transform 2";
	//private ImagePlus imp;
	private Sequence seq;
	
	public byte[][] data;
	public int w,h,d;
	public int thresh = 1; // background value
	public boolean inverse = false;

	/*
	public int setup(String arg, ImagePlus imp) {
 		this.imp = imp;
		return DOES_8G;
	}
	*/
	public DistanceTransform2(){
		this.seq = getActiveSequence();
		w = seq.getSizeX();
		h = seq.getSizeY();
		d = seq.getSizeZ();
	}
	
	
	public void run() {
		
		//ImageStack stack = imp.getStack();
				
		
		int nThreads = Runtime.getRuntime().availableProcessors();

		//if(!getScale())return;		
		
		ImageWare imW = Builder.create(w, h, d, ImageWare.BYTE);			
		
		for (int z=0; z<d; z++)
		for (int y=0; y<h; y++)
		for (int x=0; x<w; x++){
			double val = seq.getData(0, z, 0, y, x);			
			
			if (val > 0)				
				imW.putPixel(x, y, z, 0);
			else
				imW.putPixel(x, y, z, 255);			
		}
		ImageStack stack = imW.buildImageStack();
		//ImageProcessor ip = (new ImagePlus("", stack)).getProcessor();
		
		//Create references to input data
		data = new byte[d][];
		for (int k = 0; k < d; k++)data[k] = (byte[])stack.getPixels(k+1);
		//Create 32 bit floating point stack for output, s.  Will also use it for g in Transormation 1.
		ImageStack sStack = new ImageStack(w,h);
		float[][] s = new float[d][];
		for(int k = 0; k < d; k++){
			ImageProcessor ipk = new FloatProcessor(w,h);
			sStack.addSlice(null,ipk);
			s[k] = (float[])ipk.getPixels();
		}
		float[] sk;
		//Transformation 1.  Use s to store g.
		//IJ.showStatus("EDT transformation 1/3");
		System.out.println("EDT transformation 1/3");
		
		Step1Thread[] s1t = new Step1Thread[nThreads];
		for(int thread = 0; thread < nThreads; thread++){
			s1t[thread] = new Step1Thread(thread,nThreads,w,h,d,thresh,s,data);
			s1t[thread].start();
		}
		try{
			for(int thread = 0; thread< nThreads; thread++){
				s1t[thread].join();
			}
		}catch(InterruptedException ie){
			//IJ.error("A thread was interrupted in step 1 .");
			System.out.println("A thread was interrupted in step 1 .");
		}		
		//Transformation 2.  g (in s) -> h (in s)
		System.out.println("EDT transformation 2/3");
		Step2Thread[] s2t = new Step2Thread[nThreads];
		for(int thread = 0; thread < nThreads; thread++){
			s2t[thread] = new Step2Thread(thread,nThreads,w,h,d,s);
			s2t[thread].start();
		}
		try{
			for(int thread = 0; thread< nThreads; thread++){
				s2t[thread].join();
			}
		}catch(InterruptedException ie){
			System.out.println("A thread was interrupted in step 2 .");
		}
		//Transformation 3. h (in s) -> s
		System.out.println("EDT transformation 3/3");
		Step3Thread[] s3t = new Step3Thread[nThreads];
		for(int thread = 0; thread < nThreads; thread++){
			s3t[thread] = new Step3Thread(thread,nThreads,w,h,d,s,data);
			s3t[thread].start();
		}
		try{
			for(int thread = 0; thread< nThreads; thread++){
				s3t[thread].join();
			}
		}catch(InterruptedException ie){
			System.out.println("A thread was interrupted in step 3 .");
		}		
		//Find the largest distance for scaling
		//Also fill in the background values.
		float distMax = 0;
		int wh = w*h;
		float dist;
		for(int k = 0; k < d; k++){
			sk = s[k];
			for(int ind = 0; ind < wh; ind++){
				if(((data[k][ind]&255) < thresh)^inverse){
					sk[ind] = 0;
				}else{
					dist = (float)Math.sqrt(sk[ind]);
					sk[ind] = dist;
					distMax = (dist > distMax) ? dist : distMax;
				}
			}
		}

		//IJ.showProgress(1.0);
		System.out.println("Done");
		String title = TITLE; //stripExtension(imp.getTitle());
		ImagePlus impOut = new ImagePlus(title+"EDT",sStack);
		impOut.getProcessor().setMinAndMax(0,distMax);
		//impOut.show();
		show(impOut);
		System.out.println("Fire");
	}
	
	
	//Modified from ImageJ code by Wayne Rasband
    String stripExtension(String name) {
        if (name!=null) {
            int dotIndex = name.lastIndexOf(".");
            if (dotIndex>=0)
                name = name.substring(0, dotIndex);
		}
		return name;
    }
    
    private void show(ImagePlus imp){
    	ImageWare imW = Builder.create(imp, ImageWare.FLOAT);		
		
		Sequence sequence = new Sequence(); 		
		double current;
			
		for (int k=0; k<d; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(w, h, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);
			for (int j=0; j<h; j++) {
				for (int i=0; i<w; i++) {
					current  = imW.getPixel(i, j, k);
					temp[i+j*w] = current*FACTOR;										
				}
			}             
			tempImage.dataChanged();
			sequence.addImage(tempImage);      
    	}  
		sequence.setName("Distance Transform");
		addSequence(sequence);
		
		
		
		// modify
		/*
		double val = 0;
		double max = 0;
		for (int x=0; x<w; x++)
		for (int y=0; y<h; y++)
		for (int z=0; z<d; z++){
			val = sequence.getData(0, z, 0, y, x);
			if (val > max)
				max = val;
		}
		
		Sequence sequence2 = new Sequence();
		
		for (int k=0; k<d; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(w, h, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);//.getDataXYAsDouble(0);
			for (int j=0; j<h; j++) {
				for (int i=0; i<w; i++) {					
					
					//current = sequence.getData(0, k, 0, j, i);			
					
					current = (max - sequence.getData(0, k, 0, j, i));					
					
					/*
					if (current == max)
						current = BORDER;	
					*/
					
					/*
					temp[i+j*w] = current;										
				}
			}             
			tempImage.dataChanged();
			sequence2.addImage(tempImage);      
    	}
		
		
		sequence2.setName("Distance Transform 2");
		addSequence(sequence2);		
		*/
		
    }
    
    /*
	boolean getScale() {
		thresh = (int)Prefs.get("edtS1.thresh", 128);
		inverse = Prefs.get("edtS1.inverse", false);
		GenericDialog gd = new GenericDialog("EDT...", IJ.getInstance());
		gd.addNumericField("Threshold (1 to 255; value < thresh is background)", thresh, 0);
       	gd.addCheckbox("Inverse case (background when value >= thresh)",inverse);
		gd.showDialog();
		if (gd.wasCanceled())return false;
		thresh = (int)gd.getNextNumber();
      	inverse = gd.getNextBoolean();
		Prefs.set("edtS1.thresh", thresh);
		Prefs.set("edtS1.inverse", inverse);
		return true;
	}
	*/
	
	
	class Step1Thread extends Thread{
		int thread,nThreads,w,h,d,thresh;
		float[][] s; 
		byte[][] data;
		public Step1Thread(int thread, int nThreads, int w, int h, int d, int thresh, float[][] s,  byte[][] data){
			this.thread = thread;
			this.nThreads = nThreads;
			this.w = w;
			this.h = h;
			this.d = d;
			this.thresh = thresh;
			this.data = data;
			this.s = s;
		}
		public void run(){
			float[] sk;
			byte[] dk;
			int n = w;
			if(h > n) n = h;
			if(d > n) n = d;
			int noResult = 3*(n+1)*(n+1);
			boolean[] background = new boolean[n];			
			boolean nonempty;
			int test, min;			
			for(int k = thread; k < d; k+=nThreads){
				IJ.showProgress(k/(1.*d));
				sk = s[k];
				dk = data[k];
				for(int j = 0; j < h; j++){
					for (int i = 0; i < w; i++){
						background[i] = ((dk[i+w*j]&255) < thresh)^inverse;
					}
					for (int i = 0; i < w; i++){
						min = noResult;
						for (int x = i; x < w; x++){
							if(background[x]){
								test = i - x;
								test *= test;
								min = test;
								break;
							}
						}
						for (int x = i-1; x >=0 ; x--){
							if(background[x]){
								test = i - x;
								test *= test;
								if(test < min)min = test;
								break;
							}
						}
						sk[i+w*j] = min;
					}
				}
			}
		}//run
	}//Step1Thread
	class Step2Thread extends Thread{
		int thread,nThreads,w,h,d;
		float[][] s; 
		public Step2Thread(int thread, int nThreads, int w, int h, int d, float[][] s){
			this.thread = thread;
			this.nThreads = nThreads;
			this.w = w;
			this.h = h;
			this.d = d;
			this.s = s;
		}
		public void run(){
			float[] sk;
			int n = w;
			if(h > n) n = h;
			if(d > n) n = d;
			int noResult = 3*(n+1)*(n+1);
			int[] tempInt = new int[n];
			int[] tempS = new int[n];
			boolean nonempty;
			int test, min, delta;
			for(int k = thread; k < d; k+=nThreads){
				IJ.showProgress(k/(1.*d));
				sk = s[k];
				for (int i = 0; i < w; i++){
					nonempty = false;
					for (int j = 0; j < h; j++){
						tempS[j] = (int)sk[i+w*j];
						if(tempS[j] >0)nonempty = true;
					}
					if(nonempty){
						for (int j = 0; j < h; j++){
							min = noResult;
							delta = j;
							for(int y = 0; y < h; y++){
								test = tempS[y] + delta*delta--;
								if(test < min)min = test;
							}
							tempInt[j] = min;
						}
						for (int j = 0; j < h; j++){
							sk[i+w*j] = tempInt[j];
						}
					}
				}
			}
		}//run
	}//Step2Thread	
	class Step3Thread extends Thread{
		int thread,nThreads,w,h,d;
		float[][] s; 
		byte[][] data;
		public Step3Thread(int thread, int nThreads, int w, int h, int d, float[][] s, byte[][] data){
			this.thread = thread;
			this.nThreads = nThreads;
			this.w = w;
			this.h = h;
			this.d = d;
			this.s = s;
			this.data = data;
		}
		public void run(){
			int zStart,zStop,zBegin,zEnd;
			float[] sk;
			int n = w;
			if(h > n) n = h;
			if(d > n) n = d;
			int noResult = 3*(n+1)*(n+1);
			int[] tempInt = new int[n];
			int[] tempS = new int[n];
			boolean nonempty;
			int test, min, delta;
			for(int j = thread; j < h; j+=nThreads){
				IJ.showProgress(j/(1.*h));
				for(int i = 0; i < w; i++){
					nonempty = false;
					for(int k = 0; k < d; k++){
						tempS[k] = (int)s[k][i+w*j];
						if(tempS[k] >0)nonempty = true;
					}
					if(nonempty){
						zStart = 0;
						while((zStart < (d-1))&&(tempS[zStart] == 0))zStart++;
						if(zStart > 0)zStart--;
						zStop = d-1;
						while((zStop > 0)&&(tempS[zStop] == 0))zStop--;
						if(zStop < (d-1))zStop++;
	
						for(int k = 0; k < d; k++){
							//Limit to the non-background to save time,
							if(((data[k][i+w*j]&255) >= thresh)^inverse){
								min = noResult;
								zBegin = zStart;
								zEnd = zStop;
								if(zBegin > k)zBegin = k;
								if(zEnd < k)zEnd = k;
								delta = k - zBegin;
								for (int z = zBegin; z <= zEnd; z++){
									test = tempS[z] + delta*delta--;
									if(test < min)min = test;
									//min = (test < min) ? test : min;
								}
								tempInt[k] = min;
							}
						}
						for(int k = 0; k < d; k++){
							s[k][i+w*j] = tempInt[k];
						}
					}
				}
			}
		}//run
	}//Step2Thread	
}