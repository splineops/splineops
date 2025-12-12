package plugins.big.vascular.additionalProcessing;

import icy.image.IcyBufferedImage;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.type.DataType;
import plugins.big.vascular.utils.ConnectedComponentsCustom;

public class BinarizeDistMap extends Plugin{
	
	Sequence distMap; 	
	int nx, ny, nz;	
	
	public BinarizeDistMap(){
		distMap = getActiveSequence();
		
		nx = distMap.getSizeX();
		ny = distMap.getSizeY();
		nz = distMap.getSizeZ();
	}
	
	public void process(){
		
		Sequence sequence = new Sequence(); 
		double current;
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);//.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {							
					current = distMap.getData(0, k, 0, j, i);	
					
					if (current > 30)
						current = 1;
					else 
						current = 0;
					
					temp[i+j*nx] = current;										
				}
			}             
			tempImage.dataChanged();
			sequence.addImage(tempImage);      
    	}
		
		
		sequence.setName("Distance Transform");
		addSequence(sequence);	
		
		ConnectedComponentsCustom cc = new ConnectedComponentsCustom(sequence);
		cc.process();
		
		addSequence(cc.getSequenceProcessed());
		System.out.println("largest Comp = " + cc.getLabelLargestComponent());
		
		int largestComponent = cc.getLabelLargestComponent();
		Sequence seqCC = cc.getSequenceProcessed();
		
		Sequence sequence2 = new Sequence(); 
		
		for (int k=0; k<nz; k++) {
			IcyBufferedImage tempImage  = new IcyBufferedImage(nx, ny, 1, DataType.DOUBLE);
			double[] temp = tempImage.getDataXYAsDouble(0);//.getDataXYAsDouble(0);
			for (int j=0; j<ny; j++) {
				for (int i=0; i<nx; i++) {							
					current = seqCC.getData(0, k, 0, j, i);	
					
					if (current == largestComponent)
						current = 1;
					else 
						current = 0;
					
					temp[i+j*nx] = current;										
				}
			}             
			tempImage.dataChanged();
			sequence2.addImage(tempImage); 			
    	}
		addSequence(sequence2);
	}

}
