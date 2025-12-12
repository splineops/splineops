package plugins.big.vascular.utils;

import icy.image.IcyBufferedImage;
import icy.image.colormap.FireColorMap;
import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import icy.sequence.VolumetricImage;
import icy.type.DataType;
import icy.type.collection.array.Array1DUtil;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import javax.vecmath.Point3i;

import plugins.adufour.connectedcomponents.ConnectedComponent;
//import plugins.adufour.vars.lang.VarGenericArray;
import plugins.adufour.vars.lang.VarSequence;

public class ConnectedComponentsCustom extends Plugin{
	
	double BACKGROUND_VALUE = 0.0;	
	
	boolean NO_EDGE_X = false;
	boolean NO_EDGE_Y = false;
	boolean NO_EDGE_Z = false;
	
	int MIN_SIZE = 0;
	int MAX_SIZE =  Integer.MAX_VALUE;
	
	Sequence inputSequence;
	Sequence output;
	
	VarSequence outputSequence =  new VarSequence("output", null);
	//VarGenericArray<ConnectedComponent[]> outputCCs   = new VarGenericArray<ConnectedComponent[]>("components", ConnectedComponent[].class, null);
	
	Map<Integer, List<ConnectedComponent>> componentsMap;
	ArrayList<ConnectedComponent> componentsList;
	
	static int LARGEST_COMPONENT = 0;
	static int LABEL_LARGETS_COMP = 0;
	
	
	/**
     * List of extraction methods suitable for the Connected Components plugin
     * 
     * @author Alexandre Dufour
     * 
     */
    public static enum ExtractionType
    {
        /**
         * The user defined value is considered as the background, and all components with a
         * different intensity should be extracted (regardless of intensity variations)
         */
        BACKGROUND,
        /**
         * The user defined value is considered as the background, and all components with a
         * different intensity should be extracted, accounting for intensity variations to
         * distinguish touching components
         */
        BACKGROUND_LABELED,
        /**
         * Extracts components with pixel intensities matching the user defined value
         */
        VALUE
    }
	
	public ConnectedComponentsCustom(Sequence inputSequence){
		this.inputSequence = inputSequence;
		output = new Sequence();
		
		componentsMap = new TreeMap<Integer, List<ConnectedComponent>>();
	}
	
	public void process(){
		LARGEST_COMPONENT = 0;
		LABEL_LARGETS_COMP = 0;
		
		
		componentsList = new ArrayList<ConnectedComponent>();
		componentsMap = extractConnectedComponents(inputSequence, BACKGROUND_VALUE, ExtractionType.BACKGROUND,  NO_EDGE_X, NO_EDGE_Y, NO_EDGE_Z,
                MIN_SIZE, MAX_SIZE, output);
		
		outputSequence.setValue(output);
        
		/*
        int nbObjects = 0;          
        
        for (List<ConnectedComponent> ccs : componentsMap.values())
        {
            nbObjects += ccs.size();
            componentsList.addAll(ccs);                	
        }      
        */
		
		System.out.println("largest comp = " + LARGEST_COMPONENT + ", label = " + LABEL_LARGETS_COMP);
		
		if (Const.GUI2_chkSteps){	
			output.setName("Step 7: Connected components");
			addSequence(output);
		}
	}
	
	public Sequence getSequenceProcessed(){
		return output;
	}
	
	public int getLabelLargestComponent(){
		return LABEL_LARGETS_COMP;
	}
	
	/**
     * Extracts the connected components in the given sequence with specified size bounds. Note that
     * the method works on both binary gray-scale images. If the input is not binary, an option can
     * be set to distinguish touching objects which have a different intensity value, or to extract
     * a specific intensity value
     * 
     * @param inputSequence
     *            A binary or gray-scale sequence.
     * @param value
     *            the user value to be interpreted depending on the type parameter
     * @param type
     *            the extraction type (or how to interpret the value parameter)
     * @param noEdgeX
     *            set to true if components touching the image edge along X should be discarded
     *            during the extraction
     * @param noEdgeY
     *            set to true if components touching the image edge along Y should be discarded
     *            during the extraction
     * @param noEdgeZ
     *            set to true if components touching the image edge along Z should be discarded
     *            during the extraction
     * @param minSize
     *            the minimum size of the objects to extract
     * @param maxSize
     *            the maximum size of the objects to extract
     * @param labeledSequence
     *            a single-band integer sequence that will be filled with the labeled components
     * @return a map with extracted components, indexed by the sequence time points
     * @see ExtractionType
     * @see ConnectedComponent
     */
    public static Map<Integer, List<ConnectedComponent>> extractConnectedComponents(Sequence inputSequence, double value, ExtractionType type, boolean noEdgeX, boolean noEdgeY, boolean noEdgeZ,
            int minSize, int maxSize, Sequence labeledSequence)
    {
        if (inputSequence == null || inputSequence.getSizeT() == 0) throw new IllegalArgumentException("Cannot extract connected components from an emtpy sequence !");
        
        int width = inputSequence.getSizeX();
        int height = inputSequence.getSizeY();
        
        if (labeledSequence == null) labeledSequence = new Sequence();
        
        Map<Integer, List<ConnectedComponent>> componentsMap = new TreeMap<Integer, List<ConnectedComponent>>();
        
        for (int t = 0; t < inputSequence.getSizeT(); t++)
        {
            for (int z = 0; z < inputSequence.getSizeZ(); z++)
                labeledSequence.setImage(t, z, new IcyBufferedImage(width, height, 1, DataType.UINT));
            
            VolumetricImage volIN = inputSequence.getVolumetricImage(t);
            VolumetricImage volOUT = labeledSequence.getVolumetricImage(t);
            
            List<ConnectedComponent> components = extractConnectedComponents(volIN, value, type, noEdgeX, noEdgeY, noEdgeZ, minSize, maxSize, volOUT);
            
            // int cpt=1;
            for (ConnectedComponent cc : components)
            {
                cc.setT(t);               
                
                // System.out.println(t + "\t" + cpt++ + "\t" + (int)cc.getMassCenter().x + "\t" +
                // (int)cc.getMassCenter().y);
            }
            
            componentsMap.put(t, components);
        }
        
        labeledSequence.updateChannelsBounds(true);
        labeledSequence.getColorModel().setColormap(0, new FireColorMap());
        
        return componentsMap;
    }
    
    
    /**
     * Extracts the connected components in the given volumetric image with specified size bounds.
     * The the method works on both binary gray-scale images. If the input is not binary, an option
     * can be set to distinguish touching objects which have a different intensity value, or to
     * extract a specific intensity value
     * 
     * @param stack
     *            A binary or gray-scale volumetric image.
     * @param value
     *            the user value to be interpreted depending on the type parameter
     * @param type
     *            the extraction type (or how to interpret the value parameter)
     * @param noEdgeX
     *            set to true if components touching the image edge along X should be discarded
     *            during the extraction
     * @param noEdgeY
     *            set to true if components touching the image edge along Y should be discarded
     *            during the extraction
     * @param noEdgeZ
     *            set to true if components touching the image edge along Z should be discarded
     *            during the extraction
     * @param minSize
     *            the minimum size of the objects to extract
     * @param maxSize
     *            the maximum size of the objects to extract
     * @param labeledStack
     *            a volumetric image where labeled information will be stored during the extraction.
     *            Note that this volumetric image should be non-null and filled with zero-values (no
     *            cleaning is performed to optimize recursive processes)
     * @return a collection containing all components extracted.
     * @throws NullPointerException
     *             is labeledStack is null
     * @see ExtractionType
     * @see ConnectedComponent
     */
    public static List<ConnectedComponent> extractConnectedComponents(VolumetricImage stack, double value, ExtractionType type, boolean noEdgeX, boolean noEdgeY, boolean noEdgeZ, int minSize,
            int maxSize, final VolumetricImage labeledStack) throws NullPointerException
    {
        int width = stack.getFirstImage().getSizeX();
        int height = stack.getFirstImage().getSizeY();
        int depth = stack.getSize();
        
        int[] neighborLabelValues = new int[13];
        int neighborhoodSize = 0;
        
        boolean extractUserValue = (type == ExtractionType.VALUE);
        
        Label[] labels = new Label[width * height * depth];
        
        // first image pass: naive labeling with simple backward neighborhood
        
        int highestKnownLabel = 0;
        
        boolean onEdgeX = false;
        boolean onEdgeY = false;
        boolean onEdgeZ = false;
        
        for (int z = 0; z < depth; z++)
        {
            onEdgeZ = (z == 0 || z == depth - 1);
            
            // retrieve the direct pointer to the current slice
            int[] _labelsInCurrentSlice = labeledStack.getImage(z).getDataXYAsInt(0);
            // retrieve a direct pointer to the previous slice
            int[] _labelsInUpperSlice = (z == 0) ? null : labeledStack.getImage(z - 1).getDataXYAsInt(0);
            
            int voxelOffset = 0;
            
            Object inputData = stack.getImage(z).getDataXY(0);
            DataType dataType = stack.getImage(z).getDataType_();
            
            for (int y = 0; y < height; y++)
            {
                onEdgeY = (y == 0 || y == height - 1);
                
                for (int x = 0; x < width; x++, voxelOffset++)
                {
                    onEdgeX = (x == 0 || x == width - 1);
                    
                    double pixelValue = Array1DUtil.getValue(inputData, voxelOffset, dataType);
                    
                    boolean pixelEqualsUserValue = (pixelValue == value);
                    
                    // do not process the current pixel if:
                    // - extractUserValue is true and pixelEqualsUserValue is false
                    // - extractUserValue is false and pixelEqualsUserValue is true
                    
                    if (extractUserValue != pixelEqualsUserValue) continue;
                    
                    // the current pixel should be labeled
                    
                    // -> look for existing labels in its neighborhood
                    
                    // 1) define the neighborhood of interest here
                    // NB: this is a single pass method, so backward neighborhood is sufficient
                    
                    // legend:
                    // e = edge
                    // x = current pixel
                    // n = valid neighbor
                    // . = other neighbor
                    
                    if (z == 0)
                    {
                        if (y == 0)
                        {
                            if (x == 0)
                            {
                                // e e e
                                // e x .
                                // e . .
                                
                                // do nothing
                            }
                            else
                            {
                                // e e e
                                // n x .
                                // . . .
                                
                                neighborLabelValues[0] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 1;
                            }
                        }
                        else
                        {
                            int north = voxelOffset - width;
                            
                            if (x == 0)
                            {
                                // e n n
                                // e x .
                                // e . .
                                
                                neighborLabelValues[0] = _labelsInCurrentSlice[north];
                                neighborLabelValues[1] = _labelsInCurrentSlice[north + 1];
                                neighborhoodSize = 2;
                            }
                            else if (x == width - 1)
                            {
                                // n n e
                                // n x e
                                // . . e
                                
                                neighborLabelValues[0] = _labelsInCurrentSlice[north - 1];
                                neighborLabelValues[1] = _labelsInCurrentSlice[north];
                                neighborLabelValues[2] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 3;
                            }
                            else
                            {
                                // n n n
                                // n x .
                                // . . .
                                
                                neighborLabelValues[0] = _labelsInCurrentSlice[north - 1];
                                neighborLabelValues[1] = _labelsInCurrentSlice[north];
                                neighborLabelValues[2] = _labelsInCurrentSlice[north + 1];
                                neighborLabelValues[3] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 4;
                            }
                        }
                    }
                    else
                    {
                        if (y == 0)
                        {
                            int south = voxelOffset + width;
                            
                            if (x == 0)
                            {
                                // e e e | e e e
                                // e n n | e x .
                                // e n n | e . .
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[1] = _labelsInUpperSlice[voxelOffset + 1];
                                neighborLabelValues[2] = _labelsInUpperSlice[south];
                                neighborLabelValues[3] = _labelsInUpperSlice[south + 1];
                                neighborhoodSize = 4;
                            }
                            else if (x == width - 1)
                            {
                                // e e e | e e e
                                // n n e | n x e
                                // n n e | . . e
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[voxelOffset - 1];
                                neighborLabelValues[1] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[2] = _labelsInUpperSlice[south - 1];
                                neighborLabelValues[3] = _labelsInUpperSlice[south];
                                neighborLabelValues[4] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 5;
                            }
                            else
                            {
                                // e e e | e e e
                                // n n n | n x .
                                // n n n | . . .
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[voxelOffset - 1];
                                neighborLabelValues[1] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[2] = _labelsInUpperSlice[voxelOffset + 1];
                                neighborLabelValues[3] = _labelsInUpperSlice[south - 1];
                                neighborLabelValues[4] = _labelsInUpperSlice[south];
                                neighborLabelValues[5] = _labelsInUpperSlice[south + 1];
                                neighborLabelValues[6] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 7;
                            }
                        }
                        else if (y == height - 1)
                        {
                            int north = voxelOffset - width;
                            
                            if (x == 0)
                            {
                                // e n n | e n n
                                // e n n | e x .
                                // e e e | e e e
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[north];
                                neighborLabelValues[1] = _labelsInUpperSlice[north + 1];
                                neighborLabelValues[2] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[3] = _labelsInUpperSlice[voxelOffset + 1];
                                neighborLabelValues[4] = _labelsInCurrentSlice[north];
                                neighborLabelValues[5] = _labelsInCurrentSlice[north + 1];
                                neighborhoodSize = 6;
                            }
                            else if (x == width - 1)
                            {
                                // n n e | n n e
                                // n n e | n x e
                                // e e e | e e e
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[north - 1];
                                neighborLabelValues[1] = _labelsInUpperSlice[north];
                                neighborLabelValues[2] = _labelsInUpperSlice[voxelOffset - 1];
                                neighborLabelValues[3] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[4] = _labelsInCurrentSlice[north - 1];
                                neighborLabelValues[5] = _labelsInCurrentSlice[north];
                                neighborLabelValues[6] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 7;
                            }
                            else
                            {
                                // n n n | n n n
                                // n n n | n x .
                                // e e e | e e e
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[north - 1];
                                neighborLabelValues[1] = _labelsInUpperSlice[north];
                                neighborLabelValues[2] = _labelsInUpperSlice[north + 1];
                                neighborLabelValues[3] = _labelsInUpperSlice[voxelOffset - 1];
                                neighborLabelValues[4] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[5] = _labelsInUpperSlice[voxelOffset + 1];
                                neighborLabelValues[6] = _labelsInCurrentSlice[north - 1];
                                neighborLabelValues[7] = _labelsInCurrentSlice[north];
                                neighborLabelValues[8] = _labelsInCurrentSlice[north + 1];
                                neighborLabelValues[9] = _labelsInCurrentSlice[voxelOffset - 1];
                                neighborhoodSize = 10;
                            }
                        }
                        else
                        {
                            int north = voxelOffset - width;
                            int south = voxelOffset + width;
                            
                            if (x == 0)
                            {
                                // e n n | e n n
                                // e n n | e x .
                                // e n n | e . .
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[north];
                                neighborLabelValues[1] = _labelsInUpperSlice[north + 1];
                                neighborLabelValues[2] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[3] = _labelsInUpperSlice[voxelOffset + 1];
                                neighborLabelValues[4] = _labelsInUpperSlice[south];
                                neighborLabelValues[5] = _labelsInUpperSlice[south + 1];
                                neighborLabelValues[6] = _labelsInCurrentSlice[north];
                                neighborLabelValues[7] = _labelsInCurrentSlice[north + 1];
                                neighborhoodSize = 8;
                            }
                            else if (x == width - 1)
                            {
                                int northwest = north - 1;
                                int west = voxelOffset - 1;
                                
                                // n n e | n n e
                                // n n e | n x e
                                // n n e | . . e
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[northwest];
                                neighborLabelValues[1] = _labelsInUpperSlice[north];
                                neighborLabelValues[2] = _labelsInUpperSlice[west];
                                neighborLabelValues[3] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[4] = _labelsInUpperSlice[south - 1];
                                neighborLabelValues[5] = _labelsInUpperSlice[south];
                                neighborLabelValues[6] = _labelsInCurrentSlice[northwest];
                                neighborLabelValues[7] = _labelsInCurrentSlice[north];
                                neighborLabelValues[8] = _labelsInCurrentSlice[west];
                                neighborhoodSize = 9;
                            }
                            else
                            {
                                int northwest = north - 1;
                                int west = voxelOffset - 1;
                                int northeast = north + 1;
                                int southwest = south - 1;
                                int southeast = south + 1;
                                
                                // n n n | n n n
                                // n n n | n x .
                                // n n n | . . .
                                
                                neighborLabelValues[0] = _labelsInUpperSlice[northwest];
                                neighborLabelValues[1] = _labelsInUpperSlice[north];
                                neighborLabelValues[2] = _labelsInUpperSlice[northeast];
                                neighborLabelValues[3] = _labelsInUpperSlice[west];
                                neighborLabelValues[4] = _labelsInUpperSlice[voxelOffset];
                                neighborLabelValues[5] = _labelsInUpperSlice[voxelOffset + 1];
                                neighborLabelValues[6] = _labelsInUpperSlice[southwest];
                                neighborLabelValues[7] = _labelsInUpperSlice[south];
                                neighborLabelValues[8] = _labelsInUpperSlice[southeast];
                                neighborLabelValues[9] = _labelsInCurrentSlice[northwest];
                                neighborLabelValues[10] = _labelsInCurrentSlice[north];
                                neighborLabelValues[11] = _labelsInCurrentSlice[northeast];
                                neighborLabelValues[12] = _labelsInCurrentSlice[west];
                                neighborhoodSize = 13;
                            }
                        }
                    }
                    
                    // 2) the neighborhood is ready, move to the labeling step
                    
                    int currentVoxelLabelValue = Integer.MAX_VALUE;
                    
                    // to avoid creating too many labels and fuse them later on,
                    // find the minimum non-zero label in the neighborhood
                    // and assign that minimum label right now
                    
                    for (int i = 0; i < neighborhoodSize; i++)
                    {
                        int neighborLabelValue = neighborLabelValues[i];
                        
                        // zero labels are not interesting...
                        if (neighborLabelValue == 0) continue;
                        
                        // neighbor labels should have the same/different ? image value
                        if (type == ExtractionType.BACKGROUND_LABELED && labels[neighborLabelValue].imageValue != pixelValue) continue;
                        
                        // here, the neighbor label is valid
                        // => check if it is lower
                        if (neighborLabelValue < currentVoxelLabelValue)
                        {
                            currentVoxelLabelValue = neighborLabelValue;
                        }
                    }
                    
                    if (currentVoxelLabelValue == Integer.MAX_VALUE)
                    {
                        // currentVoxelLabel didn't change
                        // => no lower neighbor value found
                        // => new label
                        highestKnownLabel++;
                        currentVoxelLabelValue = highestKnownLabel;
                        labels[currentVoxelLabelValue] = new Label(pixelValue, currentVoxelLabelValue);
                    }
                    else
                    {
                        // currentVoxelLabelValue has been modified
                        // -> browse its neighborhood again
                        // -> find all neighbors with a higher label value
                        // -> change their value to currentVoxelLabelValue
                        // -> change their target to currentVoxelLabel
                        
                        Label currentVoxelLabel = labels[currentVoxelLabelValue];
                        
                        for (int i = 0; i < neighborhoodSize; i++)
                        {
                            int neighborLabelValue = neighborLabelValues[i];
                            
                            if (neighborLabelValue > currentVoxelLabelValue)
                            {
                                Label label = labels[neighborLabelValue];
                                
                                if (type == ExtractionType.BACKGROUND_LABELED && label.imageValue != pixelValue) continue;
                                
                                int finalLabelValue = label.getFinalLabelValue();
                                Label finalLabel = labels[finalLabelValue];
                                
                                if (currentVoxelLabel.targetLabelValue == finalLabelValue) continue;
                                
                                if (currentVoxelLabelValue < finalLabelValue)
                                {
                                    finalLabel.targetLabel = currentVoxelLabel;
                                    finalLabel.targetLabelValue = currentVoxelLabelValue;
                                }
                                else if (currentVoxelLabelValue > finalLabelValue)
                                {
                                    currentVoxelLabel.targetLabel = finalLabel;
                                    currentVoxelLabel.targetLabelValue = finalLabelValue;
                                }
                            }
                        }
                    }
                    
                    // -> store this label in the labeled image
                    _labelsInCurrentSlice[voxelOffset] = currentVoxelLabelValue;
                    labels[currentVoxelLabelValue].size++;
                    labels[currentVoxelLabelValue].onEdgeX |= onEdgeX;
                    labels[currentVoxelLabelValue].onEdgeY |= onEdgeY;
                    labels[currentVoxelLabelValue].onEdgeZ |= onEdgeZ;
                }
            }
        }
        
        // end of the first pass, all pixels have a label
        // (though might not be unique within a given component)
        
        HashMap<Integer, ConnectedComponent> componentsMap = new HashMap<Integer, ConnectedComponent>();
        
        // fusion strategy: fuse higher labels with lower ones
        // "highestKnownLabel" holds the highest known label
        // -> loop backward from there to accumulate object size recursively
        
        int finalLabel = 0;
        
        for (int labelValue = highestKnownLabel; labelValue > 0; labelValue--)
        {
            Label label = labels[labelValue];
            
            int targetLabelValue = label.targetLabelValue;
            
            
            
            if (targetLabelValue < labelValue)
            {
                // label should be fused to targetLabel
                
                Label targetLabel = labels[targetLabelValue];
                
                // -> add label's size to targetLabel
                targetLabel.size += label.size;
                
                // -> mark targetLabel as onEdge if label is
                targetLabel.onEdgeX |= label.onEdgeX;
                targetLabel.onEdgeY |= label.onEdgeY;
                targetLabel.onEdgeZ |= label.onEdgeZ;
                
                // -> mark label to fuse with targetLabel
                label.targetLabel = labels[targetLabelValue];                
                
            }
            else
            {
                // label has same labelValue and targetLabelValue
                // -> it cannot be fused to anything
                // -> this is a terminal label
                
                // -> check if it obeys to user constraints          	       	
            	
            	
                
                if (label.size < minSize || label.size > maxSize)
                {
                    // the component size is out of the given range
                    // -> mark the object for deletion
                    label.targetLabelValue = 0;
                }
                else if ((noEdgeX && label.onEdgeX) || (noEdgeY && label.onEdgeY) || (noEdgeZ && label.onEdgeZ))
                {
                    // the component size is on an edge to discard
                    // -> mark the object for deletion
                    label.targetLabelValue = 0;
                }
                
                
                
                else
                {
                    // the label is clean and user-valid
                    // -> assign its final labelValue (for the final image labeling pass)
                    finalLabel++;
                    label.targetLabelValue = finalLabel;
                    
                    // -> add this label to the list of valid labels
                    
                    ConnectedComponent component = new ConnectedComponent(label.size);
                    
                    // erased
                    /*
                    component.onEdgeX = label.onEdgeX;
                    component.onEdgeY = label.onEdgeY;
                    component.onEdgeZ = label.onEdgeZ;
                    */
                    
                    componentsMap.put(finalLabel, component);
                    
                    /*
                    // find largest component
                    if (label.size > LARGEST_COMPONENT){
                		LARGEST_COMPONENT = label.size;
                		LABEL_LARGETS_COMP = label.targetLabelValue;
                	} 
                    */
                }
                
            }
        }
        
        // 3) second image pass: replace all labels by their final values
        
        for (int z = 0; z < depth; z++)
        {
            int[] _outputSlice = labeledStack.getImage(z).getDataXYAsInt(0);
            
            int pixelOffset = 0;
            
            for (int j = 0; j < height; j++)
            {
                for (int i = 0; i < width; i++, pixelOffset++)
                {
                    int targetLabelValue = _outputSlice[pixelOffset];
                    
                    if (targetLabelValue == 0) continue;
                    
                    // if a fusion was indicated, retrieve the final label value
                    targetLabelValue = labels[targetLabelValue].getFinalLabelValue();                     
                    
                    // assign the final label in the output image
                    _outputSlice[pixelOffset] = targetLabelValue;
                    
                    if (targetLabelValue == 0) continue;
                    
                    // store the current pixel in the component
                    //componentsMap.get(targetLabelValue).addPointInternal(new Point3i(i, j, z));
                    componentsMap.get(targetLabelValue).addPoint(new Point3i(i, j, z));
                }
            }
            
            
        }
        
        // find largest component       
        System.out.println("map.size = " + componentsMap.size());
        System.out.println("map.values().size = " + componentsMap.values().size());
        System.out.println("LARGEST_COMPONENT = " + LARGEST_COMPONENT + ", LABEL_LARGETS_COMP = " + LABEL_LARGETS_COMP);
        
        
        for (int i=0; i<componentsMap.size(); i++)   
        if (componentsMap.get(i) != null)
        if (componentsMap.get(i).getSize() > LARGEST_COMPONENT){        	
        	LARGEST_COMPONENT = componentsMap.get(i).getSize();
        	LABEL_LARGETS_COMP = i;
        }
        
        /*
        if (labels[targetLabelValue].size > LARGEST_COMPONENT){
    		LARGEST_COMPONENT = labels[targetLabelValue].size;
    		LABEL_LARGETS_COMP = targetLabelValue;
    	}
        */
        
        return new ArrayList<ConnectedComponent>(componentsMap.values());
    }
    
    
    
    private static class Label
    {
        final double    imageValue;
        
        /**
         * final label that should replace the current label if fusion is needed
         */
        int             targetLabelValue;
        
        /**
         * if non-null, indicates the parent object with which the current object should be fused
         */
        Label           targetLabel;
        
        int             size;
        
        private boolean onEdgeX;
        
        private boolean onEdgeY;
        
        private boolean onEdgeZ;
        
        /**
         * Creates a new label with the given value. If no parent is set to this label, the given
         * value will be the final one
         * 
         * @param value
         *            the pixel value
         * @param label
         *            the label value
         * @param onEdgeX
         *            true if the pixel is on the image edge along X
         * @param onEdgeY
         *            true if the pixel is on the image edge along Y
         * @param onEdgeZ
         *            true if the pixel is on the image edge along Z
         */
        Label(double value, int label)
        {
            this.imageValue = value;
            this.targetLabelValue = label;
        }
        
        /**
         * Retrieves the final object label (recursively)
         * 
         * @return
         */
        int getFinalLabelValue()
        {
            return targetLabel == null ? targetLabelValue : targetLabel.getFinalLabelValue();
        }
    }

}
