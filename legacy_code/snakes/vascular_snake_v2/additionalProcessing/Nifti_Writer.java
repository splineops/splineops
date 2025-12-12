package plugins.big.vascular.additionalProcessing;

import java.io.*; 
import java.awt.*; 

import icy.plugin.abstract_.Plugin;
import icy.sequence.Sequence;
import ij.*; 
import ij.gui.*; 
//import ij.plugin.*;
import ij.process.*;
import ij.io.*;
import ij.measure.Calibration;
import imageware.Builder;
import imageware.ImageWare;

/**	This plugin saves Analyze or Nifti-1 format files.

	The first characters of the input variable arg is used to determine in
	which format to save:

		::ANALYZE_7_5:		- Analyze 7.5 format
		::NIFTI_ANALYZE:	- Nifti-1 .img/.hdr pair
		::NIFTI_FILE:		- Nifti-1 combined header/image file (.nii)
	
	If arg does not start with these characters, .nii format is assumed.
	The rest of the arg string (if present) is used as a filename. If this is blank, 
	then a dialog box is presented.

	- Recompiling with the flag "littleEndian" set to be true will force the image 
	  to be saved in little endian format.
	  
	- Requires ImageJ 1.34p or later 

	Guy Williams, gbw1000@wbic.cam.ac.uk 	19/08/2005	
*/  

public class Nifti_Writer extends Plugin {

	public static final int ANALYZE_7_5 = 0; 
	public static final int NIFTI_ANALYZE = 1; 
	public static final int NIFTI_FILE = 2; 
	
	public boolean littleEndian = false;	// Change this to force little endian output 

	private int output_type = NIFTI_FILE;	// Change this to change default output

	private boolean signed16Bit = false; 
	
	private Sequence seq;
	private int nx, ny, nz;
	
	public Nifti_Writer(){
		this.seq = getFocusedSequence();
		nx = seq.getSizeX();
		ny = seq.getSizeY();
		nz = seq.getSizeZ();		
	}
	

	public void run(String arg) {
		
		//
		ImageWare imW = Builder.create(nx, ny, nz, ImageWare.DOUBLE);			
		
		for (int z=0; z<nz; z++)
		for (int y=0; y<ny; y++)
		for (int x=0; x<nx; x++){
			double val = seq.getData(0, z, 0, y, x);			
			imW.putPixel(x, y, z, val);		
		}
		ImageStack imWStack = imW.buildImageStack();
		ImagePlus imp = new ImagePlus("", imWStack);
		//
		
		/*
		ImagePlus imp = WindowManager.getCurrentImage();
	 	if (imp==null) {
			IJ.noImage(); 
			return; 
		}
	 	*/
	 	
	 	
		String directory = "", name = ""; 
		
		if (arg!=null) { 
			if (arg.startsWith("::ANALYZE_7_5:")) {
				output_type = ANALYZE_7_5; 
				arg=arg.substring(14).trim();
			} else if (arg.startsWith("::NIFTI_ANALYZE:")) { 
				output_type = NIFTI_ANALYZE; 
				arg=arg.substring(16).trim();
			} else if (arg.startsWith("::NIFTI_FILE:")) { 
				output_type = NIFTI_FILE; 
				arg=arg.substring(13).trim(); 
			}
		}
		if ((arg==null) || (arg.equals(""))) {
			String fType = "", suffix = "";
			switch(output_type) { 
				case ANALYZE_7_5:
				case NIFTI_ANALYZE:
					fType = "Analyze"; 
					suffix = ".img";
					break;
				case  NIFTI_FILE:
					fType = "Nifti"; 
					suffix = ".nii";
					break;
			}
			
			SaveDialog sd = new SaveDialog("Save as " + fType, imp.getTitle(), suffix);
			directory = sd.getDirectory();
			name = sd.getFileName();
		} else {
			File file = new File(arg); 
			if (file.isDirectory()) {
				directory = arg;
				name = imp.getTitle();
			} else {
				directory = file.getParent(); 
				name = file.getName(); 
			}	
		}
		if (name == null || name == "") {
			IJ.showStatus(""); 
			//IJ.error("No filename selected");
			return;
		}
		if (is16BitSigned(imp)) {
			add(imp, -32768);
			signed16Bit = true;
		}
		ImageStack stack = imp.getStack();
		if (output_type==ANALYZE_7_5) { 
			for (int i=1; i<=stack.getSize(); i++) {
				ImageProcessor ip = stack.getProcessor(i);
				ip.flipVertical();
			}
		}
		int nChannels = imp.getNChannels();
		int nOthers = imp.getNFrames() * imp.getNSlices();
		if (nChannels != 1) { 
			reshuffleStack( stack.getImageArray(), nChannels, nChannels * nOthers ); 
		}
		save(imp, directory, name);
		if (imp.getNChannels() != 1) { 
			reshuffleStack( stack.getImageArray(), nOthers, nChannels * nOthers ); 
		}
		if (output_type==ANALYZE_7_5) { 
			for (int i=1; i<=stack.getSize(); i++) {
				ImageProcessor ip = stack.getProcessor(i);
				ip.flipVertical();
			}
		}
		if (signed16Bit) {
			add(imp, 32768);
			signed16Bit = false;	
		}

		IJ.showStatus(""); 
	}

	// Save return false if one of the files already exists and the user pressed Cancel.
	public boolean save(ImagePlus imp, String directory, String name) {
		if (name == null) return false;
		name = name.trim();
		directory = directory.trim();
		if (output_type!=NIFTI_FILE) { 
			if (name.toLowerCase().endsWith(".img")) name = name.substring(0, name.length() - 4); 
			if (name.toLowerCase().endsWith(".hdr")) name = name.substring(0, name.length() - 4); 
		}

		if (!directory.endsWith(File.separator)) directory += File.separator; 

		try {
			String hdrFile = (output_type==NIFTI_FILE) ? directory + name : directory + name + ".hdr";
			FileOutputStream stream = new FileOutputStream(hdrFile);
			DataOutputStream output = new DataOutputStream(stream);
		
			IJ.showStatus("Saving as Analyze: " + directory + name);
		
			writeHeader( imp, output, output_type );			
			if (output_type!=NIFTI_FILE) { 
				output.close(); 
				stream.close();
				String fileName = directory + name + ".img";				
			
				stream = new FileOutputStream(fileName); 
				output = new DataOutputStream(stream);
			}
			FileInfo fi = imp.getFileInfo();
			fi.intelByteOrder = littleEndian;
			if (fi.fileType != FileInfo.RGB) { 
				if (imp.getStackSize()>1 && imp.getStack().isVirtual()) {
					fi.virtualStack = (VirtualStack)imp.getStack();
					fi.fileName = "FlipTheseImages";
				}
				ImageWriter iw = new ImageWriter( fi );
				iw.write(output); 
			} else { 
				writeRGBPlanar(imp, output); 
			}

			output.close();
			stream.close();

			return true;	
		}
		catch (IOException e) {
			IJ.log("Nifti_Writer: "+ e.getMessage());
			return false;	
		}
	} 

	private void writeHeader( ImagePlus imp, DataOutputStream output, int type ) throws IOException {
		FileInfo fi = imp.getFileInfo();
		short bitsallocated, datatype;

		NiftiHeader nfti_hdr = (NiftiHeader) imp.getProperty("nifti");
		Calibration c = imp.getCalibration(); 
		switch (fi.fileType) {	
			case FileInfo.GRAY8:
				datatype = 2; 		// DT_UNSIGNED_CHAR 
				bitsallocated = 8;
				break;
			case FileInfo.GRAY16_SIGNED:
			case FileInfo.GRAY16_UNSIGNED:
				datatype = 4; 		// DT_SIGNED_SHORT 
				bitsallocated = 16;
				break;
			case FileInfo.GRAY32_INT:
				datatype = 8; 		// DT_SIGNED_INT
				bitsallocated = 32;
				break; 
			case FileInfo.GRAY32_FLOAT:
				datatype = 16; 		// DT_FLOAT 
				bitsallocated = 32;
				break; 
			case FileInfo.RGB:
				datatype = 128; 	// DT_RGB 
				bitsallocated = 24;
				break; 
			default:
				datatype = 0;		// DT_UNKNOWN
				bitsallocated = (short) (fi.getBytesPerPixel() * 8) ; 
		}

		//     header_key  

		writeInt(output, 348); 				// sizeof_hdr
		int i;
		for (i = 0; i < 10; i++) output.write( 0 );	// data_type
		for (i = 0; i < 18; i++) output.write( 0 ); 	// db_name 
		writeInt(output, 16384); 			// extents 
		output.writeShort( 0); 				// session_error
		output.writeByte ( (int) 'r' );			// regular 
		output.writeByte ( 0 );				// hkey_un	dim_info 

		// image_dimension
		short [] dims = new short[8];
		dims[0] = (imp.getNChannels() == 1) ? (short) 4 : (short) 5;
		dims[1] = (short) fi.width; 
		dims[2] = (short) fi.height; 
		dims[3] = (short) imp.getNSlices();
		dims[4] = (short) imp.getNFrames(); 
		dims[5] = (short) imp.getNChannels(); 
		
		for (i = 0; i < 8; i++) writeShort( output, dims[i] );		// dim[0-7]

		if (type==ANALYZE_7_5) { 
			String unit = c.getUnit() + "\0\0\0\0"; 
			unit = unit.substring(0,4);
			
			output.writeBytes ( unit );				// vox_units
			for (i = 0; i < 8; i++) output.write( 0 );		// cal_units[8] 
			output.writeShort( 0 );				// unused1
		} else { 
			writeFloat( output, (nfti_hdr==null) ? 0.0 : nfti_hdr.intent_p1 );	//		intent_p1 	
			writeFloat( output, (nfti_hdr==null) ? 0.0 : nfti_hdr.intent_p2 );	//		intent_p2 	
			writeFloat( output, (nfti_hdr==null) ? 0.0 : nfti_hdr.intent_p3 );	//		intent_p3 	
			writeShort( output, (nfti_hdr==null) ? 0 : nfti_hdr.intent_code );	//		intent_code 	
				
		}
		writeShort( output, (short) datatype );			// datatype 
		writeShort( output, (short) bitsallocated );		// bitpix
		if ((type==ANALYZE_7_5) || (nfti_hdr==null)) { 
			output.writeShort( 0 );				// dim_un0
		} else { 
			writeShort( output, nfti_hdr.slice_start );		//		slice_start 
		}
		
		double [] quaterns = new double [ 5 ]; 
		int qform_code = NiftiHeader.NIFTI_XFORM_UNKNOWN;
		int sform_code = NiftiHeader.NIFTI_XFORM_UNKNOWN;
		double qoffset_x = 0.0, qoffset_y = 0.0, qoffset_z = 0.0; 
		for (i=0; i<5; i++) quaterns[i] = 0.0;
		double [][] srow = new double[3][4];
		for (i=0; i<3; i++) srow[i][i] = 1.0;
		if (type != ANALYZE_7_5) { 
			Object orientProp = imp.getProperty("coors"); 
			if (orientProp instanceof CoordinateMapper[] ) { 
				CoordinateMapper[] mapper = ( CoordinateMapper[] ) orientProp;
				for (i=0; i<mapper.length; i++) { 
					if ( mapper[i] instanceof AffineCoors ) { 
						AffineCoors mp = (AffineCoors) ((CoordinateMapper) mapper[i]).copy();
						if (mp.convertToType( CoordinateMapper.NIFTI ) ) { 
							srow = mp.getMatrix(); 
							sform_code = NiftiHeader.getCoorTypeCode( mp.getName() );	
						}
					}
				
					if (mapper[i] instanceof QuaternCoors) { 
						QuaternCoors mp = (QuaternCoors) ((CoordinateMapper) mapper[i]).copy();
						
						if (mp.convertToType( CoordinateMapper.NIFTI ) ) {

							qform_code = NiftiHeader.getCoorTypeCode( mp.getName() );
							quaterns = mp.getQuaterns();

							qoffset_x = mapper[i].getX(0,0,0);
							qoffset_y = mapper[i].getY(0,0,0);
							qoffset_z = mapper[i].getZ(0,0,0);
						}
					}
				}
			} else if (nfti_hdr != null) { 
				quaterns[0] = nfti_hdr.pixdim[0]; 
				quaterns[2] = nfti_hdr.quatern_b; 
				quaterns[3] = nfti_hdr.quatern_c; 
				quaterns[4] = nfti_hdr.quatern_d; 
				qform_code = nfti_hdr.qform_code;	
				qoffset_x =  nfti_hdr.qoffset_x;
				qoffset_y =  nfti_hdr.qoffset_y;
				qoffset_z =  nfti_hdr.qoffset_z;
				
				sform_code = nfti_hdr.sform_code;
				for (i=0; i<4; i++) { 
					srow[0][i] = nfti_hdr.srow_x[i];
					srow[1][i] = nfti_hdr.srow_y[i];
					srow[2][i] = nfti_hdr.srow_z[i];
				}
			}
		}				
		float [] pixdims = new float[8];
		pixdims[0] = (float) quaterns[0];	
		pixdims[1] = (float) fi.pixelWidth;	
		pixdims[2] = (float) fi.pixelHeight;	
		pixdims[3] = (float) fi.pixelDepth;	
		pixdims[4] = (float) fi.frameInterval;	
		if ((type!=ANALYZE_7_5) && (nfti_hdr!=null)) {
			for (i=5; i<8; i++) pixdims[i] = nfti_hdr.pixdim[i]; 
		}
		
		for (i = 0; i < 8; i++) writeFloat( output, pixdims[i] );	// pixdim[4-7]
		
		writeFloat( output, (type==NIFTI_FILE) ? 352 : 0 );		// vox_offset 
		double [] coeff = new double[2];
		coeff[0] = 0.0;
		coeff[1] = 1.0;
		if (c.getFunction()==Calibration.STRAIGHT_LINE) { 
			coeff = c.getCoefficients(); 
		}	
		double c_offset = coeff[0];
		if (signed16Bit) { 
			if (coeff[1]!=0.0) c_offset += 32768.0 * coeff[1];	
			else c_offset += 32768.0;	
		}
		if (type==ANALYZE_7_5) { 
			writeFloat( output, 1 );		// roi_scale 
			writeFloat( output, 0 );		// funused1 
			writeFloat( output, 0 );		// funused2 
		} else { 
			writeFloat( output, coeff[1] ); 	// scl_slope
			writeFloat( output, c_offset ); 	// scl_inter

			writeShort( output, (nfti_hdr!=null) ? (short) nfti_hdr.slice_end : (short) (dims[3]-1) ); 	// slice_end
			output.write( (nfti_hdr!=null) ? nfti_hdr.slice_code : 0 );	 	// slice_code
			
			String unit = c.getUnit().toLowerCase().trim(); 
			byte xyzt_unit = NiftiHeader.UNITS_UNKNOWN;
			if (unit.equals("meter") || unit.equals("metre") || unit.equals("m") ) { 
				xyzt_unit |= NiftiHeader.UNITS_METER;
			} else if (unit.equals("mm")) { 
				xyzt_unit |= NiftiHeader.UNITS_MM;
			} else if (unit.equals("micron")) { 
				xyzt_unit |= NiftiHeader.UNITS_MICRON;
			}	
			output.write( xyzt_unit ); 			// xyzt_units	
					
		}
		double min = imp.getProcessor().getMin();
		double max = imp.getProcessor().getMax()+1; 
		
		writeFloat( output, coeff[0]+(coeff[1]*max) );		// cal_max 
		writeFloat( output, coeff[0]+(coeff[1]*min) );		// cal_min 
		if (type==ANALYZE_7_5) { 
			output.writeInt( 0 );			// compressed
			output.writeInt( 0 );			// verified  
			ImageStatistics s = imp.getStatistics();
			writeInt(output, signed16Bit ? (int) s.max+32768 : (int) s.max);		// glmax 
			writeInt(output, signed16Bit ? (int) s.min+32768 : (int) s.min);		// glmin 
		} else { 
			writeFloat( output, (nfti_hdr!=null) ? nfti_hdr.slice_duration : 0.0); 		// slice_duration
			writeFloat( output, (nfti_hdr!=null) ? nfti_hdr.toffset : 0.0); 		// toffset
			writeFloat( output, 0.0 );			// glmax
			writeFloat( output, 0.0 );			// glmin
		}

		// data_history 

		if (type==ANALYZE_7_5) { 
			for (i = 0; i < 80; i++) output.write( 0 );		// descrip  
			for (i = 0; i < 24; i++) output.write( 0 );		// aux_file 
		
			output.write(0);											// orient 
			for (i = 0; i < 10; i++) output.write( 0 );		// originator 
			for (i = 0; i < 10; i++) output.write( 0 );		// generated 
			for (i = 0; i < 10; i++) output.write( 0 );		// scannum 
			for (i = 0; i < 10; i++) output.write( 0 );		// patient_id  
			for (i = 0; i < 10; i++) output.write( 0 );		// exp_date 
			for (i = 0; i < 10; i++) output.write( 0 );		// exp_time  
			for (i = 0; i < 3; i++)  output.write( 0 );		// hist_un0
			output.writeInt( 0 );		// views 
			output.writeInt( 0 );		// vols_added 
			output.writeInt( 0 );		// start_field  
			output.writeInt( 0 );		// field_skip
			output.writeInt( 0 );		// omax  
			output.writeInt( 0 );		// omin 
			output.writeInt( 0 );		// smax  
			output.writeInt( 0 );		// smin 
		} else { 
			String desc = (nfti_hdr==null) ? new String() : nfti_hdr.descrip.trim();
			int length = desc.length(); 
			if (length > 80) { 
				output.writeBytes( desc.substring(0,80) ); 
			} else { 
				output.writeBytes( desc );
				for (i=length; i<80; i++) output.write(0); 
			}

			desc = (nfti_hdr==null) ? "" : nfti_hdr.aux_file.trim();
			length = desc.length(); 
			if (length > 24) { 
				output.writeBytes( desc.substring(0,24) ); 
			} else { 
				output.writeBytes( desc );
				for (i=length; i<24; i++) output.write(0); 
			}
	
			writeShort( output, (short) qform_code ); 	// qform_code 
			writeShort( output, (short) sform_code );  	// sform_code 
			writeFloat( output, quaterns[2]); 	// quatern_b	
			writeFloat( output, quaterns[3]); 	// quatern_c 	
			writeFloat( output, quaterns[4]); 	// quatern_d 	
		
			writeFloat( output, qoffset_x ); 	// qoffset_x 
			writeFloat( output, qoffset_y ); 	// qoffset_y  
			writeFloat( output, qoffset_z ); 	// qoffset_z  
		
			for (i=0; i<3; i++) { 
				for (int j=0; j<4; j++) writeFloat( output, srow[i][j]); // srow_x, srow_y, srow_z
			}	
			
			desc = (nfti_hdr == null) ? "" : nfti_hdr.intent_name.trim();
			length = desc.length(); 
			if (length > 16) { 
				output.writeBytes( desc.substring(0,16) ); 
			} else { 
				output.writeBytes( desc );
				for (i=length; i<16; i++) output.write(0); 
			}

			output.writeBytes( (type==NIFTI_ANALYZE) ? "ni1\0" : "n+1\0" ); 
			if (type==NIFTI_FILE) output.writeInt( 0 ); 	// extension	
		}
		
	}

	/* Resort the stack to make "channels" the 5th dimension */
	private void reshuffleStack(Object [] stack, int gap, int length) { 
		Object [] oldStack = new Object[ stack.length ];
		for (int i=0; i<oldStack.length; i++) oldStack[i] = stack[i];

		for (int i=0, n=0; i<gap; i++) { 
			for (int c=i; c<length; c+= gap, n++) { 
				stack[n] = oldStack[c]; 
			}
		}
	}

	private void writeRGBPlanar(ImagePlus imp, OutputStream out) throws IOException { 
		ImageStack stack = null; 
		int nImages = imp.getStackSize(); 
		if (nImages > 1) stack = imp.getStack();

		int w = imp.getWidth();
		int h = imp.getHeight();
		byte [] r,g,b;

		for (int i=1; i<=nImages; i++) { 
			ColorProcessor cp = (nImages == 1) ? (ColorProcessor) imp.getProcessor() : 
				(ColorProcessor) stack.getProcessor(i); 
			r = new byte[ w*h ];
			g = new byte[ w*h ];
			b = new byte[ w*h ];
			cp.getRGB(r,g,b);
			out.write(r, 0, w*h); 
			out.write(g, 0, w*h); 
			out.write(b, 0, w*h); 
			IJ.showProgress((double) i/nImages );	
		}
	}

	private void writeInt(DataOutputStream input, int value) throws IOException {
		if (littleEndian) { 
			byte b1 = (byte) (value & 0xff);
			byte b2 = (byte) ((value >> 8) & 0xff);
			byte b3 = (byte) ((value >> 16) & 0xff);
			byte b4 = (byte) ((value >> 24) & 0xff); 
			input.writeByte(b1);
			input.writeByte(b2);
			input.writeByte(b3);
			input.writeByte(b4);
		} else { 
			input.writeInt( value );  
		}
	}
	
	private void writeShort(DataOutputStream input, short value) throws IOException {
		if (littleEndian) { 
			byte b1 = (byte) (value & 0xff);
			byte b2 = (byte) ((value >> 8) & 0xff);
			input.writeByte(b1);
			input.writeByte(b2);
		} else { 	
			input.writeShort( value ); 
		}
	}
	
	private void writeFloat(DataOutputStream input, float value) throws IOException {
		writeInt(input, Float.floatToIntBits( value ) ); 
	}
	private void writeFloat(DataOutputStream input, double value) throws IOException {
		writeFloat(input, (float) value ); 
	}

	boolean is16BitSigned(ImagePlus imp) { 
		if (imp.getType() != ImagePlus.GRAY16) return false;
		int min = 65536, max = 0; 
		ImageStack s = imp.getStack();
		int sliceSize = imp.getWidth()*imp.getHeight();
		for (int i=1; i<=s.getSize(); i++) { 
			short [] pixels = (short []) s.getProcessor(i).getPixels(); 
			for (int j=0; j<sliceSize; j++) { 
				min = (min<(pixels[j]&0xffff)) ? min : pixels[j]&0xffff;
				max = (max>(pixels[j]&0xffff)) ? max : pixels[j]&0xffff;
			}
		}
		return (max>32767);

		// Old version
		//Calibration c = imp.getCalibration(); 
		//if (c.getFunction()!=Calibration.STRAIGHT_LINE) return false; 
		//double [] coeff = c.getCoefficients();
		//if (coeff[1] == 0.0) return false;
		//return (coeff[0] == - 32768.0 * coeff[1]); 
	}

        // adds the specified value to every pixel in the stack
        void add(ImagePlus imp, int value) {
            //IJ.log("add: "+value);
            ImageStack stack = imp.getStack();
            for (int slice=1; slice<=stack.getSize(); slice++) {
                ImageProcessor ip = stack.getProcessor(slice);
                short[] pixels = (short[])ip.getPixels();
                for (int i=0; i<pixels.length; i++)
                    pixels[i] = (short)((pixels[i]&0xffff)+value);
            }
         }

}

