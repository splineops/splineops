/*******************************************************************************
 * Copyright (c) 2012-2013 Biomedical Image Group (BIG), EPFL, Switzerland.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the GNU Public License v3.0
 * which accompanies this distribution, and is available at
 * http://www.gnu.org/licenses/gpl.html
 * 
 * Contributors:
 *     Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 *     Nicolas Chenouard (nicolas.chenouard@gmail.com)
 *     Philippe Th&#233;venaz (philippe.thevenaz@epfl.ch)
 *     Emrah Bostan (emrah.bostan@gmail.com)
 *     Ulugbek S. Kamilov (kamilov@gmail.com)
 *     Ramtin Madani (ramtin_madani@yahoo.com)
 *     Masih Nilchian (masih_n85@yahoo.com)
 *     C&#233;dric Vonesch (cedric.vonesch@epfl.ch)
 *     Virginie Uhlmann (virginie.uhlmann@epfl.ch)
 *     Cl&#233;ment Marti (clement.marti@epfl.ch)
 *     Julien Jacquemot (julien.jacquemot@epfl.ch)
 ******************************************************************************/
package plugins.big.bigsnake3d.core;

import icy.gui.frame.progress.ProgressFrame;
import icy.image.IcyBufferedImage;
import icy.sequence.Sequence;
import icy.sequence.SequenceUtil;
import icy.type.collection.array.Array1DUtil;
import plugins.big.bigsnakeutils.process.process3D.LoG;

/**
 * Class that encapsulates the image-related look-up tables.
 * 
 * @version May 3, 2014
 * 
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class ImageLUTContainerMSBrain {

	// ----------------------------------------------------------------------------
	// PROCESSING PARAMETERS

	/** Image smoothing factor. */
	private double sigma_ = 0;

	/** Active channel of the image. */
	private int channelNumber_ = 0;

	// ----------------------------------------------------------------------------
	// INPUT IMAGE

	/** Image of interest. */
	private Sequence sequence_ = null;
	/** Width of the original image. */
	private int width_ = 0;
	/** Height of the original image. */
	private int height_ = 0;
	/** Depth of the original image. */
	private int depth_ = 0;
	/** Array containing the image data of the active channel. */
	private double[][] imageDataArray_ = null;

	// ----------------------------------------------------------------------------
	// LOOK-UP TABLES WITH THE PREPROCESSED IMAGE

	/**
	 * Array containing a filtered image data of the active channel using a
	 * smoothed Laplacian filter.
	 */
	private double[][] filteredImageDataArray_ = null;
	/**
	 * Array containing the preintegrated image data along the vertical
	 * direction of the active channel.
	 */
	private double[][] preintegratedImageDataArray_ = null;
	/**
	 * Array containing the preintegrated and filtered image data along the
	 * vertical direction of the active channel with a smoothed Laplacian
	 * filter.
	 */
	private double[][] preintegratedFilteredImageDataArray_ = null;

	// ----------------------------------------------------------------------------
	// STATUS PARAMETERS OF THE CLASS

	/** If <code>true</code>, the LUTs are consistent with the input parameters. */
	private boolean isLUTUpToDate_ = false;

	/**
	 * Label of the XML tag containing the standard deviation of the Gaussian
	 * smoothing applied to the input image when computing the contour energy.
	 */
	public static final String ID_SIGMA = "sigma";

	/**
	 * Label of the XML tag containing the information of the channel number
	 * used in the snake.
	 */
	public static final String ID_ACTIVECHANNEL = "activeChannel";
	
	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public ImageLUTContainerMSBrain() {
	}

	// ----------------------------------------------------------------------------

	/** Constructs the look-up-tables that the snake needs to function. */
	public void buildLUTs() throws Exception {
		System.out.println("build ImageLUTContainter");
		
		Sequence singleChannelSequence = SequenceUtil.extractChannel(sequence_,
				channelNumber_);

		width_ = singleChannelSequence.getWidth();
		height_ = singleChannelSequence.getHeight();
		depth_ = singleChannelSequence.getSizeZ();

		ProgressFrame pFrame = new ProgressFrame("Precomputing LUTs");
		pFrame.setLength(4 * depth_);

		pFrame.setPosition(0);
		imageDataArray_ = new double[depth_][];
		for (int z = 0; z < depth_; z++) {
			IcyBufferedImage singleFrameImage = singleChannelSequence.getImage(
					0, z);
			imageDataArray_[z] = Array1DUtil.arrayToDoubleArray(
					singleFrameImage.getDataXY(0), singleFrameImage
							.getDataType_().isSigned());
			if (imageDataArray_[z] == null) {
				throw new Exception(
						"Image LUT failed to create. Please, contact the developer ");
			}
		}
		pFrame.setPosition(depth_);

		LoG log = new LoG(imageDataArray_, width_, height_, depth_, sigma_);
		filteredImageDataArray_ = log.filter();
		pFrame.setPosition(2 * depth_);

		preintegratedImageDataArray_ = new double[depth_][width_ * height_];
		preintegratedFilteredImageDataArray_ = new double[depth_][width_
				* height_];
		pFrame.setPosition(3 * depth_);

		double fyLap_val, fy_val;
		for (int z = 0; z < depth_; z++) {
			pFrame.incPosition();
			double[] imagePixels = null;
			double[] laplacianImagePixels = null;
			imagePixels = imageDataArray_[z];
			laplacianImagePixels = filteredImageDataArray_[z];
			for (int x = 0; x < width_; x++) {
				fy_val = 0;
				fyLap_val = 0;
				for (int y = 0; y < height_; y++) {
					int index = x + width_ * y;
					fy_val += imagePixels[index];
					fyLap_val += laplacianImagePixels[index];
					preintegratedImageDataArray_[z][index] = fy_val;
					preintegratedFilteredImageDataArray_[z][index] = fyLap_val;
				}
			}
			pFrame.incPosition();
		}
		
		//imageDataArray_ = null;
		filteredImageDataArray_ = null;
		pFrame.close();
		isLUTUpToDate_ = true;
	}

	// ----------------------------------------------------------------------------

	/** If <code>true</code>, the LUTs are consistent with the input parameters. */
	public boolean isLUTUpToDate() {
		return isLUTUpToDate_;
	}

	// ----------------------------------------------------------------------------

	/** */
	@Override
	public String toString() {
		return new String("[LUT container: sigma = " + sigma_
				+ ", sequence title = " + sequence_.getName()
				+ ", channel number = " + channelNumber_
				+ ", LUT up to date = " + isLUTUpToDate_ + "]");
	}

	// ============================================================================
	// SETTERS AND GETTERS

	/** Sets the active channel. */
	public void setChannelNumber(int channelNumber) {
		isLUTUpToDate_ = false;
		channelNumber_ = channelNumber;
	}

	// ----------------------------------------------------------------------------

	/** Returns the active channel. */
	public int getChannelNumber() {
		return channelNumber_;
	}

	// ----------------------------------------------------------------------------

	/** Sets the standard deviation of the Gaussian kernel used for smoothing. */
	public void setSigma(double sigma) {
		isLUTUpToDate_ = false;
		if (sigma < 0) {
			sigma_ = 0;
		} else {
			sigma_ = sigma;
		}
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns the standard deviation of the Gaussian kernel used for smoothing.
	 */
	public double getSigma() {
		return sigma_;
	}

	// ----------------------------------------------------------------------------

	/** Sets the active image from where the data is extracted. */
	public void setSequence(Sequence sequence) {
		isLUTUpToDate_ = false;
		sequence_ = sequence;
	}

	// ----------------------------------------------------------------------------

	/** Returns the image from where the image data is extracted. */
	public Sequence getOriginalSequence() {
		return sequence_;
	}

	// ----------------------------------------------------------------------------

	/** Returns the with in pixels of the image. */
	public int getImageWidth() {
		return width_;
	}

	// ----------------------------------------------------------------------------

	/** Returns the height in pixels of the image. */
	public int getImageHeight() {
		return height_;
	}

	// ----------------------------------------------------------------------------

	/** Returns the depth in pixels of the image. */
	public int getImageDepth() {
		return depth_;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a two dimensional array containing the pixel values of the
	 * original image of the active channel.
	 */
	public double[][] getImageDataArray() {
		return imageDataArray_;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a two dimensional array containing the pixel values of the image
	 * after applying a smoothed Laplacian filter of the active channel.
	 */
	public double[][] getFilteredImageDataArray() {
		return filteredImageDataArray_;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a one dimensional array containing the pixel values of the
	 * original image after preintegration of the active channel.
	 */
	public double[][] getPreintegratedImageDataArray() {
		return preintegratedImageDataArray_;
	}

	// ----------------------------------------------------------------------------

	/**
	 * Returns a one dimensional array containing the pixel values of the image
	 * after applying a smoohted Laplacian filter after preintegration of the
	 * active channel.
	 */
	public double[][] getPreintegratedFilteredImageDataArray() {
		return preintegratedFilteredImageDataArray_;
	}
}
