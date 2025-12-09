/*====================================================================
| Version: June 28, 2011
\===================================================================*/

/*====================================================================
| Philippe Thevenaz
| Bldg. BM-Ecublens 4.137
| Biomedical Imaging Group/BIOE/STI
| Swiss Federal Institute of Technology Lausanne
| CH-1015 Lausanne
| Switzerland
|
| phone (CET): +41(21)693.51.61
| fax: +41(21)693.37.01
| RFC-822: philippe.thevenaz@epfl.ch
| X-400: /C=ch/A=400net/P=switch/O=epfl/S=thevenaz/G=philippe/
| URL: http://bigwww.epfl.ch/
\===================================================================*/

import ij.*;
import ij.gui.*;
import ij.plugin.*;
import ij.process.*;
import java.awt.*;
import java.awt.event.*;
import java.text.*;	
import java.util.*;

/*====================================================================
|	InterpolationContest_
\===================================================================*/

/*------------------------------------------------------------------*/
public class InterpolationContest_
	implements
		PlugIn

{ /* begin class InterpolationContest_ */

/*....................................................................
	Public variables
....................................................................*/

public static final double X_CENTER_PERTURBATION = 0.5;
public static final double Y_CENTER_PERTURBATION = 0.5;

/*....................................................................
	Private variables
....................................................................*/

private static final String[] interpolation = new String[4];
private static final double[] seconds = new double[4];
private static final long[] elapsedTime = new long[4];
private static final long[] pixels = new long[4];
private static final int[] rate = new int[4];
private static int rank = 0;

/*....................................................................
	Public methods
....................................................................*/

/*------------------------------------------------------------------*/
public void run (
	final String arg
) {
	Runtime.getRuntime().gc();
	ImagePlus[] grayscaleImageList = createGrayscaleImageList();
	if (grayscaleImageList.length < 4) {
		IJ.error("At least four grayscale images are required");
		return;
	}
	final interpolationContestDialog dialog = new interpolationContestDialog(IJ.getInstance(),
		grayscaleImageList);
	GUI.center(dialog);
	dialog.setVisible(true);
} /* end run */

/*------------------------------------------------------------------*/
public static void showTime (
	final ImagePlus imp,
	final long start,
	final String str,
	final int iterations
) {
	elapsedTime[rank] = System.currentTimeMillis() - start;
	pixels[rank] = imp.getWidth() * imp.getHeight() * imp.getStack().getSize() * iterations;
	interpolation[rank] = new String(str);
	rank++;
	if (4 <= rank) {
		rank = 0;
		final DecimalFormat df = new DecimalFormat("0.0");
		String rateStr;
		seconds[0] = (double)elapsedTime[0] / 4000.0;
		seconds[1] = seconds[0] + (double)(elapsedTime[1] - elapsedTime[0]) / 3000.0;
		seconds[2] = seconds[1] + (double)(elapsedTime[2] - elapsedTime[1]) / 2000.0;
		seconds[3] = seconds[2] + (double)(elapsedTime[3] - elapsedTime[2]) / 1000.0;
		for (int i = 0; (i < 4); i++) {
			rate[i] = (int)((double)pixels[i] / seconds[i]);
			if (rate[i] > 1000000000) {
				rateStr = "";
			}
			else if (rate[i] < 1000000) {
				rateStr = ", " + df.format(rate[i]) + " pixels/second";
			}
			else {
				rateStr = ", " + df.format(rate[i] / 1000000.0) + " million pixels/second";
			}
			IJ.log(interpolation[i]);
			IJ.log("  " + df.format(seconds[i]) + " seconds" + rateStr);
		}
	}
} /* showTime */
	
/*....................................................................
	Private methods
....................................................................*/

/*------------------------------------------------------------------*/
private ImagePlus[] createGrayscaleImageList (
) {
	final int[] windowList = WindowManager.getIDList();
	final Stack<ImagePlus> stack = new Stack<ImagePlus>();
	for (int k = 0; ((windowList != null) && (k < windowList.length)); k++) {
		final ImagePlus imp = WindowManager.getImage(windowList[k]);
		if ((imp != null) && ((imp.getType() == ImagePlus.GRAY8)
			|| (imp.getType() == ImagePlus.GRAY16)
			|| (imp.getType() == ImagePlus.GRAY32))) {
			stack.push(imp);
		}
	}
	final ImagePlus[] grayscaleImageList = new ImagePlus[stack.size()];
	int k = 0;
	while (!stack.isEmpty()) {
		grayscaleImageList[k++] = (ImagePlus)stack.pop();
	}
	return(grayscaleImageList);
} /* end createGrayscaleImageList */

} /* end class InterpolationContest_ */

/*====================================================================
|	interpolationContestCausal
\===================================================================*/

/*------------------------------------------------------------------*/
class interpolationContestCausal
	implements Runnable

{ /* begin class interpolationContestCausal */

/*....................................................................
	Private variables
....................................................................*/

private FloatProcessor fp;
private ImagePlus imp;
private Thread t;
private float inImg[];
private float outImg[];
private double halfHeight;
private double halfWidth;
private int iterations;
private int height;
private int width;

/*....................................................................
	Public methods
....................................................................*/

/*------------------------------------------------------------------*/
public Thread getThread (
) {
	return(t);
} /* end getThread */

/*------------------------------------------------------------------*/
public interpolationContestCausal (
) {
	t = new Thread(this);
	t.start();
} /* end interpolationContestCausal */

/*------------------------------------------------------------------*/
public void run (
) {
	try {
		synchronized(t) {
			t.wait();
		}
	}
	catch (InterruptedException e) {
	}
	imp.startTiming();
	if (1 < imp.getStackSize()) {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new StackConverter(imp).convertToGray32();
		}
	}
	else {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new ImageConverter(imp).convertToGray32();
		}
	}
	final ImageStack stack = imp.getStack();
	final int stackSize = stack.getSize();
	Undo.reset();
	imp.setSlice(1);
	imp.getProcessor().resetMinAndMax();
	imp.updateAndRepaintWindow();
	for (int i = 1; (i <= stackSize); i++) {
		inImg = (float[])stack.getProcessor(i).getPixels();
		imp.setSlice(i);
		causal();
		imp.updateAndRepaintWindow();
	}
	if (2 <= stackSize) {
		imp.setSlice(1);
		imp.updateAndRepaintWindow();
	}
	IJ.showTime(imp, imp.getStartTime(), "Causal interpolation: ");
	InterpolationContest_.showTime(imp, imp.getStartTime(), "Causal interpolation: ",
		iterations);
	ImageWindow win = imp.getWindow();
	if (win != null) {
		win.running = false;
	}
} /* end run */

/*------------------------------------------------------------------*/
public void setup (
	ImagePlus imp,
	int iterations
) {
	this.imp = imp;
	this.iterations = iterations;
	height = imp.getProcessor().getHeight();
	width = imp.getProcessor().getWidth();
	halfHeight = 0.5 * (double)(height - 1);
	halfWidth = 0.5 * (double)(width - 1);
	fp = new FloatProcessor(width, height);
	outImg = (float[])fp.getPixels();
	imp.setTitle(imp.getTitle() + " - Causal");
} /* end setup */

/*....................................................................
	Private methods
....................................................................*/

private final double[] xWeight = new double[4];
private final double[] yWeight = new double[4];
private final int[] xIndex = new int[4];
private final int[] yIndex = new int[4];
private double result, partialResult;
private double x, y;
private double z1, z12, xz1, yz1, z2;
private int base;
private int p, q;

/*------------------------------------------------------------------*/
private void causal (
) {
	double theta = 2.0 * Math.PI / (double)iterations;
	double totalAngle = 0.0;
	double c = Math.cos(theta);
	double s = Math.sin(theta);
	double yx, yy, dx, dy;
	int k;
	for (int i = 0; (i < iterations); i++) {
		toCoefficients();
		k = 0;
		totalAngle += theta;
		dx = InterpolationContest_.X_CENTER_PERTURBATION * Math.cos(totalAngle)
			- InterpolationContest_.Y_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ halfWidth * (1.0 - c) + halfHeight * s;
		dy = InterpolationContest_.X_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ InterpolationContest_.Y_CENTER_PERTURBATION * Math.cos(totalAngle)
			- halfWidth * s + halfHeight * (1.0 - c);
		for (int v = 0; (v < height); v++) {
			yx = dx - s * (double)v;
			yy = dy + c * (double)v;
			for (int u = 0; (u < width); u++) {
				x = yx + c * (double)u;
				y = yy + s * (double)u;
				p = (0.0 <= x) ? ((int)(x + 0.5)) : ((int)(x - 0.5));
				q = (0.0 <= y) ? ((int)(y + 0.5)) : ((int)(y - 0.5));
				if ((0 <= p) && (p < width) && (0 <= q) && (q < height)) {
					xIndexes();
					yIndexes();
					x -= (0.0 <= x) ? ((int)x) : ((int)x - 1);
					y -= (0.0 <= y) ? ((int)y) : ((int)y - 1);
					xWeights();
					yWeights();
					outImg[k++] = (float)interpolate();
				}
				else {
					outImg[k++] = 0.0F;
				}
			}
		}
		float[] img = outImg;
		outImg = inImg;
		inImg = img;
		imp.updateAndDraw();
	}
} /* end causal */

/*------------------------------------------------------------------*/
private double interpolate (
) {
	result = 0.0;
	for (int j = 0; (j < 4); j++) {
		partialResult = 0.0;
		base = yIndex[j];
		for (int i = 0; (i < 4); i++) {
			partialResult += xWeight[i] * (double)inImg[base + xIndex[i]];
		}
		result += yWeight[j] * partialResult;
	}
	return(result);
} /* end interpolate */

/*------------------------------------------------------------------*/
private void toCoefficients (
) {
	double c;
	int k = 0;
	for (int v = 0; (v < height); v++) {
		c = 1.5 * (double)inImg[k];
		outImg[k++] = (float)c;
		for (int u = 1; (u < width); u++) {
			c = -0.5 * c + 1.5 * (double)inImg[k];
			outImg[k++] = (float)c;
		}
	}
	for (int u = 0; (u < width); u++) {
		k = u;
		c = 1.5 * (double)outImg[k];
		outImg[k] = (float)c;
		for (int v = 1; (v < height); v++) {
			k += width;
			c = -0.5 * c + 1.5 * (double)outImg[k];
			outImg[k] = (float)c;
		}
	}
	float[] img = outImg;
	outImg = inImg;
	inImg = img;
} /* toCoefficients */

/*------------------------------------------------------------------*/
private void xIndexes (
) {
	p = (0.0 <= x) ? ((int)x + 2) : ((int)x + 1);
	for (int k = 0; (k < 4); p--, k++) {
		xIndex[k] = (width <= p) ? (-1) : (p);
	}
} /* xIndexes */

/*------------------------------------------------------------------*/
private void xWeights (
) {
	z1 = x - 1.0;
	z12 = z1 * z1;
	xz1 = x * z1;
	z2 = x - 2.0;
	if (xIndex[0] < 0) {
		xIndex[0] = 0;
		xWeight[0] = 0.0;
	}
	else {
		xWeight[0] = (1.0 / 6.0) * x * xz1;
	}
	if (xIndex[1] < 0) {
		xIndex[1] = 0;
		xWeight[1] = 0.0;
	}
	else {
		xWeight[1] = 0.5 * x * (4.0 / 3.0 - z12);
	}
	if (xIndex[2] < 0) {
		xIndex[2] = 0;
		xWeight[2] = 0.0;
	}
	else {
		xWeight[2] = z2 * (0.5 * xz1 - 1.0 / 3.0);
	}
	if (xIndex[3] < 0) {
		xIndex[3] = 0;
		xWeight[3] = 0.0;
	}
	else {
		xWeight[3] = (-1.0 / 6.0) * z12 * z2;
	}
} /* xWeights */

/*------------------------------------------------------------------*/
private void yIndexes (
) {
	q = (0.0 <= y) ? ((int)y + 2) : ((int)y + 1);
	for (int k = 0; (k < 4); q--, k++) {
		yIndex[k] = (height <= q) ? (-1) : (q * width);
	}
} /* yIndexes */

/*------------------------------------------------------------------*/
private void yWeights (
) {
	z1 = y - 1.0;
	z12 = z1 * z1;
	yz1 = y * z1;
	z2 = y - 2.0;
	if (yIndex[0] < 0) {
		yIndex[0] = 0;
		yWeight[0] = 0.0;
	}
	else {
		yWeight[0] = (1.0 / 6.0) * y * yz1;
	}
	if (yIndex[1] < 0) {
		yIndex[1] = 0;
		yWeight[1] = 0.0;
	}
	else {
		yWeight[1] = 0.5 * y * (4.0 / 3.0 - z12);
	}
	if (yIndex[2] < 0) {
		yIndex[2] = 0;
		yWeight[2] = 0.0;
	}
	else {
		yWeight[2] = z2 * (0.5 * yz1 - 1.0 / 3.0);
	}
	if (yIndex[3] < 0) {
		yIndex[3] = 0;
		yWeight[3] = 0.0;
	}
	else {
		yWeight[3] = (-1.0 / 6.0) * z12 * z2;
	}
} /* yWeights */

} /* end class interpolationContestCausal */

/*====================================================================
|	interpolationContestCredits
\===================================================================*/

/*------------------------------------------------------------------*/
class interpolationContestCredits
	extends
		Dialog

{ /* begin class interpolationContestCredits */

/*....................................................................
	Private variables
....................................................................*/

private static final long serialVersionUID = 1L;

/*....................................................................
	Public methods
....................................................................*/

/*------------------------------------------------------------------*/
public Insets getInsets (
) {
	return(new Insets(0, 20, 20, 20));
} /* end getInsets */

/*------------------------------------------------------------------*/
public interpolationContestCredits (
	final Frame parentWindow
) {
	super(parentWindow, "InterpolationContest", true);
	setLayout(new BorderLayout(0, 20));
	final Label separation = new Label("");
	final Panel buttonPanel = new Panel();
	buttonPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Button doneButton = new Button("Done");
	doneButton.addActionListener(
		new ActionListener (
		) {
			public void actionPerformed (
				final ActionEvent ae
			) {
				if (ae.getActionCommand().equals("Done")) {
					dispose();
				}
			}
		}
	);
	buttonPanel.add(doneButton);
	final TextArea text = new TextArea(8, 72);
	text.setEditable(false);
	text.append(" Relevant on-line publications are available at\n");
	text.append(" http://bigwww.epfl.ch/publications/\n");
	text.append("\n");
	text.append(" You'll be free to use this software for research purposes, but\n");
	text.append(" you should not redistribute it without our consent. In addition,\n");
	text.append(" we expect you to include a citation or acknowledgment whenever\n");
	text.append(" you present or publish results that are based on it.\n");
	add("North", separation);
	add("Center", text);
	add("South", buttonPanel);
	pack();
} /* end interpolationContestCredits */

} /* end class interpolationContestCredits */

/*====================================================================
|	interpolationContestDialog
\===================================================================*/

/*------------------------------------------------------------------*/
class interpolationContestDialog
	extends
		Dialog
	implements
		ActionListener

{ /* begin class interpolationContestDialog */

/*....................................................................
	Private variables
....................................................................*/

private final interpolationContestCausal causal
	= new interpolationContestCausal();
private final interpolationContestKeys keys
	= new interpolationContestKeys();
private final interpolationContestLinear linear
	= new interpolationContestLinear();
private final interpolationContestShiftedLinear shiftedLinear
	= new interpolationContestShiftedLinear();
private ImagePlus[] grayscaleImageList;
private TextField iterationTextField;
private int causalChoiceIndex = 0;
private int keysChoiceIndex = 1;
private int linearChoiceIndex = 2;
private int shiftedLinearChoiceIndex = 3;
private int iterations = 12;
private static final long serialVersionUID = 1L;

/*....................................................................
	Public methods
....................................................................*/

public void actionPerformed (
	final ActionEvent ae
) {
	if (ae.getActionCommand().equals("Cancel")) {
		dispose();
	}
	else if (ae.getActionCommand().equals("Go")) {
		try {
			int attempt = Integer.parseInt(iterationTextField.getText());
			if (attempt < 3) {
				throw(new NumberFormatException());
			}
			else {
				iterations = attempt;
			}
		} catch (NumberFormatException n) {
			IJ.error("At least 3 iterations required");
			return;
		}
		dispose();
		causal.setup(grayscaleImageList[causalChoiceIndex], iterations);
		keys.setup(grayscaleImageList[keysChoiceIndex], iterations);
		linear.setup(grayscaleImageList[linearChoiceIndex], iterations);
		shiftedLinear.setup(grayscaleImageList[shiftedLinearChoiceIndex], iterations);
		causal.getThread().interrupt();
		keys.getThread().interrupt();
		linear.getThread().interrupt();
		shiftedLinear.getThread().interrupt();
	}
	else if (ae.getActionCommand().equals("Credits...")) {
		final interpolationContestCredits dialog
			= new interpolationContestCredits(IJ.getInstance());
		GUI.center(dialog);
		dialog.setVisible(true);
	}
} /* end actionPerformed */

/*------------------------------------------------------------------*/
public Insets getInsets (
) {
	return(new Insets(0, 20, 20, 20));
} /* end getInsets */

/*------------------------------------------------------------------*/
public interpolationContestDialog (
	final Frame parentWindow,
	final ImagePlus[] grayscaleImageList
) {
	super(parentWindow, "InterpolationContest", false);
	this.grayscaleImageList = grayscaleImageList;
	setLayout(new GridLayout(0, 1));
	final Choice keysChoice = new Choice();
	final Panel keysPanel = new Panel();
	keysPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Label keysLabel = new Label("Keys interpolation: ");
	addImageList(keysChoice);
	keysChoice.select(keysChoiceIndex);
	keysPanel.add(keysLabel);
	keysPanel.add(keysChoice);
	final Choice causalChoice = new Choice();
	final Panel causalPanel = new Panel();
	causalPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Label causalLabel = new Label("Causal interpolation: ");
	addImageList(causalChoice);
	causalChoice.select(causalChoiceIndex);
	causalPanel.add(causalLabel);
	causalPanel.add(causalChoice);
	final Choice linearChoice = new Choice();
	final Panel linearPanel = new Panel();
	linearPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Label linearLabel = new Label("Linear interpolation: ");
	addImageList(linearChoice);
	linearChoice.select(linearChoiceIndex);
	linearPanel.add(linearLabel);
	linearPanel.add(linearChoice);
	final Choice shiftedLinearChoice = new Choice();
	final Panel shiftedLinearPanel = new Panel();
	shiftedLinearPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Label shiftedLinearLabel = new Label("Shifted-linear interpolation: ");
	addImageList(shiftedLinearChoice);
	shiftedLinearChoice.select(shiftedLinearChoiceIndex);
	shiftedLinearPanel.add(shiftedLinearLabel);
	shiftedLinearPanel.add(shiftedLinearChoice);
	causalChoice.addItemListener(
		new ItemListener (
		) {
			public void itemStateChanged (
				final ItemEvent ie
			) {
				final int newChoiceIndex = causalChoice.getSelectedIndex();
				if (causalChoiceIndex != newChoiceIndex) {
					if ((keysChoiceIndex != newChoiceIndex)
						&& (linearChoiceIndex != newChoiceIndex)
						&& (shiftedLinearChoiceIndex != newChoiceIndex)) {
						causalChoiceIndex = newChoiceIndex;
					}
					else {
						if (newChoiceIndex == keysChoiceIndex) {
							keysChoiceIndex = causalChoiceIndex;
							keysChoice.select(keysChoiceIndex);
						}
						else if (newChoiceIndex == linearChoiceIndex) {
							linearChoiceIndex = causalChoiceIndex;
							linearChoice.select(linearChoiceIndex);
						}
						else if (newChoiceIndex == shiftedLinearChoiceIndex) {
							shiftedLinearChoiceIndex = causalChoiceIndex;
							shiftedLinearChoice.select(shiftedLinearChoiceIndex);
						}
						causalChoiceIndex = newChoiceIndex;
					}
				}
				repaint();
			}
		}
	);
	keysChoice.addItemListener(
		new ItemListener (
		) {
			public void itemStateChanged (
				final ItemEvent ie
			) {
				final int newChoiceIndex = keysChoice.getSelectedIndex();
				if (keysChoiceIndex != newChoiceIndex) {
					if ((causalChoiceIndex != newChoiceIndex)
						&& (linearChoiceIndex != newChoiceIndex)
						&& (shiftedLinearChoiceIndex != newChoiceIndex)) {
						keysChoiceIndex = newChoiceIndex;
					}
					else {
						if (newChoiceIndex == causalChoiceIndex) {
							causalChoiceIndex = keysChoiceIndex;
							causalChoice.select(causalChoiceIndex);
						}
						else if (newChoiceIndex == linearChoiceIndex) {
							linearChoiceIndex = keysChoiceIndex;
							linearChoice.select(linearChoiceIndex);
						}
						else if (newChoiceIndex == shiftedLinearChoiceIndex) {
							shiftedLinearChoiceIndex = keysChoiceIndex;
							shiftedLinearChoice.select(shiftedLinearChoiceIndex);
						}
						keysChoiceIndex = newChoiceIndex;
					}
				}
				repaint();
			}
		}
	);
	linearChoice.addItemListener(
		new ItemListener (
		) {
			public void itemStateChanged (
				final ItemEvent ie
			) {
				final int newChoiceIndex = linearChoice.getSelectedIndex();
				if (linearChoiceIndex != newChoiceIndex) {
					if ((causalChoiceIndex != newChoiceIndex)
						&& (keysChoiceIndex != newChoiceIndex)
						&& (shiftedLinearChoiceIndex != newChoiceIndex)) {
						linearChoiceIndex = newChoiceIndex;
					}
					else {
						if (newChoiceIndex == causalChoiceIndex) {
							causalChoiceIndex = linearChoiceIndex;
							causalChoice.select(causalChoiceIndex);
						}
						else if (newChoiceIndex == keysChoiceIndex) {
							keysChoiceIndex = linearChoiceIndex;
							keysChoice.select(keysChoiceIndex);
						}
						else if (newChoiceIndex == shiftedLinearChoiceIndex) {
							shiftedLinearChoiceIndex = linearChoiceIndex;
							shiftedLinearChoice.select(shiftedLinearChoiceIndex);
						}
						linearChoiceIndex = newChoiceIndex;
					}
				}
				repaint();
			}
		}
	);
	shiftedLinearChoice.addItemListener(
		new ItemListener (
		) {
			public void itemStateChanged (
				final ItemEvent ie
			) {
				final int newChoiceIndex = shiftedLinearChoice.getSelectedIndex();
				if (shiftedLinearChoiceIndex != newChoiceIndex) {
					if ((causalChoiceIndex != newChoiceIndex)
						&& (keysChoiceIndex != newChoiceIndex)
						&& (linearChoiceIndex != newChoiceIndex)) {
						shiftedLinearChoiceIndex = newChoiceIndex;
					}
					else {
						if (newChoiceIndex == causalChoiceIndex) {
							causalChoiceIndex = shiftedLinearChoiceIndex;
							causalChoice.select(causalChoiceIndex);
						}
						else if (newChoiceIndex == keysChoiceIndex) {
							keysChoiceIndex = shiftedLinearChoiceIndex;
							keysChoice.select(keysChoiceIndex);
						}
						else if (newChoiceIndex == linearChoiceIndex) {
							linearChoiceIndex = shiftedLinearChoiceIndex;
							linearChoice.select(linearChoiceIndex);
						}
						shiftedLinearChoiceIndex = newChoiceIndex;
					}
				}
				repaint();
			}
		}
	);
	final Panel iterationPanel = new Panel();
	iterationPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Label iterationLabel = new Label("Number of rotations: ");
	iterationTextField = new TextField("" + iterations, 3);
	iterationPanel.add(iterationLabel);
	iterationPanel.add(iterationTextField);
	final Panel buttonPanel = new Panel();
	buttonPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Button cancelButton = new Button("Cancel");
	cancelButton.addActionListener(this);
	final Button goButton = new Button("Go");
	goButton.addActionListener(this);
	buttonPanel.add(cancelButton);
	buttonPanel.add(goButton);
	final Panel creditsPanel = new Panel();
	creditsPanel.setLayout(new FlowLayout(FlowLayout.CENTER));
	final Button creditsButton = new Button("Credits...");
	creditsButton.addActionListener(this);
	creditsPanel.add(creditsButton);
	final Label separation1 = new Label("");
	final Label separation2 = new Label("");
	final Label separation3 = new Label("");
	add(separation1);
	add(causalPanel);
	add(keysPanel);
	add(linearPanel);
	add(shiftedLinearPanel);
	add(separation2);
	add(iterationPanel);
	add(separation3);
	add(buttonPanel);
	add(creditsPanel);
	pack();
} /* end interpolationContestDialog */

/*....................................................................
	Private methods
....................................................................*/

/*------------------------------------------------------------------*/
private void addImageList (
	final Choice choice
) {
	for (int k = 0; (k < grayscaleImageList.length); k++) {
		choice.add(grayscaleImageList[k].getTitle());
	}
} /* end addImageList */

} /* end class interpolationContestDialog */

/*====================================================================
|	interpolationContestKeys
\===================================================================*/

/*------------------------------------------------------------------*/
class interpolationContestKeys
	implements Runnable

{ /* begin class interpolationContestKeys */

/*....................................................................
	Private variables
....................................................................*/

private FloatProcessor fp;
private ImagePlus imp;
private Thread t;
private float inImg[];
private float outImg[];
private double halfHeight;
private double halfWidth;
private int iterations;
private int height, doubleHeight;
private int width, doubleWidth;

/*....................................................................
	Public methods
....................................................................*/

/*------------------------------------------------------------------*/
public Thread getThread (
) {
	return(t);
} /* end getThread */

/*------------------------------------------------------------------*/
public interpolationContestKeys (
) {
	t = new Thread(this);
	t.start();
} /* end interpolationContestKeys */

/*------------------------------------------------------------------*/
public void run (
) {
	try {
		synchronized(t) {
			t.wait();
		}
	}
	catch (InterruptedException e) {
	}
	imp.startTiming();
	if (1 < imp.getStackSize()) {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new StackConverter(imp).convertToGray32();
		}
	}
	else {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new ImageConverter(imp).convertToGray32();
		}
	}
	final ImageStack stack = imp.getStack();
	final int stackSize = stack.getSize();
	Undo.reset();
	imp.setSlice(1);
	imp.getProcessor().resetMinAndMax();
	imp.updateAndRepaintWindow();
	for (int i = 1; (i <= stackSize); i++) {
		inImg = (float[])stack.getProcessor(i).getPixels();
		imp.setSlice(i);
		keys();
		imp.updateAndRepaintWindow();
	}
	if (2 <= stackSize) {
		imp.setSlice(1);
		imp.updateAndRepaintWindow();
	}
	IJ.showTime(imp, imp.getStartTime(), "Keys interpolation: ");
	InterpolationContest_.showTime(imp, imp.getStartTime(), "Keys interpolation: ",
		iterations);
	ImageWindow win = imp.getWindow();
	if (win != null) {
		win.running = false;
	}
} /* end run */

/*------------------------------------------------------------------*/
public void setup (
	ImagePlus imp,
	int iterations
) {
	this.imp = imp;
	this.iterations = iterations;
	height = imp.getProcessor().getHeight();
	width = imp.getProcessor().getWidth();
	halfHeight = 0.5 * (double)(height - 1);
	halfWidth = 0.5 * (double)(width - 1);
	doubleHeight = 2 * height;
	doubleWidth = 2 * width;
	fp = new FloatProcessor(width, height);
	outImg = (float[])fp.getPixels();
	imp.setTitle(imp.getTitle() + " - Keys");
} /* end setup */

/*....................................................................
	Private methods
....................................................................*/

private final double[] xWeight = new double[4];
private final double[] yWeight = new double[4];
private final int[] xIndex = new int[4];
private final int[] yIndex = new int[4];
private double result, partialResult;
private double x, y, z, z2;
private int base;
private int p, q;

/*------------------------------------------------------------------*/
private double interpolate (
) {
	result = 0.0;
	for (int j = 0; (j < 4); j++) {
		partialResult = 0.0;
		base = yIndex[j];
		for (int i = 0; (i < 4); i++) {
			partialResult += xWeight[i] * (double)inImg[base + xIndex[i]];
		}
		result += yWeight[j] * partialResult;
	}
	return(result);
} /* end interpolate */

/*------------------------------------------------------------------*/
private void keys (
) {
	double theta = 2.0 * Math.PI / (double)iterations;
	double totalAngle = 0.0;
	double c = Math.cos(theta);
	double s = Math.sin(theta);
	double yx, yy, dx, dy;
	int k;
	for (int i = 0; (i < iterations); i++) {
		k = 0;
		totalAngle += theta;
		dx = InterpolationContest_.X_CENTER_PERTURBATION * Math.cos(totalAngle)
			- InterpolationContest_.Y_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ halfWidth * (1.0 - c) + halfHeight * s;
		dy = InterpolationContest_.X_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ InterpolationContest_.Y_CENTER_PERTURBATION * Math.cos(totalAngle)
			- halfWidth * s + halfHeight * (1.0 - c);
		for (int v = 0; (v < height); v++) {
			yx = dx - s * (double)v;
			yy = dy + c * (double)v;
			for (int u = 0; (u < width); u++) {
				x = yx + c * (double)u;
				y = yy + s * (double)u;
				p = (0.0 <= x) ? ((int)(x + 0.5)) : ((int)(x - 0.5));
				q = (0.0 <= y) ? ((int)(y + 0.5)) : ((int)(y - 0.5));
				if ((0 <= p) && (p < width) && (0 <= q) && (q < height)) {
					xIndexes();
					yIndexes();
					x -= (0.0 <= x) ? ((int)x) : ((int)x - 1);
					y -= (0.0 <= y) ? ((int)y) : ((int)y - 1);
					xWeights();
					yWeights();
					outImg[k++] = (float)interpolate();
				}
				else {
					outImg[k++] = 0.0F;
				}
			}
		}
		System.arraycopy(outImg, 0, inImg, 0, width * height);
		imp.updateAndDraw();
	}
} /* end keys */

/*------------------------------------------------------------------*/
private void xIndexes (
) {
	p = (0.0 <= x) ? ((int)x + 2) : ((int)x + 1);
	for (int k = 0; (k < 4); p--, k++) {
		q = (p < 0) ? (-1 - p) : (p);
		if (doubleWidth <= q) {
			q -= doubleWidth * (q / doubleWidth);
		}
		xIndex[k] = (width <= q) ? (doubleWidth - 1 - q) : (q);
	}
} /* xIndexes */

/*------------------------------------------------------------------*/
private void xWeights (
) {
	z = 1.0 - x;
	z2 = x * x;
	xWeight[0] = -0.5 * z * z2;
	xWeight[3] = -0.5 * x * z * z;
	xWeight[2] = z2 * (1.5 * x - 2.5) + 1.0;
	xWeight[1] = 1.0 - xWeight[0] - xWeight[2] - xWeight[3];
} /* xWeights */

/*------------------------------------------------------------------*/
private void yIndexes (
) {
	p = (0.0 <= y) ? ((int)y + 2) : ((int)y + 1);
	for (int k = 0; (k < 4); p--, k++) {
		q = (p < 0) ? (-1 - p) : (p);
		if (doubleHeight <= q) {
			q -= doubleHeight * (q / doubleHeight);
		}
		yIndex[k] = (height <= q) ? ((doubleHeight - 1 - q) * width) : (q * width);
	}
} /* yIndexes */

/*------------------------------------------------------------------*/
private void yWeights (
) {
	z = 1.0 - y;
	z2 = y * y;
	yWeight[0] = -0.5 * z * z2;
	yWeight[3] = -0.5 * y * z * z;
	yWeight[2] = z2 * (1.5 * y - 2.5) + 1.0;
	yWeight[1] = 1.0 - yWeight[0] - yWeight[2] - yWeight[3];
} /* yWeights */

} /* end class interpolationContestKeys */

/*====================================================================
|	interpolationContestLinear
\===================================================================*/

/*------------------------------------------------------------------*/
class interpolationContestLinear
	implements Runnable

{ /* begin class interpolationContestLinear */

/*....................................................................
	Private variables
....................................................................*/

private FloatProcessor fp;
private ImagePlus imp;
private Thread t;
private float inImg[];
private float outImg[];
private double halfHeight;
private double halfWidth;
private int iterations;
private int height;
private int width;

/*....................................................................
	Public methods
....................................................................*/

/*------------------------------------------------------------------*/
public Thread getThread (
) {
	return(t);
} /* end getThread */

/*------------------------------------------------------------------*/
public interpolationContestLinear (
) {
	t = new Thread(this);
	t.start();
} /* end interpolationContestLinear */

/*------------------------------------------------------------------*/
public void run (
) {
	try {
		synchronized(t) {
			t.wait();
		}
	}
	catch (InterruptedException e) {
	}
	imp.startTiming();
	if (1 < imp.getStackSize()) {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new StackConverter(imp).convertToGray32();
		}
	}
	else {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new ImageConverter(imp).convertToGray32();
		}
	}
	final ImageStack stack = imp.getStack();
	final int stackSize = stack.getSize();
	Undo.reset();
	imp.setSlice(1);
	imp.getProcessor().resetMinAndMax();
	imp.updateAndRepaintWindow();
	for (int i = 1; (i <= stackSize); i++) {
		inImg = (float[])stack.getProcessor(i).getPixels();
		imp.setSlice(i);
		linear();
		imp.updateAndRepaintWindow();
	}
	if (2 <= stackSize) {
		imp.setSlice(1);
		imp.updateAndRepaintWindow();
	}
	IJ.showTime(imp, imp.getStartTime(), "Linear interpolation: ");
	InterpolationContest_.showTime(imp, imp.getStartTime(), "Linear interpolation: ",
		iterations);
	ImageWindow win = imp.getWindow();
	if (win != null) {
		win.running = false;
	}
} /* end run */

/*------------------------------------------------------------------*/
public void setup (
	ImagePlus imp,
	int iterations
) {
	this.imp = imp;
	this.iterations = iterations;
	height = imp.getProcessor().getHeight();
	width = imp.getProcessor().getWidth();
	halfHeight = 0.5 * (double)(height - 1);
	halfWidth = 0.5 * (double)(width - 1);
	fp = new FloatProcessor(width, height);
	outImg = (float[])fp.getPixels();
	imp.setTitle(imp.getTitle() + " - Linear");
} /* end setup */

/*....................................................................
	Private methods
....................................................................*/

private double xWeight0, xWeight1;
private double result, partialResult;
private double x, y;
private int xIndex0, xIndex1, yIndex0, yIndex1;
private int p, q;

/*------------------------------------------------------------------*/
private void linear (
) {
	double theta = 2.0 * Math.PI / (double)iterations;
	double totalAngle = 0.0;
	double c = Math.cos(theta);
	double s = Math.sin(theta);
	double yx, yy, dx, dy;
	int k;
	for (int i = 0; (i < iterations); i++) {
		k = 0;
		totalAngle += theta;
		dx = InterpolationContest_.X_CENTER_PERTURBATION * Math.cos(totalAngle)
			- InterpolationContest_.Y_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ halfWidth * (1.0 - c) + halfHeight * s;
		dy = InterpolationContest_.X_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ InterpolationContest_.Y_CENTER_PERTURBATION * Math.cos(totalAngle)
			- halfWidth * s + halfHeight * (1.0 - c);
		for (int v = 0; (v < height); v++) {
			yx = dx - s * (double)v;
			yy = dy + c * (double)v;
			for (int u = 0; (u < width); u++) {
				x = yx + c * (double)u;
				y = yy + s * (double)u;
				p = (0.0 <= x) ? ((int)x) : ((int)x - 1);
				q = (0.0 <= y) ? ((int)y) : ((int)y - 1);
				if ((-1 <= p) && (p < width) && (-1 <= q) && (q < height)) {
					if (0 <= p) {
						xIndex1 = p++;
					}
					else {
						xIndex1 = ++p;
					}
					if (0 <= q) {
						yIndex1 = q++ * width;
					}
					else {
						yIndex1 = ++q;
					}
					if (p < width) {
						xIndex0 = p;
					}
					else {
						xIndex0 = --p;
					}
					if (q < height) {
						yIndex0 = q * width;
					}
					else {
						yIndex0 = --q * width;
					}
					x -= (0.0 <= x) ? ((int)x) : ((int)x - 1);
					y -= (0.0 <= y) ? ((int)y) : ((int)y - 1);
					xWeight0 = x;
					xWeight1 = 1.0 - x;
					partialResult = xWeight0 * (double)inImg[yIndex0 + xIndex0];
					partialResult += xWeight1 * (double)inImg[yIndex0 + xIndex1];
					result = y * partialResult;
					partialResult = xWeight0 * (double)inImg[yIndex1 + xIndex0];
					partialResult += xWeight1 * (double)inImg[yIndex1 + xIndex1];
					result += (1.0 - y) * partialResult;
					outImg[k++] = (float)result;
				}
				else {
					outImg[k++] = 0.0F;
				}
			}
		}
		System.arraycopy(outImg, 0, inImg, 0, width * height);

		imp.updateAndDraw();
	}
} /* end linear */

} /* end class interpolationContestLinear */

/*====================================================================
|	interpolationContestShiftedLinear
\===================================================================*/

/*------------------------------------------------------------------*/
class interpolationContestShiftedLinear
	implements Runnable

{ /* begin class interpolationContestShiftedLinear */

/*....................................................................
	Private variables
....................................................................*/

private static final double shift = 0.5 * (1.0 - Math.sqrt(1.0 / 3.0));
private FloatProcessor fp;
private ImagePlus imp;
private Thread t;
private float inImg[];
private float outImg[];
private double halfHeight;
private double halfWidth;
private int iterations;
private int height;
private int width;

/*....................................................................
	Public methods
....................................................................*/

/*------------------------------------------------------------------*/
public Thread getThread (
) {
	return(t);
} /* end getThread */

/*------------------------------------------------------------------*/
public interpolationContestShiftedLinear (
) {
	t = new Thread(this);
	t.start();
} /* end interpolationContestShiftedLinear */

/*------------------------------------------------------------------*/
public void run (
) {
	try {
		synchronized(t) {
			t.wait();
		}
	}
	catch (InterruptedException e) {
	}
	imp.startTiming();
	if (1 < imp.getStackSize()) {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new StackConverter(imp).convertToGray32();
		}
	}
	else {
		if (!(imp.getProcessor().getPixels() instanceof float[])) {
			new ImageConverter(imp).convertToGray32();
		}
	}
	final ImageStack stack = imp.getStack();
	final int stackSize = stack.getSize();
	Undo.reset();
	imp.setSlice(1);
	imp.getProcessor().resetMinAndMax();
	imp.updateAndRepaintWindow();
	for (int i = 1; (i <= stackSize); i++) {
		inImg = (float[])stack.getProcessor(i).getPixels();
		imp.setSlice(i);
		shiftedLinear();
		imp.updateAndRepaintWindow();
	}
	if (2 <= stackSize) {
		imp.setSlice(1);
		imp.updateAndRepaintWindow();
	}
	IJ.showTime(imp, imp.getStartTime(), "Shifted-linear interpolation: ");
	InterpolationContest_.showTime(imp, imp.getStartTime(), "Shifted-linear interpolation: ",
		iterations);
	ImageWindow win = imp.getWindow();
	if (win != null) {
		win.running = false;
	}
} /* end run */

/*------------------------------------------------------------------*/
public void setup (
	ImagePlus imp,
	int iterations
) {
	this.imp = imp;
	this.iterations = iterations;
	height = imp.getProcessor().getHeight();
	width = imp.getProcessor().getWidth();
	halfHeight = 0.5 * (double)(height - 1);
	halfWidth = 0.5 * (double)(width - 1);
	fp = new FloatProcessor(width, height);
	outImg = (float[])fp.getPixels();
	imp.setTitle(imp.getTitle() + " - Shifted-linear");
} /* end setup */

/*....................................................................
	Private methods
....................................................................*/

private double xWeight0, xWeight1;
private double result, partialResult;
private double x, y;
private int xIndex0, xIndex1, yIndex0, yIndex1;
private int p, q;

/*------------------------------------------------------------------*/
private void shiftedLinear (
) {
	double theta = 2.0 * Math.PI / (double)iterations;
	double totalAngle = 0.0;
	double c = Math.cos(theta);
	double s = Math.sin(theta);
	double yx, yy, dx, dy;
	int k;
	for (int i = 0; (i < iterations); i++) {
		toCoefficients();
		k = 0;
		totalAngle += theta;
		dx = InterpolationContest_.X_CENTER_PERTURBATION * Math.cos(totalAngle)
			- InterpolationContest_.Y_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ halfWidth * (1.0 - c) + halfHeight * s - shift;
		dy = InterpolationContest_.X_CENTER_PERTURBATION * Math.sin(totalAngle)
			+ InterpolationContest_.Y_CENTER_PERTURBATION * Math.cos(totalAngle)
			- halfWidth * s + halfHeight * (1.0 - c) - shift;
		for (int v = 0; (v < height); v++) {
			yx = dx - s * (double)v;
			yy = dy + c * (double)v;
			for (int u = 0; (u < width); u++) {
				x = yx + c * (double)u;
				y = yy + s * (double)u;
				p = (0.0 <= x) ? ((int)x) : ((int)x - 1);
				q = (0.0 <= y) ? ((int)y) : ((int)y - 1);
				if ((-1 <= p) && (p < width) && (-1 <= q) && (q < height)) {
					if (0 <= p) {
						xIndex1 = p++;
					}
					else {
						xIndex1 = ++p;
					}
					if (0 <= q) {
						yIndex1 = q++ * width;
					}
					else {
						yIndex1 = ++q;
					}
					if (p < width) {
						xIndex0 = p;
					}
					else {
						xIndex0 = --p;
					}
					if (q < height) {
						yIndex0 = q * width;
					}
					else {
						yIndex0 = --q * width;
					}
					x -= (0.0 <= x) ? ((int)x) : ((int)x - 1);
					y -= (0.0 <= y) ? ((int)y) : ((int)y - 1);
					xWeight0 = x;
					xWeight1 = 1.0 - x;
					partialResult = xWeight0 * (double)inImg[yIndex0 + xIndex0];
					partialResult += xWeight1 * (double)inImg[yIndex0 + xIndex1];
					result = y * partialResult;
					partialResult = xWeight0 * (double)inImg[yIndex1 + xIndex0];
					partialResult += xWeight1 * (double)inImg[yIndex1 + xIndex1];
					result += (1.0 - y) * partialResult;
					outImg[k++] = (float)result;
				}
				else {
					outImg[k++] = 0.0F;
				}
			}
		}
		float[] img = outImg;
		outImg = inImg;
		inImg = img;
		imp.updateAndDraw();
	}
} /* end shiftedLinear */

/*------------------------------------------------------------------*/
private void toCoefficients (
) {

/**/
	// ---
	// No clipping
	// ---
	final double b = 1.0 / (1.0 - shift);
	double c;
	int k = -1;
	for (int v = 0; (v < height); v++) {
		c = (double)inImg[++k];
		outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
		for (int u = 1; (u < width); u++) {
			c = (double)outImg[k++];
			outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
		}
	}
	for (int u = 0; (u < width); u++) {
		k = u;
		c = (double)inImg[k];
		outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
		for (int v = 1; (v < height); v++) {
			c = (double)outImg[k];
			k += width;
			outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
		}
	}
/**/

/*
	// ---
	// On-the-fly clipping
	// ---
	final double b = 1.0 / (1.0 - shift);
	double c;
	float left, right;
	int k = -1;
	for (int v = 0; (v < height); v++) {
		c = (double)inImg[++k];
		outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
		for (int u = 1; (u < (width - 1)); u++) {
			c = (double)outImg[k++];
			left = inImg[k];
			right = inImg[k + 1];
			outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
			if (left < right) {
				left -= EPSILON;
				right += EPSILON;
				outImg[k] = (outImg[k] < left) ? (left) : (outImg[k]);
				outImg[k] = (right < outImg[k]) ? (right) : (outImg[k]);
			}
			else {
				left += EPSILON;
				right -= EPSILON;
				outImg[k] = (outImg[k] < right) ? (right) : (outImg[k]);
				outImg[k] = (left < outImg[k]) ? (left) : (outImg[k]);
			}
		}
		c = (double)outImg[k++];
		outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
	}
	for (int u = 0; (u < width); u++) {
		k = u;
		c = (double)inImg[k];
		outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
		for (int v = 1; (v < (height - 1)); v++) {
			c = (double)outImg[k];
			k += width;
			left = outImg[k];
			right = outImg[k + 1];
			outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
			if (left < right) {
				left -= EPSILON;
				right += EPSILON;
				outImg[k] = (outImg[k] < left) ? (left) : (outImg[k]);
				outImg[k] = (right < outImg[k]) ? (right) : (outImg[k]);
			}
			else {
				left += EPSILON;
				right -= EPSILON;
				outImg[k] = (outImg[k] < right) ? (right) : (outImg[k]);
				outImg[k] = (left < outImg[k]) ? (left) : (outImg[k]);
			}
		}
		c = (double)outImg[k];
		k += width;
		outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
	}
*/

/*
	// ---
	// Deferred clipping
	// ---
	final double b = 1.0 / (1.0 - shift);
	double c;
	float topLeft, topRight, bottomLeft, bottomRight, swap;
	int k = -1;
	for (int v = 0; (v < height); v++) {
		c = (double)inImg[++k];
		outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
		for (int u = 1; (u < width); u++) {
			c = (double)inImg[k++];
			outImg[k] = (float)(b * ((double)inImg[k] - shift * c));
		}
	}
	for (int u = 0; (u < width); u++) {
		k = u;
		c = (double)inImg[k];
		outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
		for (int v = 1; (v < height); v++) {
			c = (double)inImg[k];
			k += width;
			outImg[k] = (float)(b * ((double)outImg[k] - shift * c));
		}
	}
	k = 0;
	for (int v = 0; (v < (height - 1)); v++) {
		for (int u = 0; (u < (width - 1)); u++, k++) {
			topLeft = inImg[k];
			topRight = inImg[k + 1];
			if (topRight < topLeft) {
				swap = topLeft;
				topLeft = topRight;
				topRight = swap;
			}
			bottomLeft = inImg[k + width];
			bottomRight = inImg[k + width + 1];
			if (bottomRight < bottomLeft) {
				swap = bottomLeft;
				bottomLeft = bottomRight;
				bottomRight = swap;
			}
			if (topLeft < bottomLeft) {
				bottomLeft = topLeft;
			}
			if (topRight < bottomRight) {
				topRight = bottomRight;
			}
			topRight += EPSILON;
			bottomLeft -= EPSILON;
			outImg[k] = (outImg[k] < bottomLeft) ? (bottomLeft) : (outImg[k]);
			outImg[k] = (topRight < outImg[k]) ? (topRight) : (outImg[k]);
		}
		k++;
	}
*/

	float[] img = outImg;
	outImg = inImg;
	inImg = img;

} /* toCoefficients */

} /* end class interpolationContestShiftedLinear */
