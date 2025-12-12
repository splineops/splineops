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
package plugins.big.bigsnake3d.keeper;

import icy.main.Icy;
import icy.roi.ROIEvent;
import icy.roi.ROIListener;
import icy.sequence.Sequence;
import icy.sequence.SequenceEvent;
import icy.sequence.SequenceListener;
import icy.util.StringUtil;

import java.util.ArrayList;
import java.util.Observable;
import java.util.Observer;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

import plugins.big.bigsnake3d.BIGSnake3DMSBrain;
import plugins.big.bigsnake3d.core.DisplaySettingsMSBrain;
import plugins.big.bigsnake3d.core.SettingsMSBrain;
import plugins.big.bigsnake3d.roi.ActionPlaneMSBrain;
import plugins.big.bigsnake3d.roi.ROI3DSnakeMSBrain;
import plugins.big.bigsnake3d.roi.SnakeEditModeMSBrain;
import plugins.big.bigsnake3d.snake.SphereSnakeMSBrain;
import plugins.big.bigsnake3d.snake.SphereSnakeParametersMSBrain;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DNode;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DOptimizer;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DPowellOptimizer;
import plugins.big.bigsnakeutils.icy.snake3D.Snake3DScale;

/**
 * Class that takes care of the optimization of the snake.
 * 
 * @version May 3, 2014
 * 
 * @author Nicolas Chenouard (nicolas.chenouard@gmail.com)
 * @author Ricard Delgado-Gonzalo (ricard.delgado@gmail.com)
 */
public class SnakeKeeperMSBrain implements ROIListener, Observer, SequenceListener {

	private final ArrayList<Thread> runningThreadList_ = new ArrayList<Thread>();

	private OptimizationThread optimizationThread_ = null;

	private final Lock isOptimizingLock_ = new ReentrantLock();
	private final Lock snakeLock_ = new ReentrantLock();
	private final Lock roiLock_ = new ReentrantLock();
	private final Lock screenUpdateLock = new ReentrantLock();

	private String ID = null;

	private final Condition updateCondition = screenUpdateLock.newCondition();

	/** The ROI managing control points. */
	protected ROI3DSnakeMSBrain roiNodes_ = null;

	/** The snake associated to this keeper. */
	private SphereSnakeMSBrain snake_ = null;

	/** Image where the snake lives. */
	private Sequence sequence_ = null;

	private boolean isRefreshDuringOptimization_ = SettingsMSBrain.REFRESH_SCREEN_DEFAULT;

	/** Pointer to the main class. */
	private BIGSnake3DMSBrain mainPlugin_ = null;

	private boolean isAttachedToSequence_ = false;
	private boolean isOptimizing_ = false;

	/** Update thread. */
	private final Thread updateThread_ = new Thread() {
		@Override
		public void run() {
			while (isAlive() && !this.isInterrupted()) {
				screenUpdateLock.lock();
				try {
					try {
						updateCondition.await();
						refreshViewerFromSnake();
					} catch (InterruptedException e) {
						e.printStackTrace();
					}
				} finally {
					screenUpdateLock.unlock();
				}
			}
		}
	};

	// ============================================================================
	// PUBLIC METHODS

	/** Default constructor. */
	public SnakeKeeperMSBrain(Sequence sequence, SphereSnakeMSBrain snake,
			BIGSnake3DMSBrain mainPlugin) {
		if (sequence == null) {
			System.err.println("sequence is null");
			return;
		}
		if (snake == null) {
			System.err.println("snake is null");
			return;
		}
		sequence_ = sequence;
		isAttachedToSequence_ = true;
		mainPlugin_ = mainPlugin;

		snakeLock_.lock();
		snake_ = snake;
		snakeLock_.unlock();

		roiLock_.lock();
		roiNodes_ = new ROI3DSnakeMSBrain(snake_, this);
		roiNodes_.setPixelSize(sequence.getPixelSizeX(),
				sequence.getPixelSizeY(), sequence.getPixelSizeZ());
		roiLock_.unlock();

		sequence.addROI(roiNodes_);
		sequence.addListener(this);

		refreshViewerFromSnake();

		roiNodes_.addListener(this);
		this.sequence_.roiChanged();
		updateThread_.start();
	}

	// ----------------------------------------------------------------------------

	public SphereSnakeParametersMSBrain getESnakeParameters() {
		return snake_.getSnakeParameters();
	}

	// ----------------------------------------------------------------------------

	public void activateSnake() {
		mainPlugin_.activateSnake(this);
	}

	// ----------------------------------------------------------------------------

	public Snake3DNode[] getNodesCopy() {
		Snake3DNode[] snakeNodes = snake_.getNodes();
		Snake3DNode[] snakeNodesCopy = new Snake3DNode[snakeNodes.length];
		for (int i = 0; i < snakeNodes.length; i++) {
			snakeNodesCopy[i] = new Snake3DNode(snakeNodes[i].x,
					snakeNodes[i].y, snakeNodes[i].z, snakeNodes[i].isFrozen(),
					snakeNodes[i].isHidden());
		}
		return snakeNodesCopy;
	}

	// ----------------------------------------------------------------------------

	public void rasterizeSnake() {
		Sequence binaryMask = snake_.getBinaryMask();
		Icy.getMainInterface().addSequence(binaryMask);
	}

	// ----------------------------------------------------------------------------

	/** Launches the optimization procedure in a new thread. */
	public void startOptimization() {
		if (isAttachedToSequence_) {
			isOptimizingLock_.lock();
			try {
				if (!isOptimizing_) {
					isOptimizing_ = true;
					optimizationThread_ = new OptimizationThread();
					optimizationThread_.setPriority(Thread.MIN_PRIORITY);
					optimizationThread_.start();
				}
			} finally {
				isOptimizingLock_.unlock();
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Stops the optimization procedure. */
	public void stopOptimization() {
		if (optimizationThread_ != null) {
			optimizationThread_.optimizer_.stopOptimizing();
		}
	}

	// ----------------------------------------------------------------------------

	public void removeFromSequence() {
		if (isAttachedToSequence_) {
			isAttachedToSequence_ = false;
			roiNodes_.delete3DPainters();
			sequence_.removeROI(roiNodes_);
			sequence_.removeListener(this);
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void roiChanged(ROIEvent event) {
		if (!isOptimizing_) {
			refreshSnakeFromViewer();
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void update(Observable observable, Object object) {
		if (observable instanceof Snake3DOptimizer) {
			if (((Snake3DOptimizer) observable).isCurrentBest) {
				refreshViewerFromSnakeThreaded();
			}
		}
	}

	// ----------------------------------------------------------------------------

	/** Applies a shift on the position of the snake. */
	public void shiftSnake(final int dx, final int dy, final int dz) {
		Thread thr = new Thread() {
			@Override
			public void run() {
				if (roiLock_.tryLock()) {
					try {
						roiNodes_.translate2D(dx, dy, dz);
					} finally {
						roiLock_.unlock();
					}
				}
			}
		};
		runningThreadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------

	public void setActionPlane(ActionPlaneMSBrain actionPlane) {
		roiLock_.lock();
		try {
			roiNodes_.setActionPlane(actionPlane);
		} finally {
			roiLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public void setDisplaySettings(final DisplaySettingsMSBrain displaySettings) {
		roiLock_.lock();
		try {
			switch (displaySettings.getMeshResolution()) {
			case HIGH:
				roiNodes_.setScaleSubsamplingFactor(1);
				break;
			case NORMAL:
				roiNodes_.setScaleSubsamplingFactor(2);
				break;
			case LOW:
				roiNodes_.setScaleSubsamplingFactor(4);
				break;
			}
			roiNodes_.setStrokeThicknessMultiplier(displaySettings
					.getStrokeThickness());
			isRefreshDuringOptimization_ = displaySettings.refresh();
			roiNodes_.setDisplayRange(displaySettings.getDepthTransparency());
		} finally {
			roiLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public void setEditingMode(final SnakeEditModeMSBrain editingMode) {
		roiLock_.lock();
		try {
			roiNodes_.setEditMode(editingMode);
		} finally {
			roiLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public void setSnakeEditMode(final SnakeEditModeMSBrain editingMode) {
		roiLock_.lock();
		try {
			roiNodes_.setEditMode(editingMode);
		} finally {
			roiLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	/** Updates the parameters of the snake. */
	public void setSnakeParameters(final SphereSnakeParametersMSBrain snakeParameters) {
		Thread thr = new Thread() {
			@Override
			public void run() {
				if (snakeLock_.tryLock()) {
					try {
						snake_.setSnakeParameters(snakeParameters);
					} finally {
						snakeLock_.unlock();
					}
				}
			}
		};
		runningThreadList_.add(thr);
		thr.start();
	}

	// ----------------------------------------------------------------------------

	public void setSelected(boolean selected) {
		roiNodes_.setEditable(selected);
		roiNodes_.painterChanged();
	}

	// ----------------------------------------------------------------------------

	public void setID(String id) {
		ID = id;
		roiLock_.lock();
		try {
			roiNodes_.setName(ID);
		} finally {
			roiLock_.unlock();
		}
	}

	// ----------------------------------------------------------------------------

	public String getID() {
		return ID;
	}

	// ----------------------------------------------------------------------------

	@Override
	public void sequenceChanged(SequenceEvent sequenceEvent) {
		if (sequenceEvent.getSequence() != this.sequence_) {
			return;
		}
		switch (sequenceEvent.getSourceType()) {
		case SEQUENCE_META:
			String metadataName = (String) sequenceEvent.getSource();
			if (StringUtil.equals(metadataName, Sequence.ID_PIXEL_SIZE_X)
					|| StringUtil
							.equals(metadataName, Sequence.ID_PIXEL_SIZE_Y)
					|| StringUtil
							.equals(metadataName, Sequence.ID_PIXEL_SIZE_Z)) {
				roiNodes_.setPixelSize(sequence_.getPixelSizeX(),
						sequence_.getPixelSizeY(), sequence_.getPixelSizeZ());
			}
			break;
		default:
			break;
		}
	}

	// ----------------------------------------------------------------------------

	@Override
	public void sequenceClosed(Sequence sequence) {
		if (sequence == sequence_) {
			removeFromSequence();
		}
	}

	// ============================================================================
	// PRIVATE METHODS

	public void refreshViewerFromSnake() {
		Snake3DNode[] nodes = null;
		Snake3DScale[] newScales = null;
		try {
			snakeLock_.lock();
			nodes = snake_.getNodes();
			newScales = snake_.getScales();
		} finally {
			snakeLock_.unlock();
		}
		if (nodes != null && newScales != null) {
			ArrayList<double[]> newPoints = new ArrayList<double[]>(
					nodes.length);
			for (Snake3DNode node : nodes) {
				double[] pos = new double[3];
				pos[0] = node.x;
				pos[1] = node.y;
				pos[2] = node.z;
				newPoints.add(pos);
			}
			try {
				roiLock_.lock();
				roiNodes_.changePosition(newPoints, newScales);
			} finally {
				roiLock_.unlock();
			}
		}
	}

	// ----------------------------------------------------------------------------

	private void refreshSnakeFromViewer() {
		if (isOptimizing_) {
			return;
		}
		if (snakeLock_.tryLock()) {
			try {
				if (roiLock_.tryLock()) {
					try {

						Snake3DNode[] nodes = new Snake3DNode[snake_.getNodes().length];
						for (int i = 0; i < nodes.length; i++) {
							nodes[i] = new Snake3DNode(
									roiNodes_.getControlPoint(i)[0],
									roiNodes_.getControlPoint(i)[1],
									roiNodes_.getControlPoint(i)[2]);
						}
						// recompute outline
						snake_.setNodes(nodes);

						// change the scales
						Snake3DScale[] newScales = snake_.getScales();
						roiNodes_.changeScales(newScales);
					} finally {
						roiLock_.unlock();
					}
				}
			} finally {
				snakeLock_.unlock();
			}
		}
	}

	// ----------------------------------------------------------------------------

	private void refreshViewerFromSnakeThreaded() {
		if (screenUpdateLock.tryLock()) {
			try {
				updateCondition.signalAll();
			} finally {
				screenUpdateLock.unlock();
			}
		}
	}

	// ----------------------------------------------------------------------------

	private class OptimizationThread extends Thread {

		public Snake3DOptimizer optimizer_ = new Snake3DPowellOptimizer();

		@Override
		public void run() {
			if (isOptimizingLock_.tryLock()) {
				try {
					roiNodes_.setEditable(false);
					snakeLock_.lock();
					final Snake3DNode[] youngSnake = snake_.getNodes();
					final int K = youngSnake.length;
					final Snake3DNode[] X = new Snake3DNode[K];
					for (int k = 0; (k < K); k++) {
						X[k] = new Snake3DNode(youngSnake[k].x,
								youngSnake[k].y, youngSnake[k].z,
								youngSnake[k].isFrozen(),
								youngSnake[k].isHidden());
					}
					snake_.setNodes(X);
					snakeLock_.unlock();

					if (isRefreshDuringOptimization_) {
						optimizer_.addObserver(SnakeKeeperMSBrain.this);
						isOptimizing_ = true;
						optimizer_.optimize(snake_, X);
						isOptimizing_ = false;
						snake_.reviveSnake();
						optimizer_.deleteObserver(SnakeKeeperMSBrain.this);
						roiNodes_.setEditable(true);
						refreshViewerFromSnakeThreaded();
					} else {
						isOptimizing_ = true;
						optimizer_.optimize(snake_, X);
						isOptimizing_ = false;
						snake_.reviveSnake();
						roiNodes_.setEditable(true);
						refreshViewerFromSnakeThreaded();
					}
				} finally {
					isOptimizingLock_.unlock();
				}
			}
		}
	}
}
