package plugins.big.vascular.vascularGui;


import icy.plugin.abstract_.Plugin;
import icy.plugin.interface_.PluginImageAnalysis;
import icy.sequence.Sequence;

public class VascularSnake extends Plugin implements PluginImageAnalysis {

	public void compute() {
		Sequence sequence = getFocusedSequence();
		//MessageDialog.showDialog("BrainSegmentation is working fine!");
		
		Gui gui = new Gui(sequence);
	}

}
//petit test