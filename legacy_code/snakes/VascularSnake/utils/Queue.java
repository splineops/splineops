package plugins.big.vascular.utils;

import java.util.Stack;

/********************************************************************/
public class Queue 
{
	private Stack Input;
	private Stack Output;
	
/*------------------------------------------------------------------*/
	public Queue(){
		Input = new Stack();
		Output = new Stack();
	}
	
/*------------------------------------------------------------------*/
   	// adds an item to the back
   	public void enqueue(Object x){
   		Input.push(x);
   	}

/*------------------------------------------------------------------*/
   	// returns an item from the front;
   	public Object getFront(){
   		if(Output.empty())
   		{
   			while(!Input.empty())
   				Output.push(Input.pop());
   		}
   		return Output.peek();
   	}

/*------------------------------------------------------------------*/
   	// returns and removes an item from the front;
   	public Object dequeue(){
   		if(Output.empty())
   		{
   			while(!Input.empty())
   				Output.push(Input.pop());
   		}
   		return Output.pop();
   	}

/*------------------------------------------------------------------*/
   	// returns true if the queue is empty, otherwise false
   	public boolean isEmpty(){
   		return Input.empty() && Output.empty();
   	}

/*------------------------------------------------------------------*/
   	// removes all items from the queue.
   	public void clear(){
   		Input = new Stack();
   		Output = new Stack();
   	}

} // end class Queue