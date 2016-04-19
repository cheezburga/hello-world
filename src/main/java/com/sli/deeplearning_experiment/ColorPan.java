package com.sli.deeplearning_experiment;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import javax.swing.JComponent;
import javax.swing.JFrame;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.ops.transforms.Transforms;

public class ColorPan extends JComponent {

	private INDArray x_100x400;
	public ColorPan(INDArray x_100x400){
		this.x_100x400 = x_100x400;
	
	}
	
	public void plot(){
		JFrame frame = new JFrame("ColorPan");
	    frame.getContentPane().add(this);
	    frame.setSize(221, 241);
	    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	    frame.setVisible(true);
	}
  public void paint(Graphics g) {
	  int m=100;
	  int n=400;
	  int example_width = 20;
	  int example_height = (n / example_width);

	  // Compute number of items to display
	  int display_rows = (int) Math.floor((Math.sqrt((double)m)));
	  int display_cols = (int)Math.ceil( (double)m/display_rows);
	  int pad = 1;
	  int width = pad+display_rows*(example_height+pad);
	  int height = pad+display_cols*(example_width+pad);
	  INDArray display_array = Nd4j.ones(width, height).neg();
	  
//	  INDArray debug = x_100x400.get(NDArrayIndex.point(0), NDArrayIndex.interval(0, n)).dup();
//	  System.out.println(debug);
//	  int l = 0;
//	  for(double k : debug.data().asDouble()){
//		  System.out.println("col"+(l++)+":"+k);
//	  }
//	  System.out.println(debug.reshape('A',20, 20));
//	  
	  int curr_ex = 0;
	  for(int j=0; j<display_rows; j++){
		  for(int i=0; i<display_cols; i++){
			  INDArray x_1x400 = x_100x400.get(NDArrayIndex.point(curr_ex++), NDArrayIndex.interval(0, n)).dup();
			  double max = Transforms.abs(x_1x400, true).maxNumber().doubleValue();
			  int j_patch_step = pad+j*(example_height+pad);
			  INDArrayIndex j_patch = NDArrayIndex.interval(j_patch_step, example_height+j_patch_step); 
			  System.out.println("j_patch range:"+j_patch_step+" to"+ (example_height+j_patch_step));
			  int i_patch_step = pad+i*(example_width+pad);
			  INDArrayIndex i_patch = NDArrayIndex.interval(i_patch_step, example_width+i_patch_step);
			  System.out.println("i_patch range:"+i_patch_step+" to"+ (example_width+i_patch_step));
			//reshape works differently in octave, java has to use transpose to get the octave behavior.  Java uses 'c' order while octave uses 'f'.
			  display_array.get(j_patch,i_patch).assign(x_1x400.reshape(example_height, example_width).transpose().div(max));
		  }
	  }
	  System.out.println(display_array);
	  
    double[] dataDouble = display_array.data().asDouble();
    int[] dataInt = new int[dataDouble.length];
    for(int i=0; i<dataDouble.length; i++){
    	int blue = new Integer(Math.round(0xFF*(dataDouble[i]+1.0)/2)+"");
    	int green = blue<<8;
    	int red = blue<<16;
    	dataInt[i] = red|green|blue;
    }

    
    BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    image.setRGB(0, 0, width, height, dataInt, 0, width);
    g.drawImage(image, 0, 0, this);
  }
  



}