package com.sli.linear_regression;
import java.awt.Color;
import java.awt.Shape;
import java.io.IOException;
import java.util.Scanner;

import javax.swing.JFrame;
import javax.swing.WindowConstants;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.util.ShapeUtilities;
import org.jzy3d.maths.Range;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.core.io.ClassPathResource;

import com.sli.deeplearning_experiment.ContourPlot;
import com.sli.deeplearning_experiment.CostMapper;
import com.sli.deeplearning_experiment.SurfacePlot;

public class Ex1 {
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		//==================== Part 1: Basic Function ====================
		System.out.println("Running warmUpExercise ... \n");
		System.out.println("5x5 Identity Matrix: \n");
		warmUpExercise();
		//pause();
		
		//======================= Part 2: Plotting =======================
		System.out.println("Plotting Data ...\n");
		int totalSamples = 97;
		INDArray data = load("machine-learning/linear_regression/ex1data1.txt", totalSamples);
		INDArray x = data.getColumn(0).dup();
		INDArray y = data.getColumn(1).dup();
	
		
		JFrame jFrame = plotData(x, y);
		//=================== Part 3: Gradient descent ===================
		System.out.println("Running Gradient Descent ...\n");
		INDArray X = Nd4j.hstack(Nd4j.ones(totalSamples, 1), x);
		INDArray theta = Nd4j.zeros(2, 1);
		
		
		//Some gradient descent settings
		int iterations = 1500;
		double alpha = 0.01;
		
		
		double j = computeCost(X,y, theta);
		System.out.println(j);
		
		//run gradient descent
		INDArray[] results = gradientDescent(X,y,theta, alpha, iterations);
		theta = results[0];
		INDArray jHistory = results[1];
		System.out.println("Theta found by gradient descent: ");
		System.out.println(theta);
		
		// Plot the linear fit
		
		plotData2(x, y, x, X.mmul(theta));
		
		
		// Predict values for population sizes of 35,000 and 70,000
		double predict1 = Nd4j.create(new double[]{1, 3.5}).mmul(theta).getDouble(0);
		System.out.println("For population = 35,000, we predict a profit of "+predict1*10000+",...");
		double predict2 = Nd4j.create(new double[]{1, 7}).mmul(theta).getDouble(0);
		System.out.println("For population = 70,000, we predict a profit of "+predict2*10000+",...");

		//============= Part 4: Visualizing J(theta_0, theta_1) =============
		System.out.println("Visualizing J(theta_0, theta_1) ...\n");
		//surf(X,y);
		contour(X,y, theta.getDouble(0), theta.getDouble(1));
	}
	
	private static void surf(INDArray X, INDArray y) throws Exception{
	      // Define a function to plot
       CostMapper mapper = new CostMapper(X, y);

        // Define range and precision for the function to plot
        Range rangeX = new Range(-10, 10);
        int stepX = 100;
        Range rangeY = new Range(-1,4);
        int stepY = 100;
        SurfacePlot sp = new SurfacePlot(rangeX, stepX, rangeY, stepY, mapper);
        sp.plot();
	}
	
	private static void contour(INDArray X, INDArray y, double theta0, double theta1) throws Exception{
	      // Define a function to plot
     CostMapper mapper = new CostMapper(X, y);

      // Define range and precision for the function to plot
      Range rangeX = new Range(-10, 10);
      int stepX = 150;
      Range rangeY = new Range(-1,4);
      int stepY = 150;
      ContourPlot sp = new ContourPlot(rangeX, stepX, rangeY, stepY, mapper, theta0, theta1);
      sp.plot();
	}
	private static INDArray[] gradientDescent(INDArray X, INDArray y, INDArray theta, double alpha, int iterations){
		int m = y.size(0);
		INDArray jHistory = Nd4j.zeros(m, 1);
		for(int i=0; i<iterations; i++){
			INDArray secondTerm = X.mmul(theta).sub(y).mul((double) alpha/m);
			theta = theta.sub(X.transpose().mmul(secondTerm));			
			jHistory.getRow(i).assign(computeCost(X, y, theta));
		}
		
		return new INDArray[]{theta,jHistory};
	}
	
	public static double computeCost(INDArray X, INDArray y, INDArray theta){
		int m = y.size(0);
		return Transforms.pow(X.mmul(theta).sub(y),2,true).sumNumber().doubleValue()/(2*m);
	}
	private static INDArray load(String filePath, int totalSamples) throws IOException, InterruptedException{
		
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(new ClassPathResource(filePath).getFile()));
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader);     
        DataSet set = iterator.next(totalSamples);      
        return set.getFeatureMatrix();
	}
	
	private static void warmUpExercise(){
		INDArray tmp = Nd4j.eye(5);
		System.out.println(tmp);
	}
	
	private static void pause(){
	   System.out.println("Program paused. Press enter to continue.\n");
	   Scanner scanner = new Scanner(System.in);
	   scanner.nextLine();	
	}

	private static JFrame plotData2(INDArray x, INDArray y, INDArray lrx, INDArray lry){
		final XYSeriesCollection dataSet = new XYSeriesCollection();
		final XYSeriesCollection dataSet2 = new XYSeriesCollection();
        addSeries(dataSet,x,y,"Training Data");
        addSeries(dataSet2,lrx,lry,"Linear regression");

        final JFreeChart chart = ChartFactory.createScatterPlot(
                " ",      // chart title
                "Population of City in 10,000s",                        // x axis label
                "Profit in $10,000s", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);
        XYPlot xyPlot = chart.getXYPlot();
        XYItemRenderer renderer1 = new XYLineAndShapeRenderer(false, true);//shapes
        Shape cross = ShapeUtilities.createDiagonalCross(3, 1);
        renderer1.setSeriesShape(0, cross);
        renderer1.setSeriesPaint(0, Color.red);
        xyPlot.setRenderer(0, renderer1);
        xyPlot.setDataset(0, dataSet);
        
        XYItemRenderer renderer2 = new XYLineAndShapeRenderer(true, false);//lines
        renderer2.setSeriesPaint(0, Color.blue);
        xyPlot.setRenderer(1, renderer2);
        xyPlot.setDataset(1, dataSet2);
        
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
        return f;
	}
	private static JFrame plotData(INDArray x, INDArray y){
		final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,x,y,"Training Data");
      

        final JFreeChart chart = ChartFactory.createScatterPlot(
                " ",      // chart title
                "Population of City in 10,000s",                        // x axis label
                "Profit in $10,000s", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);
        
        XYItemRenderer renderer = chart.getXYPlot().getRenderer();
        Shape cross = ShapeUtilities.createDiagonalCross(3, 1);
        renderer.setSeriesShape(0, cross);
        renderer.setSeriesPaint(0, Color.red);
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
        return f;
	}
	private static void addSeries(final XYSeriesCollection dataSet, final INDArray x, final INDArray y, final String label){
        final double[] xd = x.data().asDouble();
        final double[] yd = y.data().asDouble();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<xd.length; j++ ) {
        	s.add(xd[j],yd[j]);
        }
        dataSet.addSeries(s);
        
    }
}
