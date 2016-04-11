package com.sli.linear_regression;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Shape;
import java.awt.Stroke;
import java.io.IOException;
import java.util.Arrays;
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
import org.jfree.data.xy.XYCoordinate;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.jfree.util.ShapeUtilities;
import org.jzy3d.maths.Range;
import org.jzy3d.plot3d.builder.Mapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.inverse.InvertMatrix;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.springframework.core.io.ClassPathResource;

import com.sli.deeplearning_experiment.ContourPlot;
import com.sli.deeplearning_experiment.CostMapper;
import com.sli.deeplearning_experiment.SurfacePlot;

public class Ex1_multi {
	/**
	 * @param args
	 * @throws Exception 
	 */
	public static void main(String[] args) throws Exception {
		//================ Part 1: Feature Normalization ================
		System.out.println("Loading data ...\n");
		int totalSamples = 47;
		INDArray data = load("machine-learning/linear_regression/ex1data2.txt", totalSamples);
		INDArray x = data.getColumns(0,1).dup();
		INDArray y = data.getColumn(2).dup();
		System.out.println("First 10 examples from the dataset: \n");
		for(int i=0; i<10; i++){			
			System.out.println("x = "+x.getRows(i) +" y = "+y.getRow(i));
		}
		System.out.println("Normalizing Features ...\n");
		INDArray[] fn = featureNormalize(x);
		INDArray normalizedX = fn[0];
		INDArray mu = fn[1];
		System.out.println("mu:"+mu);
		INDArray sigma = fn[2];
		System.out.println("sigma:"+sigma);
		INDArray X = Nd4j.hstack(Nd4j.ones(totalSamples, 1),normalizedX);
		
		
		System.out.println("================ Part 2: Gradient Descent ================");		
		System.out.println("Running Gradient Descent ...\n");
		
		INDArray theta = Nd4j.zeros(3, 1);
		
		
		//Some gradient descent settings
		int iterations = 400;
		double alpha = 0.1;
		
		
	
		
		//run gradient descent
		INDArray[] results = gradientDescentMulti(X,y,theta, alpha, iterations);
		theta = results[0];
		INDArray jHistory = results[1];
		System.out.println("Theta found by gradient descent: ");
		System.out.println(theta);
		System.out.println(jHistory);
		
		
		
		plotData(iterations, jHistory);
		
		
		
		// Prediction
		double predict1 = Nd4j.hstack(Nd4j.ones(1, 1),Nd4j.create(new double[]{1650, 3}).subColumnVector(mu).divColumnVector(sigma)).mmul(theta).getDouble(0);
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): "+predict1);
		
		X = Nd4j.hstack(Nd4j.ones(totalSamples, 1),x);//no need to do feature normalization
		theta = normalEqn(X,y);
		System.out.println(theta);
		double predict2 = Nd4j.hstack(Nd4j.ones(1, 1),Nd4j.create(new double[]{1650, 3})).mmul(theta).getDouble(0);//no mu nor sigma
		System.out.println("Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) and normal eqn: "+predict2);
		
		
		
	}
	private static INDArray normalEqn(INDArray X, INDArray y){
		return InvertMatrix.invert(X.transpose().mmul(X), false).mmul(X.transpose()).mmul(y);
		
	}
	private static INDArray[] featureNormalize(INDArray X){
		INDArray mu = X.mean(0);
		System.out.println(mu);
		INDArray sigma = X.std(0);
		System.out.println(sigma);
		INDArray normX = X.subColumnVector(mu).divColumnVector(sigma);
		System.out.println(normX);
		return new INDArray[]{normX, mu, sigma};
	}
	
	
	private static INDArray[] gradientDescentMulti(INDArray X, INDArray y, INDArray theta, double alpha, int iterations){
		int m = y.size(0);
		INDArray jHistory = Nd4j.zeros(iterations, 1);
		for(int i=0; i<iterations; i++){
			INDArray secondTerm = X.mmul(theta).sub(y).mul((double) alpha/m);
			theta = theta.sub(X.transpose().mmul(secondTerm));			
			jHistory.getRow(i).assign(computeCostMulti(X, y, theta));
		}
		
		return new INDArray[]{theta,jHistory};
	}
	
	public static double computeCostMulti(INDArray X, INDArray y, INDArray theta){
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
	
	
	
	private static void pause(){
	   System.out.println("Program paused. Press enter to continue.\n");
	   Scanner scanner = new Scanner(System.in);
	   scanner.nextLine();	
	}

	private static JFrame plotData(int iteration, INDArray y){
		final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet,iteration,y,"Training Data");
      

        final JFreeChart chart = ChartFactory.createScatterPlot(
                " ",      // chart title
                "Number of iterations",                        // x axis label
                "Cost J", // y axis label
                dataSet,                    // data
                PlotOrientation.VERTICAL,
                true,                       // include legend
                true,                       // tooltips
                false                       // urls
        );

        final ChartPanel panel = new ChartPanel(chart);
        
        XYItemRenderer renderer = chart.getXYPlot().getRenderer();

        renderer.setSeriesPaint(0, Color.red);
        final JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();

        f.setVisible(true);
        return f;
	}
	private static void addSeries(final XYSeriesCollection dataSet, final int iteration, final INDArray y, final String label){
        final double[] yd = y.data().asDouble();
        final XYSeries s = new XYSeries(label);
        for( int j=0; j<iteration; j++ ) {
        	s.add(j,yd[j]);
        }
        dataSet.addSeries(s);
        
    }
}
