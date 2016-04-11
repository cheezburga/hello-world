package com.sli.deeplearning_experiment;
import java.util.HashMap;
import java.util.Map;

import org.jzy3d.plot3d.builder.Mapper;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.sli.linear_regression.Ex1;

public class CostMapper extends Mapper {
	private INDArray theX;
	private INDArray theY;
	private Map<String, Double> cached = new HashMap<String, Double>();
	public CostMapper(INDArray X, INDArray y){
		this.theX = X;
		this.theY = y;
	}
	@Override
	public double f(double theta0, double theta1) {
		String key = theta0+","+theta1;
		Double cache = cached.get(key);
		if(cache == null){
			cache = Ex1.computeCost(theX, theY, Nd4j.create(new double[]{theta0,theta1}).transpose());
			cached.put(key, cache);
		}
		return cache; 
	}

}
