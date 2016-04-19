package com.sli.deeplearning_experiment;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
//        INDArray tmp = Nd4j.create(new double[]{1,2,3,4}, new int[]{4,1});
//        System.out.println(tmp.reshape(2,2));
        
        INDArray nd2 = Nd4j.create(new float[]{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}/*, new int[]{2, 6}*/);
        
        System.out.println(nd2.ordering());
        System.out.println(nd2);
        INDArray tmp2 = nd2.dup('f').reshape(3,4);
        System.out.println(tmp2);
        System.out.println(tmp2.getRow(1).getColumn(0));

    }
}
