package proj2;

import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;


import java.util.Arrays;

public class App {

    AppSession appSession = new AppSession();
    Dataset<Row> result;
    Dataset<Row> validate;
    Dataset wineData;
    Dataset validationData;
    VectorAssembler vectorAssembler;
    LogisticRegressionTrainingSummary logisticRegressionTrainingSummary;



    public static void main(String [] args){

        App app = new App();
        app.connect();
        app.regression();
        app.result();

    }


    public void connect(){
        appSession.setSparkSession("spark://127.0.0.1:7077");

    }

    public void regression(){

        LogisticRegression logisticRegression = new LogisticRegression()
                .setFeaturesCol("features")
                .setRegParam(0.05)
                .setMaxIter(10000)
                .setLabelCol("quality");

        LogisticRegressionModel logModel = logisticRegression.fit(result);

        logModel.setThreshold(0.2);

        Dataset<Row> winePrediction = logModel.transform(validate).select("features", "quality", "prediction");

        winePrediction.show();

        LogisticRegressionModel logisticRegressionModel = logisticRegression.fit(result);
        logisticRegressionTrainingSummary = logisticRegressionModel.summary();

        wineData = appSession.sparkSession.read()
                .option("inferSchema", true)
                .option("header", true)
                .option("delimiter", ";")
                .csv("TrainingDataset.csv");

        validationData = appSession.sparkSession.read()
                .option("inferSchema", true)
                .option("header", true)
                .option("delimiter", ";")
                .csv("ValidationDataset.csv");

        vectorAssembler = new VectorAssembler()
                .setInputCols(new String[] {"fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"})
                .setOutputCol("features");


    }

    public void result(){
        result = vectorAssembler.transform(wineData).select("quality", "features");
        validate = vectorAssembler.transform(validationData).select("quality", "features");

        System.out.println("F Score:" + logisticRegressionTrainingSummary.labelCol().toString());
        System.out.print(Arrays.toString(logisticRegressionTrainingSummary.fMeasureByLabel())+ "\n");

    }

}
