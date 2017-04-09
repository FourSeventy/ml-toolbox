package main

import (
	"fmt"
	"github.com/fourseventy/ml-toolbox/dataset"
	"github.com/fourseventy/ml-toolbox/regression"
	"log"
	"os"
)

func main() {

	//build a linear model
	model := regression.NewLinearModel()

	//load training data
	trainingData, err := dataset.Load("testdata/univariate_linear.csv")
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	//train model
	err = model.Train(trainingData, 0.00001)
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	//load testing data
	testData, err := dataset.Load("testdata/univariate_linear2.csv")
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	//find the error against a test set of data
	mse, err := model.MeanSquaredError(testData)
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}

	fmt.Printf("MSE against test dataset is: %v", mse)

}
