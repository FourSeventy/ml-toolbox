package regression

import (
	"errors"
	"math"
)

//LinearModel represents a linear regression model
type LinearModel struct {
	trained bool
	theta0  float64
	theta1  float64
}

//NewLinearModel constructs an untrained LinearModel
func NewLinearModel() *LinearModel {
	model := LinearModel{trained: false, theta0: 0, theta1: 0}
	return &model
}

//Train trains a linear model with the given trainingSet and step size
//TODO: better trainingSet type
//TODO: probably want to take a pointer
//TODO: better divergence detection
func (model *LinearModel) Train(trainingSet [][]float64, step float64) error {
	//pick a convergence threshold
	const threshold = .00001

	//perform gradient descent optimization
	oldTheta0 := model.theta0 + .002
	oldTheta1 := model.theta1 + .002
	for (math.Abs(oldTheta0-model.theta0) > threshold) && (math.Abs(oldTheta1-model.theta1) > threshold) {
		//calculate new theta 0
		var sum float64
		for _, example := range trainingSet {
			//for each of the test set examples, sum the difference of h(x) and y
			sum += model.calculateHypothesis(example[0]) - example[1]
		}
		tempTheta0 := model.theta0 - (step/float64(len(trainingSet)))*sum

		//calculate new theta 1
		sum = 0
		for _, example := range trainingSet {
			//for each of the test set examples, sum the difference of h(x) and y multiplied by x
			sum += (model.calculateHypothesis(example[0]) - example[1]) * example[0]
		}
		tempTheta1 := model.theta1 - (step/float64(len(trainingSet)))*sum

		//update old thetas
		oldTheta0 = model.theta0
		oldTheta1 = model.theta1

		//assign new thetas
		model.theta0 = tempTheta0
		model.theta1 = tempTheta1

		// fmt.Printf("Theta0: %v, Theta1: %v \n", model.theta0, model.theta1)
	}

	//check if a divergence happened
	if model.theta0 == math.NaN() || model.theta0 == math.Inf(1) || model.theta0 == math.Inf(-1) ||
		model.theta1 == math.NaN() || model.theta1 == math.Inf(1) || model.theta1 == math.Inf(-1) {
		return errors.New("training error: gradient descent has diverged, try a smaller step size")
	}

	//flag the model as trained
	model.trained = true

	//no error
	return nil
}

//RunHypothesis runs the hypothesis calculation on a trained LinearModel
func (model LinearModel) RunHypothesis(x float64) (float64, error) {
	if !model.trained {
		return 0, errors.New("error running hypothesis, model untrained")
	}

	hypothesis := model.calculateHypothesis(x)
	return hypothesis, nil
}

//calculateHypothesis runs the hypothesis calculation
func (model LinearModel) calculateHypothesis(x float64) float64 {
	y := model.theta0 + model.theta1*x
	return y
}

//MeanSquaredError returns the mean squared error of the model on a given dataSet
func (model LinearModel) MeanSquaredError(dataSet [][]float64) (float64, error) {
	return 0, errors.New("NYI")
}
