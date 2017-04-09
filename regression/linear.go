package regression

import (
	"errors"
	"fmt"
	"math"
)

//LinearModel represents a linear regression model
type LinearModel struct {
	trained    bool
	parameters []float64
}

//NewLinearModel constructs an untrained LinearModel
func NewLinearModel() *LinearModel {
	model := LinearModel{trained: false}
	return &model
}

//Dimensions returns the dimentionality of a LinearModel
//The dimentionality is defined as the size of the input vector that
//can be operated on
func (model *LinearModel) Dimensions() (int, error) {
	if model.trained == false {
		return 0, fmt.Errorf("LinearModel: untrained models have no dimentionality")
	}

	return len(model.parameters) - 1, nil
}

//Train trains a linear model with the given trainingSet and step size
//TODO: better trainingSet type
//TODO: better divergence detection
func (model *LinearModel) Train(trainingSet [][]float64, step float64) error {
	//pick a convergence threshold
	const threshold = .00001

	//initialize model parameters with the dimentionality of the training set
	model.parameters = make([]float64, len(trainingSet[0]))
	oldParameters := make([]float64, len(trainingSet[0]))
	tempParameters := make([]float64, len(trainingSet[0]))

	//perform gradient descent optimization
	for { //until convergence

		//for each parameter, calculate its new value
		for i := 0; i < len(model.parameters); i++ {
			//TODO: handle param 0
			if i == 0 {
				//calculate new parameter
				sum := 0.0
				for _, example := range trainingSet {
					//for each of the test set examples, sum the difference of h(x) and y multiplied by x
					sum += (model.calculateHypothesis(example[:len(example)-1]) - example[len(example)-1])
					tempParameters[i] = model.parameters[i] - (step/float64(len(trainingSet)))*sum
				}
			} else {
				//calculate new parameter
				sum := 0.0
				for _, example := range trainingSet {
					//for each of the test set examples, sum the difference of h(x) and y multiplied by x
					sum += (model.calculateHypothesis(example[:len(example)-1]) - example[len(example)-1]) * example[i-1]
					tempParameters[i] = model.parameters[i] - (step/float64(len(trainingSet)))*sum
				}
			}
		}

		//update old parameters, and assign new params
		for i := 0; i < len(model.parameters); i++ {
			oldParameters[i] = model.parameters[i]
			model.parameters[i] = tempParameters[i]
		}

		//check if we have convergence
		convergence := true
		for i := 0; i < len(model.parameters); i++ {
			if math.Abs(oldParameters[i]-model.parameters[i]) > threshold {
				convergence = false
				break
			}
		}

		//if we have convergence break out of algorithm
		if convergence {
			break
		}

		fmt.Println(model.parameters)
	}

	//check if a divergence happened
	for i := 0; i < len(model.parameters); i++ {
		if model.parameters[i] == math.NaN() || model.parameters[i] == math.Inf(1) || model.parameters[i] == math.Inf(-1) {
			return errors.New("training error: gradient descent has diverged, try a smaller step size")
		}
	}

	//flag the model as trained
	model.trained = true

	//no error
	return nil
}

//RunHypothesis runs the hypothesis calculation on a trained LinearModel
func (model LinearModel) RunHypothesis(x []float64) (float64, error) {
	//check for trained model
	if !model.trained {
		return 0, errors.New("error running hypothesis: model untrained")
	}

	//check for dimentionality match with model
	dim, _ := model.Dimensions() //swallow error because we already check for it above
	if dim != len(x) {
		return 0, fmt.Errorf("error running hypothesis: model (%v) and input vector (%v) dimentionality don't match",
			dim, len(x))
	}

	//run calculation
	hypothesis := model.calculateHypothesis(x)
	return hypothesis, nil
}

//calculateHypothesis runs the hypothesis calculation
//h(x) = \theta^T * X
func (model LinearModel) calculateHypothesis(x []float64) float64 {
	y := model.parameters[0]
	for i := 0; i < len(x); i++ {
		y += model.parameters[i+1] * x[i]
	}
	return y
}

//MeanSquaredError returns the mean squared error of the model on a given dataSet
//TODO: better input data type to handle random and sketchy assumptions about positions of data in the slice
func (model LinearModel) MeanSquaredError(dataSet [][]float64) (float64, error) {
	if !model.trained {
		return 0, errors.New("error running hypothesis, model untrained")
	}

	//calculate mse for data set
	var sum float64
	for _, example := range dataSet {
		sum += math.Pow(model.calculateHypothesis(example[:len(example)-1])-example[len(example)-1], 2)
	}
	sum = sum / float64(len(dataSet))

	return sum, nil
}
